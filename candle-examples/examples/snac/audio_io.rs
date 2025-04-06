use anyhow::{Context, Result};
use std::sync::{Arc, Mutex};

pub const SAMPLE_RATE: usize = 24_000;

pub(crate) struct AudioOutputData_ {
    resampled_data: std::collections::VecDeque<f32>,
    resampler: rubato::FastFixedIn<f32>,
    output_buffer: Vec<f32>,
    input_buffer: Vec<f32>,
    input_len: usize,
}

impl AudioOutputData_ {
    pub(crate) fn new(input_sample_rate: usize, output_sample_rate: usize) -> Result<Self> {
        use rubato::Resampler;

        let resampled_data = std::collections::VecDeque::with_capacity(output_sample_rate * 10);
        let resample_ratio = output_sample_rate as f64 / input_sample_rate as f64;
        let resampler = rubato::FastFixedIn::new(
            resample_ratio,
            f64::max(resample_ratio, 1.0),
            rubato::PolynomialDegree::Septic,
            1024,
            1,
        )?;
        let input_buffer = resampler.input_buffer_allocate(true).remove(0);
        let output_buffer = resampler.output_buffer_allocate(true).remove(0);
        Ok(Self {
            resampled_data,
            resampler,
            input_buffer,
            output_buffer,
            input_len: 0,
        })
    }

    pub fn reset(&mut self) {
        use rubato::Resampler;
        self.output_buffer.fill(0.);
        self.input_buffer.fill(0.);
        self.resampler.reset();
        self.resampled_data.clear();
    }

    pub(crate) fn take_all(&mut self) -> Vec<f32> {
        let mut data = Vec::with_capacity(self.resampled_data.len());
        while let Some(elem) = self.resampled_data.pop_back() {
            data.push(elem);
        }
        data
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.resampled_data.is_empty()
    }

    // Assumes that the input buffer is large enough.
    fn push_input_buffer(&mut self, samples: &[f32]) {
        self.input_buffer[self.input_len..self.input_len + samples.len()].copy_from_slice(samples);
        self.input_len += samples.len()
    }

    pub(crate) fn push_samples(&mut self, samples: &[f32]) -> Result<()> {
        use rubato::Resampler;

        let mut pos_in = 0;
        loop {
            let rem = self.input_buffer.len() - self.input_len;
            let pos_end = usize::min(pos_in + rem, samples.len());
            self.push_input_buffer(&samples[pos_in..pos_end]);
            pos_in = pos_end;
            if self.input_len < self.input_buffer.len() {
                break;
            }
            let (_, out_len) = self.resampler.process_into_buffer(
                &[&self.input_buffer],
                &mut [&mut self.output_buffer],
                None,
            )?;
            for &elem in self.output_buffer[..out_len].iter() {
                self.resampled_data.push_front(elem)
            }
            self.input_len = 0;
        }
        Ok(())
    }
}

type AudioOutputData = Arc<Mutex<AudioOutputData_>>;

pub(crate) fn setup_output_stream() -> Result<(cpal::Stream, AudioOutputData)> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    println!("Setup audio output stream!");
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .context("no output device available")?;
    let mut supported_configs_range = device.supported_output_configs()?;
    let config_range = match supported_configs_range.find(|c| c.channels() == 1) {
        // On macOS, it's commonly the case that there are only stereo outputs.
        None => device
            .supported_output_configs()?
            .next()
            .context("no audio output available")?,
        Some(config_range) => config_range,
    };
    let sample_rate = cpal::SampleRate(SAMPLE_RATE as u32).clamp(
        config_range.min_sample_rate(),
        config_range.max_sample_rate(),
    );
    let config: cpal::StreamConfig = config_range.with_sample_rate(sample_rate).into();
    let channels = config.channels as usize;
    println!(
        "cpal device: {} {} {config:?}",
        device.name().unwrap_or_else(|_| "unk".to_string()),
        config.sample_rate.0
    );
    let audio_data = Arc::new(Mutex::new(AudioOutputData_::new(
        SAMPLE_RATE,
        config.sample_rate.0 as usize,
    )?));
    let ad = audio_data.clone();
    let stream = device.build_output_stream(
        &config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            data.fill(0.);
            let mut ad = ad.lock().unwrap();
            let mut last_elem = 0f32;
            for (idx, elem) in data.iter_mut().enumerate() {
                if idx % channels == 0 {
                    match ad.resampled_data.pop_back() {
                        None => break,
                        Some(v) => {
                            last_elem = v;
                            *elem = v
                        }
                    }
                } else {
                    *elem = last_elem
                }
            }
        },
        move |err| eprintln!("cpal error: {err}"),
        None, // None=blocking, Some(Duration)=timeout
    )?;
    stream.play()?;
    Ok((stream, audio_data))
}

pub(crate) fn setup_input_stream() -> Result<(cpal::Stream, AudioOutputData)> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    println!("Setup audio input stream!");
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("no input device available")?;
    let mut supported_configs_range = device.supported_input_configs()?;
    let config_range = supported_configs_range
        .find(|c| c.channels() == 1)
        .context("no audio input available")?;
    let sample_rate = cpal::SampleRate(SAMPLE_RATE as u32).clamp(
        config_range.min_sample_rate(),
        config_range.max_sample_rate(),
    );
    let config: cpal::StreamConfig = config_range.with_sample_rate(sample_rate).into();
    println!(
        "cpal device: {} {} {config:?}",
        device.name().unwrap_or_else(|_| "unk".to_string()),
        config.sample_rate.0
    );
    let audio_data = Arc::new(Mutex::new(AudioOutputData_::new(
        config.sample_rate.0 as usize,
        SAMPLE_RATE,
    )?));
    let ad = audio_data.clone();
    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mut ad = ad.lock().unwrap();
            if let Err(err) = ad.push_samples(data) {
                eprintln!("error processing audio input {err:?}")
            }
        },
        move |err| eprintln!("cpal error: {err}"),
        None, // None=blocking, Some(Duration)=timeout
    )?;
    stream.play()?;
    Ok((stream, audio_data))
}

fn conv<T>(samples: &mut Vec<f32>, data: std::borrow::Cow<symphonia::core::audio::AudioBuffer<T>>)
where
    T: symphonia::core::sample::Sample,
    f32: symphonia::core::conv::FromSample<T>,
{
    use symphonia::core::audio::Signal;
    use symphonia::core::conv::FromSample;
    samples.extend(data.chan(0).iter().map(|v| f32::from_sample(*v)))
}

pub(crate) fn pcm_decode<P: AsRef<std::path::Path>>(path: P) -> Result<(Vec<f32>, u32)> {
    use symphonia::core::audio::{AudioBufferRef, Signal};

    let src = std::fs::File::open(path)?;
    let mss = symphonia::core::io::MediaSourceStream::new(Box::new(src), Default::default());
    let hint = symphonia::core::probe::Hint::new();
    let meta_opts: symphonia::core::meta::MetadataOptions = Default::default();
    let fmt_opts: symphonia::core::formats::FormatOptions = Default::default();
    let probed = symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts)?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .expect("no supported audio tracks");
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &Default::default())
        .expect("unsupported codec");
    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(0);
    let mut pcm_data = Vec::new();
    while let Ok(packet) = format.next_packet() {
        while !format.metadata().is_latest() {
            format.metadata().pop();
        }
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet)? {
            AudioBufferRef::F32(buf) => pcm_data.extend(buf.chan(0)),
            AudioBufferRef::U8(data) => conv(&mut pcm_data, data),
            AudioBufferRef::U16(data) => conv(&mut pcm_data, data),
            AudioBufferRef::U24(data) => conv(&mut pcm_data, data),
            AudioBufferRef::U32(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S8(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S16(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S24(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S32(data) => conv(&mut pcm_data, data),
            AudioBufferRef::F64(data) => conv(&mut pcm_data, data),
        }
    }
    Ok((pcm_data, sample_rate))
}

pub(crate) fn resample(pcm_in: &[f32], sr_in: usize, sr_out: usize) -> Result<Vec<f32>> {
    use rubato::Resampler;

    let mut pcm_out =
        Vec::with_capacity((pcm_in.len() as f64 * sr_out as f64 / sr_in as f64) as usize + 1024);

    let mut resampler = rubato::FftFixedInOut::<f32>::new(sr_in, sr_out, 1024, 1)?;
    let mut output_buffer = resampler.output_buffer_allocate(true);
    let mut pos_in = 0;
    while pos_in + resampler.input_frames_next() < pcm_in.len() {
        let (in_len, out_len) =
            resampler.process_into_buffer(&[&pcm_in[pos_in..]], &mut output_buffer, None)?;
        pos_in += in_len;
        pcm_out.extend_from_slice(&output_buffer[0][..out_len]);
    }

    if pos_in < pcm_in.len() {
        let (_in_len, out_len) = resampler.process_partial_into_buffer(
            Some(&[&pcm_in[pos_in..]]),
            &mut output_buffer,
            None,
        )?;
        pcm_out.extend_from_slice(&output_buffer[0][..out_len]);
    }

    Ok(pcm_out)
}
