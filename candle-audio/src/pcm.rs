use crate::{Error, Result};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::VecDeque;

/// Interleaved f32 PCM. Sample rates and channel counts cannot be zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "RawFormat")]
pub struct AudioFormat {
    sample_rate: u32,
    channels: u16,
}

#[derive(Deserialize)]
struct RawFormat {
    sample_rate: u32,
    channels: u16,
}
impl TryFrom<RawFormat> for AudioFormat {
    type Error = Error;
    fn try_from(v: RawFormat) -> Result<Self> {
        Self::new(v.sample_rate, v.channels)
    }
}
impl AudioFormat {
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self> {
        if !(1..=384_000).contains(&sample_rate) || !(1..=64).contains(&channels) {
            return Err(Error::InvalidInput(
                "expected 1..384000 Hz and 1..64 channels".into(),
            ));
        }
        Ok(Self {
            sample_rate,
            channels,
        })
    }
    pub fn mono(sample_rate: u32) -> Result<Self> {
        Self::new(sample_rate, 1)
    }
    pub fn sample_rate(self) -> u32 {
        self.sample_rate
    }
    pub fn channels(self) -> u16 {
        self.channels
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AudioBuffer {
    format: AudioFormat,
    samples: Vec<f32>,
}
impl<'de> Deserialize<'de> for AudioBuffer {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Raw {
            format: AudioFormat,
            samples: Vec<f32>,
        }
        let raw = Raw::deserialize(deserializer)?;
        Self::new(raw.format, raw.samples).map_err(serde::de::Error::custom)
    }
}
impl AudioBuffer {
    pub fn new(format: AudioFormat, samples: Vec<f32>) -> Result<Self> {
        if !samples.len().is_multiple_of(format.channels as usize)
            || samples.iter().any(|x| !x.is_finite())
        {
            return Err(Error::InvalidInput(
                "PCM must contain complete interleaved frames and finite samples".into(),
            ));
        }
        Ok(Self { format, samples })
    }
    pub fn format(&self) -> AudioFormat {
        self.format
    }
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }
    pub fn into_samples(self) -> Vec<f32> {
        self.samples
    }
    pub fn frames(&self) -> usize {
        self.samples.len() / self.format.channels as usize
    }
    pub fn duration(&self) -> f64 {
        self.frames() as f64 / self.format.sample_rate as f64
    }
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// Stateful, band-limited conversion into a model's mono input rate. Input may
/// be arbitrarily chunked, including less than one resampler block. `finish`
/// drains delayed samples, trims padding, and returns exactly round(N*out/in)
/// frames. Conversion always averages all source channels, never just channel 0.
pub struct AudioConverter {
    input: AudioFormat,
    output: AudioFormat,
    resampler: Option<SincFixedIn<f32>>,
    pending: VecDeque<f32>,
    input_frames: u64,
    output_frames: u64,
    delay: usize,
    finished: bool,
}
impl AudioConverter {
    pub fn new(input: AudioFormat, output_rate: u32) -> Result<Self> {
        let output = AudioFormat::mono(output_rate)?;
        let resampler = if input.sample_rate == output_rate {
            None
        } else {
            let ratio = output_rate as f64 / input.sample_rate as f64;
            let params = SincInterpolationParameters {
                sinc_len: 256,
                f_cutoff: 0.95,
                interpolation: SincInterpolationType::Cubic,
                oversampling_factor: 128,
                window: WindowFunction::BlackmanHarris2,
            };
            Some(
                SincFixedIn::new(ratio, 1., params, 1024, 1)
                    .map_err(|e| Error::InvalidInput(e.to_string()))?,
            )
        };
        let delay = resampler.as_ref().map_or(0, |r| r.output_delay());
        Ok(Self {
            input,
            output,
            resampler,
            pending: VecDeque::new(),
            input_frames: 0,
            output_frames: 0,
            delay,
            finished: false,
        })
    }
    pub fn push(&mut self, audio: &AudioBuffer) -> Result<AudioBuffer> {
        if self.finished {
            return Err(Error::InvalidState("converter was finished".into()));
        }
        if audio.format != self.input {
            return Err(Error::InvalidInput(
                "audio format changed mid-stream".into(),
            ));
        }
        self.input_frames = self
            .input_frames
            .checked_add(audio.frames() as u64)
            .ok_or_else(|| Error::LimitExceeded("audio duration overflow".into()))?;
        let channels = self.input.channels as usize;
        let mono = audio
            .samples
            .chunks_exact(channels)
            .map(|frame| (frame.iter().map(|x| *x as f64).sum::<f64>() / channels as f64) as f32);
        if self.resampler.is_none() {
            self.output_frames += audio.frames() as u64;
            return AudioBuffer::new(self.output, mono.collect());
        }
        self.pending.extend(mono);
        let mut output = Vec::new();
        while self.pending.len() >= self.resampler.as_ref().unwrap().input_frames_next() {
            self.process_block(false, &mut output)?;
        }
        AudioBuffer::new(self.output, output)
    }
    fn process_block(&mut self, partial: bool, output: &mut Vec<f32>) -> Result<()> {
        let resampler = self.resampler.as_mut().unwrap();
        let n = self.pending.len().min(resampler.input_frames_next());
        let mut input: Vec<f32> = self.pending.drain(..n).collect();
        if partial {
            input.resize(resampler.input_frames_next(), 0.);
        }
        let result = resampler
            .process(&[input], None)
            .map_err(|e| Error::InvalidInput(e.to_string()))?;
        let samples = &result[0];
        let skip = self.delay.min(samples.len());
        self.delay -= skip;
        output.extend_from_slice(&samples[skip..]);
        self.output_frames += (samples.len() - skip) as u64;
        Ok(())
    }
    pub fn finish(&mut self) -> Result<AudioBuffer> {
        if self.finished {
            return Err(Error::InvalidState("converter was finished".into()));
        }
        let mut output = Vec::new();
        if self.resampler.is_some() {
            let expected = (self.input_frames as u128 * self.output.sample_rate as u128
                + self.input.sample_rate as u128 / 2)
                / self.input.sample_rate as u128;
            let before = self.output_frames;
            // Feed zero padding until the delayed end of the original signal is out.
            while self.output_frames < expected as u64 {
                self.process_block(true, &mut output)?;
            }
            output.truncate((expected as u64).saturating_sub(before) as usize);
            self.output_frames = expected as u64;
        }
        self.finished = true;
        AudioBuffer::new(self.output, output)
    }
    pub fn reset(&mut self) {
        if let Some(r) = self.resampler.as_mut() {
            r.reset();
            self.delay = r.output_delay();
        }
        self.pending.clear();
        self.input_frames = 0;
        self.output_frames = 0;
        self.finished = false;
    }
    pub fn convert(audio: &AudioBuffer, output_rate: u32) -> Result<AudioBuffer> {
        let mut converter = Self::new(audio.format, output_rate)?;
        let mut samples = converter.push(audio)?.into_samples();
        samples.extend(converter.finish()?.into_samples());
        AudioBuffer::new(converter.output, samples)
    }
}
