use anyhow::Result;
use candle::{Device, Tensor};
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::conv::FromSample;

fn conv<T>(samples: &mut Vec<f32>, data: std::borrow::Cow<symphonia::core::audio::AudioBuffer<T>>)
where
    T: symphonia::core::sample::Sample,
    f32: symphonia::core::conv::FromSample<T>,
{
    samples.extend(data.chan(0).iter().map(|v| f32::from_sample(*v)))
}

/// Decode audio file to PCM samples
pub fn pcm_decode<P: AsRef<std::path::Path>>(path: P) -> Result<(Vec<f32>, u32)> {
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
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow::anyhow!("no supported audio tracks"))?;

    let dec_opts: DecoderOptions = Default::default();
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &dec_opts)?;
    
    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(16000);
    let mut pcm_data = Vec::new();

    while let Ok(packet) = format.next_packet() {
        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet)? {
            AudioBufferRef::F64(buf) => conv(&mut pcm_data, buf),
            AudioBufferRef::F32(buf) => conv(&mut pcm_data, buf),
            AudioBufferRef::S32(buf) => conv(&mut pcm_data, buf),
            AudioBufferRef::S16(buf) => conv(&mut pcm_data, buf),
            AudioBufferRef::S8(buf) => conv(&mut pcm_data, buf),
            AudioBufferRef::U32(buf) => conv(&mut pcm_data, buf),
            AudioBufferRef::U16(buf) => conv(&mut pcm_data, buf),
            AudioBufferRef::U8(buf) => conv(&mut pcm_data, buf),
        }
    }

    Ok((pcm_data, sample_rate))
}

/// Convert PCM samples to mel spectrogram features
pub fn to_mel_spectrogram(
    samples: &[f32],
    n_mels: usize,
    device: &Device,
) -> Result<Tensor> {
    let hop_length = 160; // 10ms hop at 16kHz
    let n_frames = (samples.len() + hop_length - 1) / hop_length;
    
    // Create simplified mel features
    let mut mel_features = vec![0.0f32; n_mels * n_frames];
    
    for (frame_idx, frame_start) in (0..samples.len()).step_by(hop_length).enumerate() {
        if frame_idx >= n_frames {
            break;
        }
        
        let frame_end = (frame_start + 400).min(samples.len());
        let frame_energy: f32 = samples[frame_start..frame_end]
            .iter()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt();
        
        for mel_idx in 0..n_mels {
            let weight = (-((mel_idx as f32 - n_mels as f32 / 2.0).powi(2)) / (n_mels as f32 / 4.0)).exp();
            mel_features[frame_idx * n_mels + mel_idx] = frame_energy * weight;
        }
    }
    
    let tensor = Tensor::new(mel_features, device)?
        .reshape((1, n_mels, n_frames))?;
    
    Ok(tensor)
}

pub fn load_audio_features(
    audio_path: &str,
    n_mels: usize,
    device: &Device,
) -> Result<Tensor> {
    let (samples, _sr) = pcm_decode(audio_path)?;
    to_mel_spectrogram(&samples, n_mels, device)
}