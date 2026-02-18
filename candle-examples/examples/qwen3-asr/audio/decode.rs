//! Decoding audio bytes into a waveform.
//!
//! This is intentionally feature-gated: production ASR can accept paths/URLs/base64,
//! but for model/math bring-up you can start with in-memory waveform inputs only.

use anyhow::{bail, Result};
use std::path::Path;

#[cfg(feature = "audio-loading")]
use anyhow::Context;

#[cfg(feature = "audio-loading")]
pub fn decode_path(path: &Path) -> Result<(Vec<f32>, u32)> {
    use std::fs::File;

    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::probe::Hint;

    let file = File::open(path).with_context(|| format!("failed to open audio file {path:?}"))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
        hint.with_extension(ext);
    }

    let (samples, sr, channels) = decode_audio_stream(mss, hint)?;
    let mono = to_mono(&samples, channels)?;
    Ok((mono, sr))
}

#[cfg(not(feature = "audio-loading"))]
pub fn decode_path(_path: &Path) -> Result<(Vec<f32>, u32)> {
    bail!("decode_path requires the `audio-loading` feature")
}

#[cfg(feature = "audio-loading")]
pub fn decode_url(url: &str) -> Result<(Vec<f32>, u32)> {
    use std::io::Cursor;

    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::probe::Hint;

    let resp = reqwest::blocking::get(url)
        .with_context(|| format!("failed to fetch audio from URL {url:?}"))?;
    if !resp.status().is_success() {
        bail!("HTTP error fetching {url:?}: {}", resp.status());
    }

    let bytes = resp
        .bytes()
        .with_context(|| format!("failed to read response body for {url:?}"))?;
    let cursor = Cursor::new(bytes.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = url.rsplit('.').next() {
        let ext = ext.to_lowercase();
        if ["wav", "mp3", "flac", "ogg", "m4a", "aac", "opus", "webm"].contains(&ext.as_str()) {
            hint.with_extension(ext.as_str());
        }
    }

    let (samples, sr, channels) = decode_audio_stream(mss, hint)?;
    let mono = to_mono(&samples, channels)?;
    Ok((mono, sr))
}

#[cfg(not(feature = "audio-loading"))]
pub fn decode_url(_url: &str) -> Result<(Vec<f32>, u32)> {
    bail!("decode_url requires the `audio-loading` feature")
}

#[cfg(feature = "audio-loading")]
pub fn decode_base64(b64: &str) -> Result<(Vec<f32>, u32)> {
    use std::io::Cursor;

    use base64::Engine;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::probe::Hint;

    let data = if b64.contains(',') && b64.trim().starts_with("data:") {
        b64.split(',')
            .nth(1)
            .ok_or_else(|| anyhow::anyhow!("invalid data URL base64 format"))?
    } else {
        b64
    };

    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data)
        .map_err(|e| anyhow::anyhow!("base64 decode error: {e}"))?;

    let cursor = Cursor::new(bytes);
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());
    let (samples, sr, channels) = decode_audio_stream(mss, Hint::new())?;
    let mono = to_mono(&samples, channels)?;
    Ok((mono, sr))
}

#[cfg(not(feature = "audio-loading"))]
pub fn decode_base64(_b64: &str) -> Result<(Vec<f32>, u32)> {
    bail!("decode_base64 requires the `audio-loading` feature")
}

#[cfg(feature = "audio-loading")]
fn decode_audio_stream(
    mss: symphonia::core::io::MediaSourceStream,
    hint: symphonia::core::probe::Hint,
) -> Result<(Vec<f32>, u32, usize)> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::errors::Error as SymphoniaError;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::meta::MetadataOptions;

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| anyhow::anyhow!("failed to probe audio format: {e}"))?;

    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| anyhow::anyhow!("no audio tracks found"))?;

    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| anyhow::anyhow!("unknown sample rate"))?;
    let channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(1);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| anyhow::anyhow!("failed to create decoder: {e}"))?;

    let track_id = track.id;
    let mut samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(e) => return Err(anyhow::anyhow!("failed to read audio packet: {e}")),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => return Err(anyhow::anyhow!("failed to decode audio packet: {e}")),
        };

        let spec = *decoded.spec();
        let duration = decoded.capacity() as u64;
        let mut sample_buf = SampleBuffer::<f32>::new(duration, spec);
        sample_buf.copy_interleaved_ref(decoded);
        samples.extend_from_slice(sample_buf.samples());
    }

    Ok((samples, sample_rate, channels))
}

#[cfg(feature = "audio-loading")]
fn to_mono(samples: &[f32], channels: usize) -> Result<Vec<f32>> {
    if channels == 0 {
        bail!("invalid channel count: 0");
    }
    if channels == 1 {
        return Ok(samples.to_vec());
    }
    if !samples.len().is_multiple_of(channels) {
        bail!(
            "decoded sample length {} is not divisible by channels {}",
            samples.len(),
            channels
        );
    }

    let frames = samples.len() / channels;
    let mut mono: Vec<f32> = Vec::with_capacity(frames);
    for frame in samples.chunks_exact(channels) {
        let sum = frame.iter().copied().fold(0.0f32, |acc, x| acc + x);
        mono.push(sum / channels as f32);
    }
    Ok(mono)
}

#[cfg(all(test, feature = "audio-loading"))]
mod tests {
    use super::decode_path;

    #[test]
    fn test_decode_path_wav() -> anyhow::Result<()> {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
        let path_48k = root.join("fixtures").join("audio").join("asr_en.wav");
        let path_16k = root.join("fixtures").join("audio").join("asr_en_16k.wav");

        let (wav_48k, sr_48k) = decode_path(&path_48k)?;
        if sr_48k != 48_000 {
            anyhow::bail!("expected 48kHz wav, got sr={sr_48k}");
        }
        if wav_48k.is_empty() {
            anyhow::bail!("expected non-empty decode output for 48k wav");
        }

        let (wav_16k, sr_16k) = decode_path(&path_16k)?;
        if sr_16k != 16_000 {
            anyhow::bail!("expected 16kHz wav, got sr={sr_16k}");
        }
        if wav_16k.len() != 240_820 {
            anyhow::bail!("expected 240_820 samples, got {}", wav_16k.len());
        }

        Ok(())
    }
}
