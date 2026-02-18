//! Audio normalization to mono 16k float32 in [-1, 1].

use anyhow::{bail, Result};

use crate::audio::{decode, input::AudioInput, resample, SAMPLE_RATE_HZ};

pub fn normalize_audio_input(input: &AudioInput<'_>) -> Result<Vec<f32>> {
    let (mut wav, sr) = match input {
        AudioInput::Waveform {
            samples,
            sample_rate,
        } => (samples.to_vec(), *sample_rate),
        AudioInput::Path(path) => decode::decode_path(path)?,
        AudioInput::Url(url) => decode::decode_url(url)?,
        AudioInput::Base64(b64) => decode::decode_base64(b64)?,
    };

    if wav.is_empty() {
        return Ok(wav);
    }

    // Conservative normalization to [-1, 1] for float inputs.
    let peak = wav
        .iter()
        .fold(0.0f32, |m, &x| if x.abs() > m { x.abs() } else { m });
    if peak.is_finite() && peak > 1.0 {
        for x in &mut wav {
            *x /= peak;
        }
    }
    for x in &mut wav {
        *x = (*x).clamp(-1.0, 1.0);
    }

    if sr == SAMPLE_RATE_HZ {
        return Ok(wav);
    }

    if sr == 0 {
        bail!("invalid sample_rate=0")
    }

    resample::resample_mono_f32(&wav, sr, SAMPLE_RATE_HZ)
}
