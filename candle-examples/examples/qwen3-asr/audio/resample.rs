//! Resampling utilities.

use anyhow::{bail, Result};

#[cfg(feature = "audio-loading")]
use anyhow::Context;

#[cfg(feature = "audio-loading")]
pub fn resample_mono_f32(samples: &[f32], from_hz: u32, to_hz: u32) -> Result<Vec<f32>> {
    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };

    if from_hz == 0 || to_hz == 0 {
        bail!("invalid sample rates: from_hz={from_hz} to_hz={to_hz}");
    }
    if from_hz == to_hz {
        return Ok(samples.to_vec());
    }
    if samples.is_empty() {
        return Ok(vec![]);
    }

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let ratio = to_hz as f64 / from_hz as f64;
    let mut resampler =
        SincFixedIn::<f32>::new(ratio, 2.0, params, samples.len(), 1).map_err(|e| {
            anyhow::anyhow!("resampler creation failed for from_hz={from_hz} to_hz={to_hz}: {e}")
        })?;

    let waves_in = vec![samples.to_vec()];
    let mut waves_out = resampler
        .process(&waves_in, None)
        .context("resampling failed")?;

    waves_out
        .pop()
        .ok_or_else(|| anyhow::anyhow!("resampler returned no output channels"))
}

#[cfg(not(feature = "audio-loading"))]
pub fn resample_mono_f32(_samples: &[f32], _from_hz: u32, _to_hz: u32) -> Result<Vec<f32>> {
    bail!("resample_mono_f32 requires the `audio-loading` feature")
}

#[cfg(all(test, feature = "audio-loading"))]
mod tests {
    use super::resample_mono_f32;

    #[test]
    fn test_resample_identity() -> anyhow::Result<()> {
        let x = vec![0.0f32, 0.5, -0.25, 1.0];
        let y = resample_mono_f32(&x, 16_000, 16_000)?;
        if y != x {
            anyhow::bail!("identity resample changed samples");
        }
        Ok(())
    }
}
