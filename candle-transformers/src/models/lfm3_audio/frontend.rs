//! LFM3's center=false periodic-Hann/Slaney log-mel frontend (mono 16 kHz).
use candle::{Device, Result, Tensor};
use rustfft::{num_complex::Complex32, FftPlanner};
const FFT: usize = 400;
const HOP: usize = 160;
const MELS: usize = 128;
pub fn log_mel(samples: &[f32], device: &Device) -> Result<Tensor> {
    if samples.iter().any(|v| !v.is_finite()) {
        candle::bail!("LFM3 audio must contain finite samples")
    }
    let frames = samples.len().saturating_sub(FFT).div_ceil(HOP) + 1;
    let mut padded = vec![0.; FFT + (frames - 1) * HOP];
    padded[..samples.len()].copy_from_slice(samples);
    let max_mel = 15. + (8f64).ln() / (6.4f64.ln() / 27.);
    let frequencies = (0..MELS + 2)
        .map(|i| {
            let m = i as f64 * max_mel / (MELS + 1) as f64;
            if m < 15. {
                m * 200. / 3.
            } else {
                1000. * ((m - 15.) * 6.4f64.ln() / 27.).exp()
            }
        })
        .collect::<Vec<_>>();
    let filters = frequencies
        .windows(3)
        .map(|b| {
            (0..FFT / 2 + 1)
                .map(|k| {
                    let hz = k as f64 * 16000. / FFT as f64;
                    (((hz - b[0]) / (b[1] - b[0]))
                        .min((b[2] - hz) / (b[2] - b[1]))
                        .max(0.)
                        * 2.
                        / (b[2] - b[0])) as f32
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let window = (0..FFT)
        .map(|i| 0.5 * (1. - (std::f32::consts::TAU * i as f32 / FFT as f32).cos()))
        .collect::<Vec<_>>();
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT);
    let mut values = vec![Complex32::default(); FFT];
    let mut scratch = vec![Complex32::default(); fft.get_inplace_scratch_len()];
    let mut features = vec![0.; MELS * frames];
    for frame in 0..frames {
        for (i, v) in values.iter_mut().enumerate() {
            *v = Complex32::new(padded[frame * HOP + i] * window[i], 0.);
        }
        fft.process_with_scratch(&mut values, &mut scratch);
        for (mel, filter) in filters.iter().enumerate() {
            let power = filter
                .iter()
                .zip(&values)
                .map(|(w, v)| w * v.norm_sqr())
                .sum::<f32>();
            features[mel * frames + frame] = (power.log10().max(-12.) + 4.) / 4.;
        }
    }
    Tensor::from_vec(features, (1, MELS, frames), device)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matches_torch_reference_and_boundaries() -> Result<()> {
        let samples = (0u16..960)
            .map(|i| (f32::from(i % 17) - 8.) / 8.)
            .collect::<Vec<_>>();
        let actual = log_mel(&samples, &Device::Cpu)?
            .squeeze(0)?
            .to_vec2::<f32>()?;
        for (m, f, v) in [
            (0, 0, -0.7189214),
            (1, 0, -0.62135696),
            (8, 0, -0.5910975),
            (32, 0, 0.27699572),
            (64, 1, -0.17709458),
            (100, 2, -0.55329394),
            (127, 4, 0.30260843),
        ] {
            assert!(
                (actual[m][f] - v).abs() < 2e-4,
                "{m},{f}: {} != {v}",
                actual[m][f]
            );
        }
        for (n, frames) in [(0, 1), (399, 1), (400, 1), (401, 2), (560, 2), (561, 3)] {
            let t = log_mel(&vec![0.; n], &Device::Cpu)?;
            assert_eq!(t.dims(), &[1, 128, frames]);
            assert!(t.flatten_all()?.to_vec1::<f32>()?.iter().all(|v| *v == -2.));
        }
        Ok(())
    }
}
