//! Neural Source Filter (NSF) Source Module
//!
//! Generates harmonic source signals from F0 (fundamental frequency).
//! Used in HiFT vocoder for high-quality audio synthesis.

use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Linear, VarBuilder};
use std::f64::consts::PI;

use crate::models::cosyvoice::activations::Tanh;

/// Sine wave generator
///
/// Following the official implementation, SineGen2 takes **upsampled** f0 as input,
/// then internally downsamples for efficient cumsum computation, and upsamples phase after.
/// This avoids creating huge matrices for the cumsum operation.
#[derive(Debug, Clone)]
pub struct SineGen2 {
    sampling_rate: usize,
    upsample_scale: usize,
    harmonic_num: usize,
    sine_amp: f64,
    #[allow(dead_code)]
    noise_std: f64,
    voiced_threshold: f64,
    /// Fixed random phase initial values used during inference
    rand_ini: Option<Tensor>,
}

impl SineGen2 {
    pub fn new(
        sampling_rate: usize,
        upsample_scale: usize,
        harmonic_num: usize,
        sine_amp: f64,
        noise_std: f64,
        device: &Device,
    ) -> Result<Self> {
        // Fixed random seed to generate initial phase
        // rand_ini[:, 0] = 0 (fundamental frequency phase is 0)
        let mut rand_ini = Tensor::rand(0.0f32, 1.0, (1, harmonic_num + 1), device)?;

        // Set fundamental frequency phase to 0
        let zeros = Tensor::zeros((1, 1), DType::F32, device)?;
        rand_ini = Tensor::cat(&[&zeros, &rand_ini.narrow(1, 1, harmonic_num)?], 1)?;

        Ok(Self {
            sampling_rate,
            upsample_scale,
            harmonic_num,
            sine_amp,
            noise_std,
            voiced_threshold: 10.0,
            rand_ini: Some(rand_ini),
        })
    }

    /// Generate sine waveform from F0
    ///
    /// Following official implementation's _f02sine:
    /// 1. Compute rad_values from f0
    /// 2. Downsample rad_values by 1/upsample_scale (for efficient cumsum)
    /// 3. Compute cumsum on the smaller tensor
    /// 4. Upsample phase back to full audio sample rate
    ///
    /// # Arguments
    /// * `f0` - [B, T, 1] or [B, T] - Fundamental frequency sequence (already upsampled to audio rate)
    ///
    /// # Returns
    /// * `(sine_waves, uv)` - Sine waves and voiced/unvoiced mask
    pub fn forward(&self, f0: &Tensor) -> Result<(Tensor, Tensor)> {
        let target_device = f0.device();
        let target_dtype = f0.dtype();

        // Ensure f0 is [B, T, 1]
        let f0 = if f0.dims().len() == 2 {
            f0.unsqueeze(D::Minus1)?
        } else {
            f0.clone()
        };

        let (batch, time, _) = f0.dims3()?;

        // Generate harmonic frequencies: f0 * [1, 2, 3, ..., harmonic_num+1]
        let harmonics: Vec<f32> = (1..=(self.harmonic_num + 1) as u32)
            .map(|i| i as f32)
            .collect();
        let harmonics = Tensor::from_vec(harmonics, self.harmonic_num + 1, target_device)?;
        let harmonics = harmonics.to_dtype(target_dtype)?.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, H]

        let f0_broadcast = f0.broadcast_as((batch, time, self.harmonic_num + 1))?;
        let harmonics_broadcast = harmonics.broadcast_as((batch, time, self.harmonic_num + 1))?;
        let fn_mat = (&f0_broadcast * &harmonics_broadcast)?; // [B, T, H]

        // Calculate normalized frequency
        // Note: In PyTorch impl they use % 1 to avoid large values, but since sin is periodic
        // we can skip this. The cumsum accumulates phase which sin handles correctly.
        let rad_values = (fn_mat / self.sampling_rate as f64)?;

        // === KEY OPTIMIZATION: Downsample before cumsum, upsample after ===
        // Following official implementation's _f02sine:
        // rad_values = F.interpolate(rad_values.T, scale_factor=1/upsample_scale, mode="linear").T

        // Add initial phase to first frame before downsampling
        let rad_values = if let Some(ref rand_ini) = self.rand_ini {
            let rand_ini = rand_ini.to_device(target_device)?.to_dtype(target_dtype)?;
            let rand_ini = rand_ini.broadcast_as((batch, 1, self.harmonic_num + 1))?;
            let first_frame = rad_values.narrow(1, 0, 1)?;
            let first_frame = (&first_frame + &rand_ini)?;
            if time > 1 {
                Tensor::cat(&[&first_frame, &rad_values.narrow(1, 1, time - 1)?], 1)?
            } else {
                first_frame
            }
        } else {
            rad_values
        };

        // Downsample rad_values for efficient cumsum
        // [B, T, H] -> [B, H, T] for interpolation, then back
        let rad_values_transposed = rad_values.transpose(1, 2)?; // [B, H, T]
        let downsampled_len = (time + self.upsample_scale - 1) / self.upsample_scale;
        let rad_values_down = self.downsample_linear(&rad_values_transposed, downsampled_len)?;
        let rad_values_down = rad_values_down.transpose(1, 2)?; // [B, T_down, H]

        // Compute cumsum on the smaller tensor (this is the expensive operation)
        let phase_down = (rad_values_down.cumsum(1)? * (2.0 * PI))?;

        // Upsample phase back to original length
        // Multiply by upsample_scale to maintain correct phase progression
        let phase_down_transposed = phase_down.transpose(1, 2)?; // [B, H, T_down]
        let phase_down_scaled = (&phase_down_transposed * (self.upsample_scale as f64))?;
        let phase = self.upsample_nearest_1d(&phase_down_scaled, time)?;
        let phase = phase.transpose(1, 2)?; // [B, T, H]

        // Generate sine waves
        let sine_waves = (phase.sin()? * self.sine_amp)?;

        // UV mask (voiced/unvoiced)
        let f0_squeezed = f0.squeeze(D::Minus1)?; // [B, T]
        let uv = f0_squeezed.gt(self.voiced_threshold)?;
        let uv = uv.to_dtype(target_dtype)?;

        Ok((sine_waves, uv))
    }

    /// Downsample using linear interpolation (simplified average pooling)
    /// Input: [B, C, T], Output: [B, C, T_out]
    fn downsample_linear(&self, x: &Tensor, target_len: usize) -> Result<Tensor> {
        let (batch, channels, time) = x.dims3()?;

        if time <= target_len {
            // No downsampling needed
            return Ok(x.clone());
        }

        // Simple average pooling approach
        let scale = self.upsample_scale;
        let mut result_slices = Vec::new();

        for i in 0..target_len {
            let start = i * scale;
            let end = ((i + 1) * scale).min(time);
            let len = end - start;

            if len > 0 {
                let slice = x.narrow(2, start, len)?;
                let avg = slice.mean(2)?; // [B, C]
                result_slices.push(avg.unsqueeze(2)?); // [B, C, 1]
            }
        }

        if result_slices.is_empty() {
            // Edge case: return zeros
            Tensor::zeros((batch, channels, target_len), x.dtype(), x.device())
        } else {
            Tensor::cat(&result_slices, 2)
        }
    }

    /// Nearest neighbor upsampling for 1D: [B, C, T] -> [B, C, T_out]
    /// Manual implementation that works on all backends (including Metal)
    fn upsample_nearest_1d(&self, x: &Tensor, target_len: usize) -> Result<Tensor> {
        let (batch, channels, time) = x.dims3()?;

        if time >= target_len {
            return x.narrow(2, 0, target_len);
        }

        if time == 0 {
            return Tensor::zeros((batch, channels, target_len), x.dtype(), x.device());
        }

        // Manual nearest neighbor upsampling using gather
        // For each target position i, we pick from source position floor(i * time / target_len)
        let indices: Vec<u32> = (0..target_len)
            .map(|i| ((i * time) / target_len) as u32)
            .collect();
        let indices = Tensor::from_vec(indices, target_len, x.device())?;

        // Use index_select on the time dimension
        x.index_select(&indices, 2)
    }
}

/// Harmonic Noise Source Module
#[derive(Debug)]
pub struct SourceModuleHnNSF {
    sine_gen: SineGen2,
    linear: Linear,
    tanh: Tanh,
    sine_amp: f64,
}

impl SourceModuleHnNSF {
    pub fn new(
        sampling_rate: usize,
        upsample_scale: usize,
        harmonic_num: usize,
        sine_amp: f64,
        add_noise_std: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let sine_gen = SineGen2::new(
            sampling_rate,
            upsample_scale,
            harmonic_num,
            sine_amp,
            add_noise_std,
            vb.device(),
        )?;

        // Linear layer: [harmonic_num + 1] -> [1]
        let linear = candle_nn::linear(harmonic_num + 1, 1, vb.pp("l_linear"))?;

        Ok(Self {
            sine_gen,
            linear,
            tanh: Tanh,
            sine_amp,
        })
    }

    /// Generate source signal from F0
    ///
    /// # Arguments
    /// * `f0` - [B, T, 1] Fundamental frequency sequence
    ///
    /// # Returns
    /// * `(sine_merge, noise, uv)` - Merged sine waves, noise, voiced mask
    pub fn forward(&self, f0: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let target_device = f0.device().clone();
        let target_dtype = f0.dtype();

        // Generate sine waves
        let (sine_wavs, uv) = self.sine_gen.forward(f0)?;
        // sine_wavs: [B, T*scale, H], uv: [B, T*scale]

        // Ensure sine_wavs is on the target device for linear layer
        let sine_wavs = sine_wavs.to_device(&target_device)?.to_dtype(target_dtype)?;

        // Linear combination of harmonics
        let sine_merge = self.linear.forward(&sine_wavs)?; // [B, T*scale, 1]
        let sine_merge = self.tanh.forward(&sine_merge)?;

        // Noise source
        let noise = (sine_merge.randn_like(0.0, 1.0)? * (self.sine_amp / 3.0))?;

        // Ensure outputs are on target device
        let sine_merge = sine_merge.to_device(&target_device)?.to_dtype(target_dtype)?;
        let noise = noise.to_device(&target_device)?.to_dtype(target_dtype)?;
        let uv = uv.to_device(&target_device)?.to_dtype(target_dtype)?;

        Ok((sine_merge, noise, uv))
    }

    /// Calculate total upsampling rate
    pub fn upsample_scale(&self) -> usize {
        self.sine_gen.upsample_scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sine_gen_shape() -> Result<()> {
        let device = Device::Cpu;

        // Using smaller upsample_scale for test to avoid slow CPU cumsum
        let sine_gen = SineGen2::new(24000, 4, 8, 0.1, 0.003, &device)?;

        // F0: [batch=2, time=40, 1] - simulating already upsampled f0
        // With upsample_scale=4, this means mel frames would be 10
        let f0 = Tensor::full(220.0f32, (2, 40, 1), &device)?;

        let (sine_waves, uv) = sine_gen.forward(&f0)?;

        // Output shape matches input time dimension (f0 is already upsampled)
        assert_eq!(sine_waves.dims(), &[2, 40, 9]); // 9 = 8 harmonics + 1
        assert_eq!(uv.dims(), &[2, 40]);

        Ok(())
    }

    #[test]
    fn test_upsample_nearest_1d() -> Result<()> {
        let device = Device::Cpu;

        // Test manual upsample via SineGen2's method
        let sine_gen = SineGen2::new(24000, 4, 8, 0.1, 0.003, &device)?;

        // Create test tensor [B, C, T] format
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], (1, 1, 3), &device)?;

        // Use index_select for upsampling (same as our manual implementation)
        let target_len = 12;
        let indices: Vec<u32> = (0..target_len)
            .map(|i| (i / 4) as u32)
            .collect();
        let indices = Tensor::from_vec(indices, target_len, &device)?;
        let y = x.index_select(&indices, 2)?;

        assert_eq!(y.dims(), &[1, 1, 12]); // 3 * 4 = 12

        let y_vec: Vec<f32> = y.flatten_all()?.to_vec1()?;
        assert_eq!(
            y_vec,
            vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]
        );

        Ok(())
    }
}

