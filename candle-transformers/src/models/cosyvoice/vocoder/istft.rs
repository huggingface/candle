//! HiFT iSTFT Implementation
//!
//! 16-point inverse Short-Time Fourier Transform using matrix multiplication.
//! This implementation is cross-platform (CPU/CUDA/Metal) as it uses only
//! basic tensor operations.

use candle::{DType, Device, Result, Tensor};
use std::f64::consts::PI;
use std::sync::OnceLock;

/// Pre-computed 16x16 IDFT matrix
static IDFT_MATRICES: OnceLock<(Vec<f32>, Vec<f32>)> = OnceLock::new();

fn get_idft_matrices() -> &'static (Vec<f32>, Vec<f32>) {
    IDFT_MATRICES.get_or_init(|| {
        let n = 16usize;
        let mut real_matrix = vec![0.0f32; n * n];
        let mut imag_matrix = vec![0.0f32; n * n];

        let two_pi = 2.0 * PI;
        for k in 0..n {
            for m in 0..n {
                let angle = two_pi * (k * m) as f64 / n as f64;
                // IDFT: W[k,n] = (1/N) * exp(2πi*k*n/N)
                real_matrix[k * n + m] = (angle.cos() / n as f64) as f32;
                imag_matrix[k * n + m] = (angle.sin() / n as f64) as f32;
            }
        }
        (real_matrix, imag_matrix)
    })
}

/// HiFT specific 16-point iSTFT implementation
///
/// This implementation is fully based on matrix operations, supporting CPU/CUDA/Metal cross-platform
#[derive(Debug, Clone)]
pub struct HiFTiSTFT {
    n_fft: usize,
    hop_length: usize,
    idft_real: Tensor,
    idft_imag: Tensor,
    window: Tensor,
}

impl HiFTiSTFT {
    /// Create a new iSTFT instance
    pub fn new(device: &Device, dtype: DType) -> Result<Self> {
        let n_fft = 16usize;
        let hop_length = 4usize;

        // Get pre-computed IDFT matrix
        let (real_data, imag_data) = get_idft_matrices();
        let idft_real = Tensor::from_slice(real_data, (n_fft, n_fft), device)?.to_dtype(dtype)?;
        let idft_imag = Tensor::from_slice(imag_data, (n_fft, n_fft), device)?.to_dtype(dtype)?;

        // Create Hann window (periodic=True, fftbins=True)
        let window = Self::create_hann_window(n_fft, device, dtype)?;

        Ok(Self {
            n_fft,
            hop_length,
            idft_real,
            idft_imag,
            window,
        })
    }

    fn create_hann_window(size: usize, device: &Device, dtype: DType) -> Result<Tensor> {
        let values: Vec<f32> = (0..size)
            .map(|i| (0.5 * (1.0 - (2.0 * PI * i as f64 / size as f64).cos())) as f32)
            .collect();
        Tensor::from_vec(values, size, device)?.to_dtype(dtype)
    }

    /// Perform iSTFT: magnitude + phase → waveform
    ///
    /// # Arguments
    /// * `magnitude` - [B, n_fft/2+1, T] = [B, 9, T], magnitude spectrum (already exp())
    /// * `phase` - [B, n_fft/2+1, T] = [B, 9, T], phase spectrum
    ///
    /// # Returns
    /// * `waveform` - [B, output_len]
    ///
    /// # Note
    /// This implementation matches PyTorch's istft with center=True (default).
    /// The output is trimmed by n_fft//2 on both ends to match PyTorch behavior.
    pub fn forward(&self, magnitude: &Tensor, phase: &Tensor) -> Result<Tensor> {
        let (batch, freq_bins, n_frames) = magnitude.dims3()?;
        assert_eq!(freq_bins, self.n_fft / 2 + 1); // 9

        let original_device = magnitude.device().clone();
        let dtype = magnitude.dtype();

        // Execute on CPU to avoid contiguous issues
        let magnitude = magnitude.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let phase = phase.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let idft_real = self.idft_real.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let idft_imag = self.idft_imag.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let window = self.window.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;

        // 1. Clip magnitude to prevent numerical overflow
        let magnitude = magnitude.clamp(0.0, 100.0)?;

        // 2. Build complex: real = mag * cos(phase), imag = mag * sin(phase)
        let real = (&magnitude * phase.cos()?)?;
        let imag = (&magnitude * phase.sin()?)?;

        // 3. Build full spectrum (conjugate symmetric): [B, 9, T] -> [B, 16, T]
        // For real signals, the full spectrum has conjugate symmetry:
        // - Real part is symmetric: X_r[k] = X_r[N-k]
        // - Imag part is anti-symmetric: X_i[k] = -X_i[N-k] (only for mirrored bins)
        // - DC (k=0) and Nyquist (k=8) should have zero imaginary part
        let real_full = self.build_full_spectrum_cpu(&real, false, false)?;
        let imag_full = self.build_full_spectrum_cpu(&imag, true, true)?; // negate mirrored, zero DC/Nyquist

        // 4. Transpose to [B, T, 16] for matrix multiplication
        let real_full = real_full.transpose(1, 2)?.contiguous()?;
        let imag_full = imag_full.transpose(1, 2)?.contiguous()?;

        // 5. 16-point IDFT (matrix multiplication implementation)
        let frames = (real_full.broadcast_matmul(&idft_real)? - imag_full.broadcast_matmul(&idft_imag)?)?;

        // 6. Apply window
        let window = window.unsqueeze(0)?.unsqueeze(0)?;
        let windowed_frames = (&frames * window.broadcast_as(frames.shape())?)?;

        // 7. Overlap-Add
        let output = self.overlap_add(&windowed_frames, batch, n_frames)?;

        // 8. Trim edges to match PyTorch's istft with center=True (default)
        // PyTorch adds n_fft//2 padding on both ends during STFT with center=True,
        // and removes it during iSTFT. We need to trim the same amount.
        let trim = self.n_fft / 2; // 8 for n_fft=16
        let output_len = output.dim(1)?;
        let trimmed_output = if output_len > 2 * trim {
            output.narrow(1, trim, output_len - 2 * trim)?
        } else {
            output
        };

        // Convert back to original device
        trimmed_output.to_device(&original_device)?.to_dtype(dtype)
    }

    /// CPU version of build_full_spectrum, simplified to avoid contiguous issues
    ///
    /// For imaginary part with negate_mirror=true, negate only the mirrored bins (9-15)
    /// to maintain proper conjugate symmetry for real signals.
    /// If zero_dc_nyquist=true, set DC (bin 0) and Nyquist (bin 8) to zero.
    fn build_full_spectrum_cpu(
        &self,
        half: &Tensor,
        negate_mirror: bool,
        zero_dc_nyquist: bool,
    ) -> Result<Tensor> {
        // half: [B, 9, T] -> full: [B, 16, T]
        let half = half.contiguous()?;
        // Mirror section: reverse bins 1-7 -> indices [7, 6, 5, 4, 3, 2, 1]
        let (batch, _, time) = half.dims3()?;

        // Manually build mirror
        let half_data: Vec<f32> = half.flatten_all()?.to_vec1()?;
        let mut full_data = vec![0.0f32; batch * 16 * time];

        for b in 0..batch {
            for t in 0..time {
                // Copy first 9 bins (unchanged, except DC and Nyquist for imag)
                for f in 0..9 {
                    let val = half_data[b * 9 * time + f * time + t];
                    // For imaginary part: DC (f=0) and Nyquist (f=8) must be zero
                    full_data[b * 16 * time + f * time + t] =
                        if zero_dc_nyquist && (f == 0 || f == 8) {
                            0.0
                        } else {
                            val
                        };
                }
                // Mirror last 7 bins (bins 9-15 = reverse of bins 1-7)
                // For imaginary part: negate to maintain conjugate symmetry
                let sign = if negate_mirror { -1.0 } else { 1.0 };
                for i in 0..7 {
                    let src_f = 7 - i; // 7, 6, 5, 4, 3, 2, 1
                    let dst_f = 9 + i; // 9, 10, 11, 12, 13, 14, 15
                    full_data[b * 16 * time + dst_f * time + t] =
                        sign * half_data[b * 9 * time + src_f * time + t];
                }
            }
        }

        Tensor::from_vec(full_data, (batch, 16, time), half.device())
    }

    #[allow(dead_code)]
    fn build_full_spectrum(&self, half: &Tensor, conjugate: bool) -> Result<Tensor> {
        // half: [B, 9, T] -> full: [B, 16, T]
        // Ensure half is contiguous
        let half = half.contiguous()?;
        // Mirror section: reverse bins 1-7
        let mirror = half.narrow(1, 1, 7)?.flip(&[1])?.contiguous()?;
        let mirror = if conjugate { mirror.neg()? } else { mirror };
        Tensor::cat(&[&half, &mirror], 1)?.contiguous()
    }

    #[allow(clippy::needless_range_loop)]
    fn overlap_add(&self, frames: &Tensor, batch: usize, n_frames: usize) -> Result<Tensor> {
        // frames: [B, T, n_fft]
        let output_len = (n_frames - 1) * self.hop_length + self.n_fft;
        let dtype = frames.dtype();
        let device = frames.device();

        // Execute overlap-add on CPU (simplified implementation for compatibility)
        let frames_cpu = frames.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let frames_vec: Vec<f32> = frames_cpu.flatten_all()?.to_vec1()?;

        let mut output_data = vec![0.0f32; batch * output_len];

        for b in 0..batch {
            for i in 0..n_frames {
                let start = i * self.hop_length;
                for j in 0..self.n_fft {
                    let frame_idx = b * n_frames * self.n_fft + i * self.n_fft + j;
                    let out_idx = b * output_len + start + j;
                    if out_idx < batch * output_len {
                        output_data[out_idx] += frames_vec[frame_idx];
                    }
                }
            }
        }

        // Window function normalization (COLA condition)
        let window_sum = self.compute_window_sum(n_frames, output_len, &Device::Cpu)?;
        let window_data: Vec<f32> = window_sum.to_vec1()?;

        for b in 0..batch {
            for i in 0..output_len {
                let idx = b * output_len + i;
                output_data[idx] /= window_data[i];
            }
        }

        Tensor::from_vec(output_data, (batch, output_len), &Device::Cpu)?
            .to_device(device)?
            .to_dtype(dtype)
    }

    #[allow(clippy::needless_range_loop)]
    fn compute_window_sum(&self, n_frames: usize, output_len: usize, device: &Device) -> Result<Tensor> {
        let window_data: Vec<f32> = self.window.to_dtype(DType::F32)?.to_vec1()?;

        let mut sum = vec![0.0f32; output_len];
        for i in 0..n_frames {
            let start = i * self.hop_length;
            for j in 0..self.n_fft {
                if start + j < output_len {
                    sum[start + j] += window_data[j] * window_data[j];
                }
            }
        }

        // Avoid division by zero
        for v in sum.iter_mut() {
            if *v < 1e-8 {
                *v = 1.0;
            }
        }

        Tensor::from_vec(sum, output_len, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_istft_shape() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let istft = HiFTiSTFT::new(&device, dtype)?;

        // Test input: [batch=2, freq_bins=9, frames=10]
        let magnitude = Tensor::ones((2, 9, 10), dtype, &device)?;
        let phase = Tensor::zeros((2, 9, 10), dtype, &device)?;

        let output = istft.forward(&magnitude, &phase)?;

        // Output length: (10 - 1) * 4 + 16 = 52, then trim n_fft/2 (8) from both ends
        // Final: 52 - 16 = 36
        assert_eq!(output.dims(), &[2, 36]);
        Ok(())
    }

    #[test]
    fn test_idft_matrices() {
        let (real, imag) = get_idft_matrices();
        assert_eq!(real.len(), 256); // 16 * 16
        assert_eq!(imag.len(), 256);

        // First row should be all 1/16
        for i in 0..16 {
            assert!((real[i] - 1.0 / 16.0).abs() < 1e-6);
        }
    }
}

