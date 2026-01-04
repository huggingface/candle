//! HiFT STFT Implementation
//!
//! 16-point Short-Time Fourier Transform using matrix multiplication.
//! Used for processing the NSF source signal in HiFT Generator.

use candle::{DType, Device, Result, Tensor};
use std::f64::consts::PI;
use std::sync::OnceLock;

/// Pre-computed 16x16 DFT matrix
static DFT_MATRICES: OnceLock<(Vec<f32>, Vec<f32>)> = OnceLock::new();

fn get_dft_matrices() -> &'static (Vec<f32>, Vec<f32>) {
    DFT_MATRICES.get_or_init(|| {
        let n = 16usize;
        let mut real_matrix = vec![0.0f32; n * n];
        let mut imag_matrix = vec![0.0f32; n * n];

        let two_pi = 2.0 * PI;
        for k in 0..n {
            for m in 0..n {
                let angle = two_pi * (k * m) as f64 / n as f64;
                // DFT: W[k,m] = exp(-2πi*k*m/N) = cos - i*sin
                real_matrix[k * n + m] = angle.cos() as f32;
                imag_matrix[k * n + m] = (-angle.sin()) as f32; // Negative sign for DFT
            }
        }
        (real_matrix, imag_matrix)
    })
}

/// HiFT specific 16-point STFT implementation
#[derive(Debug, Clone)]
pub struct HiFTSTFT {
    n_fft: usize,
    hop_length: usize,
    dft_real: Tensor,
    dft_imag: Tensor,
    window: Tensor,
}

impl HiFTSTFT {
    /// Create a new STFT instance
    pub fn new(device: &Device, dtype: DType) -> Result<Self> {
        let n_fft = 16usize;
        let hop_length = 4usize;

        let (real_data, imag_data) = get_dft_matrices();
        let dft_real = Tensor::from_slice(real_data, (n_fft, n_fft), device)?.to_dtype(dtype)?;
        let dft_imag = Tensor::from_slice(imag_data, (n_fft, n_fft), device)?.to_dtype(dtype)?;

        let window = Self::create_hann_window(n_fft, device, dtype)?;

        Ok(Self {
            n_fft,
            hop_length,
            dft_real,
            dft_imag,
            window,
        })
    }

    fn create_hann_window(size: usize, device: &Device, dtype: DType) -> Result<Tensor> {
        let values: Vec<f32> = (0..size)
            .map(|i| (0.5 * (1.0 - (2.0 * PI * i as f64 / size as f64).cos())) as f32)
            .collect();
        Tensor::from_vec(values, size, device)?.to_dtype(dtype)
    }

    /// Perform STFT: waveform → (real, imag)
    ///
    /// # Arguments
    /// * `x` - [B, samples] Input waveform
    ///
    /// # Returns
    /// * `(real, imag)` - Each is [B, n_fft/2+1, n_frames] = [B, 9, n_frames]
    ///
    /// # Note
    /// This implementation uses center=True (PyTorch default), which pads n_fft//2
    /// on both sides of the input before computing STFT.
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let original_device = x.device().clone();
        let original_dtype = x.dtype();
        let (_batch, samples) = x.dims2()?;

        // Add center padding (n_fft // 2 on both sides) to match PyTorch's center=True
        let pad_size = self.n_fft / 2;
        let x = x.pad_with_zeros(1, pad_size, pad_size)?;
        let padded_samples = samples + 2 * pad_size;

        // Calculate number of frames
        if padded_samples < self.n_fft {
            candle::bail!(
                "Input samples {} is less than n_fft {}",
                padded_samples,
                self.n_fft
            );
        }
        let n_frames = (padded_samples - self.n_fft) / self.hop_length + 1;

        // Ensure input is on the same device as DFT matrices
        let x = x.to_device(self.dft_real.device())?.to_dtype(self.dft_real.dtype())?;

        // 1. Frame signal: [B, samples] -> [B, n_frames, n_fft]
        let frames = self.frame_signal(&x, n_frames)?;

        // 2. Apply window
        let window = self.window.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, 16]
        let windowed = (&frames * window.broadcast_as(frames.shape())?)?;

        // 3. DFT (matrix multiplication): [B, n_frames, 16] @ [16, 16] -> [B, n_frames, 16]
        // Use broadcast_matmul to support batch dimension
        let windowed = windowed.contiguous()?;
        let real = windowed.broadcast_matmul(&self.dft_real)?;
        let imag = windowed.broadcast_matmul(&self.dft_imag)?;

        // 4. Take one-sided spectrum: [B, n_frames, 16] -> [B, n_frames, 9]
        let real = real.narrow(2, 0, self.n_fft / 2 + 1)?;
        let imag = imag.narrow(2, 0, self.n_fft / 2 + 1)?;

        // 5. Transpose: [B, n_frames, 9] -> [B, 9, n_frames]
        let real = real.transpose(1, 2)?;
        let imag = imag.transpose(1, 2)?;

        // Convert back to original device and dtype
        let real = real.to_device(&original_device)?.to_dtype(original_dtype)?;
        let imag = imag.to_device(&original_device)?.to_dtype(original_dtype)?;

        Ok((real, imag))
    }

    fn frame_signal(&self, x: &Tensor, n_frames: usize) -> Result<Tensor> {
        let (_batch, _) = x.dims2()?;

        // Use narrow and stack to implement framing
        let frames: Vec<Tensor> = (0..n_frames)
            .map(|i| {
                let start = i * self.hop_length;
                x.narrow(1, start, self.n_fft)
            })
            .collect::<Result<Vec<_>>>()?;

        Tensor::stack(&frames, 1) // [B, n_frames, n_fft]
    }

    /// Compute magnitude and phase
    ///
    /// # Returns
    /// * `(magnitude, phase)` - Each is [B, 9, n_frames]
    pub fn forward_magnitude_phase(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let (real, imag) = self.forward(x)?;

        // magnitude = sqrt(real² + imag²)
        let magnitude = ((&real * &real)? + (&imag * &imag)?)?.sqrt()?;

        // phase = atan2(imag, real) - CPU implementation
        let phase = self.compute_atan2(&imag, &real)?;

        Ok((magnitude, phase))
    }

    /// Compute atan2(y, x) - CPU implementation
    fn compute_atan2(&self, y: &Tensor, x: &Tensor) -> Result<Tensor> {
        let y_vec: Vec<f32> = y.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let x_vec: Vec<f32> = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;

        let result: Vec<f32> = y_vec
            .iter()
            .zip(x_vec.iter())
            .map(|(&yi, &xi)| yi.atan2(xi))
            .collect();

        Tensor::from_vec(result, y.shape(), y.device())?.to_dtype(y.dtype())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stft_shape() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let stft = HiFTSTFT::new(&device, dtype)?;

        // Test input: [batch=2, samples=52]
        // n_frames = (52 - 16) / 4 + 1 = 10
        let x = Tensor::randn(0f32, 1.0, (2, 52), &device)?;

        let (real, imag) = stft.forward(&x)?;

        assert_eq!(real.dims(), &[2, 9, 10]);
        assert_eq!(imag.dims(), &[2, 9, 10]);
        Ok(())
    }

    #[test]
    fn test_dft_matrices() {
        let (real, imag) = get_dft_matrices();
        assert_eq!(real.len(), 256); // 16 * 16
        assert_eq!(imag.len(), 256);

        // First row should be all 1s (cos(0) = 1)
        for i in 0..16 {
            assert!((real[i] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_magnitude_phase() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let stft = HiFTSTFT::new(&device, dtype)?;

        let x = Tensor::randn(0f32, 1.0, (1, 52), &device)?;
        let (magnitude, phase) = stft.forward_magnitude_phase(&x)?;

        assert_eq!(magnitude.dims(), &[1, 9, 10]);
        assert_eq!(phase.dims(), &[1, 9, 10]);

        // Magnitude should be non-negative
        let mag_vec: Vec<f32> = magnitude.flatten_all()?.to_vec1()?;
        for v in mag_vec {
            assert!(v >= 0.0);
        }

        Ok(())
    }
}

