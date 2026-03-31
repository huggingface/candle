//! PolarQuant-compressed Neural Network Layers
//!
//! Provides drop-in replacements for standard neural network layers that store
//! weights in PolarQuant-compressed form. Weights are dequantized on-the-fly
//! during the forward pass, reducing model memory footprint.
//!
//! # Compression
//!
//! At 4-bit quantization with dim=128:
//! - Memory per weight row: 128 bytes (indices) + 4 bytes (norm) = 132 bytes
//! - vs F32: 512 bytes → **3.9× compression**
//! - Relative MSE ≈ 0.009
//!
//! # Supported layers
//!
//! - [`PolarQuantLinear`]: Drop-in replacement for [`Linear`](crate::Linear).
//!   Quantize pretrained weights with [`from_linear`](PolarQuantLinear::from_linear),
//!   or construct directly from quantized data.
//!
//! # Example
//!
//! ```no_run
//! use candle::{Device, DType, Tensor};
//! use candle_nn::{Linear, Module};
//! use candle_nn::polarquant_nn::PolarQuantLinear;
//!
//! let device = Device::Cpu;
//! // Start with a pretrained Linear layer
//! let weight = Tensor::randn(0f32, 1f32, (64, 128), &device).unwrap();
//! let linear = Linear::new(weight, None);
//!
//! // Compress to 4-bit PolarQuant
//! let pq_linear = PolarQuantLinear::from_linear(&linear, 4, &device).unwrap();
//!
//! // Same forward API
//! let x = Tensor::randn(0f32, 1f32, (1, 128), &device).unwrap();
//! let y = pq_linear.forward(&x).unwrap();
//! assert_eq!(y.dims(), &[1, 64]);
//! ```

use candle::{DType, Device, Result, Tensor};

use crate::polarquant::{PolarQuant, PolarQuantized};
use crate::Linear;

/// Linear layer with PolarQuant-compressed weights.
///
/// Stores the weight matrix as quantized indices + norms, dequantizing on each
/// forward pass. This trades compute for memory — the weight matrix is never
/// materialized in full precision until needed.
///
/// # When to Use
///
/// **Recommended for:**
/// - Post-training compression of large models
/// - Memory-constrained inference (edge, mobile, large batch)
///
/// **Use [`Linear`](crate::Linear) instead for:**
/// - Training (gradients require full-precision weights)
/// - Models small enough to fit in memory uncompressed
#[derive(Debug, Clone)]
pub struct PolarQuantLinear {
    quantizer: PolarQuant,
    indices: Tensor,
    norms: Tensor,
    bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
}

impl PolarQuantLinear {
    /// Quantize a pretrained [`Linear`] layer into PolarQuant-compressed form.
    ///
    /// The weight matrix (shape `[out_features, in_features]`) is quantized
    /// row-by-row. Each row becomes a vector of U8 centroid indices plus a
    /// scalar norm. The bias (if present) is kept in full precision.
    ///
    /// # Arguments
    /// * `linear` - The pretrained linear layer to compress.
    /// * `bit_width` - Quantization bits per coordinate (typically 2–4).
    /// * `device` - Device for the quantizer's rotation matrix.
    ///
    /// # Example
    /// ```ignore
    /// let pq = PolarQuantLinear::from_linear(&pretrained_layer, 4, &device)?;
    /// ```
    pub fn from_linear(linear: &Linear, bit_width: usize, device: &Device) -> Result<Self> {
        let weight = linear.weight();
        let (out_features, in_features) = weight.dims2()?;

        let dtype = weight.dtype();
        let internal_dtype = if dtype == DType::F64 {
            DType::F64
        } else {
            DType::F32
        };
        let quantizer = PolarQuant::new(in_features, bit_width, internal_dtype, device)?;
        let q = quantizer.quantize(weight)?;

        Ok(Self {
            quantizer,
            indices: q.indices.to_dtype(DType::U8)?,
            norms: q.norms,
            bias: linear.bias().cloned(),
            in_features,
            out_features,
        })
    }

    /// Create a PolarQuantLinear from pre-quantized data.
    ///
    /// # Arguments
    /// * `quantizer` - The PolarQuant quantizer (holds rotation matrix + centroids).
    /// * `indices` - Centroid indices, shape `[out_features, in_features]`, dtype U8.
    /// * `norms` - Row norms, shape `[out_features]`.
    /// * `bias` - Optional bias, shape `[out_features]`.
    pub fn new(
        quantizer: PolarQuant,
        indices: Tensor,
        norms: Tensor,
        bias: Option<Tensor>,
    ) -> Result<Self> {
        let (out_features, in_features) = indices.dims2()?;
        Ok(Self {
            quantizer,
            indices,
            norms,
            bias,
            in_features,
            out_features,
        })
    }

    /// Dequantize and return the full-precision weight matrix.
    ///
    /// Shape: `[out_features, in_features]`.
    pub fn dequantize_weight(&self) -> Result<Tensor> {
        let q = PolarQuantized {
            indices: self.indices.to_dtype(DType::U32)?,
            norms: self.norms.clone(),
        };
        self.quantizer.dequantize(&q)
    }

    /// Get the bias tensor, if present.
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    /// Number of input features.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Number of output features.
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Get the bit width used for quantization.
    pub fn bit_width(&self) -> usize {
        self.quantizer.bit_width()
    }

    /// Approximate memory usage of the quantized weights in bytes.
    ///
    /// This counts index storage (U8) + norm storage (F32), excluding the
    /// shared rotation matrix and centroids.
    pub fn quantized_weight_bytes(&self) -> usize {
        // U8 indices: out_features * in_features bytes
        // F32 norms: out_features * 4 bytes
        self.out_features * self.in_features + self.out_features * 4
    }

    /// Memory usage of the equivalent uncompressed F32 weight in bytes.
    pub fn uncompressed_weight_bytes(&self) -> usize {
        self.out_features * self.in_features * 4
    }

    /// Compression ratio vs F32.
    pub fn compression_ratio(&self) -> f64 {
        self.uncompressed_weight_bytes() as f64 / self.quantized_weight_bytes() as f64
    }
}

impl crate::Module for PolarQuantLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w = self.dequantize_weight()?;
        let wt = w.t()?;
        let x = match *x.dims() {
            [b1, b2, m, _k] => {
                if x.is_contiguous() {
                    x.reshape((b1 * b2 * m, ()))?
                        .matmul(&wt)?
                        .reshape((b1, b2, m, ()))?
                } else {
                    let wt = w.broadcast_left((b1, b2))?.t()?;
                    x.matmul(&wt)?
                }
            }
            [bsize, m, _k] => {
                if x.is_contiguous() {
                    x.reshape((bsize * m, ()))?
                        .matmul(&wt)?
                        .reshape((bsize, m, ()))?
                } else {
                    let wt = w.broadcast_left(bsize)?.t()?;
                    x.matmul(&wt)?
                }
            }
            [_k] => {
                let y = x.unsqueeze(0)?.matmul(&wt)?;
                y.squeeze(0)?
            }
            _ => x.matmul(&wt)?,
        };
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_linear_shapes() -> Result<()> {
        let device = Device::Cpu;
        let (out_f, in_f) = (64, 128);

        let w = Tensor::randn(0f32, 1f32, (out_f, in_f), &device)?;
        let b = Tensor::randn(0f32, 1f32, (out_f,), &device)?;
        let linear = Linear::new(w, Some(b));

        let pq = PolarQuantLinear::from_linear(&linear, 4, &device)?;
        assert_eq!(pq.in_features(), in_f);
        assert_eq!(pq.out_features(), out_f);
        assert_eq!(pq.bit_width(), 4);
        assert!(pq.bias().is_some());

        // Forward
        let x = Tensor::randn(0f32, 1f32, (2, in_f), &device)?;
        let y = crate::Module::forward(&pq, &x)?;
        assert_eq!(y.dims(), &[2, out_f]);

        Ok(())
    }

    #[test]
    fn test_from_linear_no_bias() -> Result<()> {
        let device = Device::Cpu;
        let (out_f, in_f) = (32, 64);

        let w = Tensor::randn(0f32, 1f32, (out_f, in_f), &device)?;
        let linear = Linear::new(w, None);

        let pq = PolarQuantLinear::from_linear(&linear, 3, &device)?;
        assert!(pq.bias().is_none());

        let x = Tensor::randn(0f32, 1f32, (in_f,), &device)?;
        let y = crate::Module::forward(&pq, &x)?;
        assert_eq!(y.dims(), &[out_f]);

        Ok(())
    }

    #[test]
    fn test_approximation_quality() -> Result<()> {
        let device = Device::Cpu;
        let (out_f, in_f) = (32, 128);

        let w = Tensor::randn(0f32, 1f32, (out_f, in_f), &device)?;
        let linear = Linear::new(w, None);

        let x = Tensor::randn(0f32, 1f32, (8, in_f), &device)?;
        let y_exact = crate::Module::forward(&linear, &x)?;

        let pq = PolarQuantLinear::from_linear(&linear, 4, &device)?;
        let y_approx = crate::Module::forward(&pq, &x)?;

        // Relative error should be small for 4-bit
        let err = (&y_exact - &y_approx)?
            .sqr()?
            .sum_all()?
            .to_scalar::<f32>()? as f64;
        let mag = y_exact.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        let rel = err / (mag + 1e-10);
        assert!(
            rel < 0.05,
            "4-bit linear output relative error too high: {rel:.4}"
        );

        Ok(())
    }

    #[test]
    fn test_compression_ratio() -> Result<()> {
        let device = Device::Cpu;
        let (out_f, in_f) = (64, 128);

        let w = Tensor::randn(0f32, 1f32, (out_f, in_f), &device)?;
        let linear = Linear::new(w, None);

        let pq = PolarQuantLinear::from_linear(&linear, 4, &device)?;

        // U8 indices: 64*128 = 8192, norms: 64*4 = 256, total = 8448
        // F32 weight: 64*128*4 = 32768
        // Ratio: 32768 / 8448 ≈ 3.88
        let ratio = pq.compression_ratio();
        assert!(ratio > 3.5, "Expected > 3.5x compression, got {ratio:.2}x");

        Ok(())
    }

    #[test]
    fn test_dequantize_weight() -> Result<()> {
        let device = Device::Cpu;
        let (out_f, in_f) = (16, 64);

        let w = Tensor::randn(0f32, 1f32, (out_f, in_f), &device)?;
        let linear = Linear::new(w.clone(), None);

        let pq = PolarQuantLinear::from_linear(&linear, 4, &device)?;
        let w_deq = pq.dequantize_weight()?;

        assert_eq!(w_deq.dims(), &[out_f, in_f]);

        // Dequantized weight should be close to original
        let err = (&w - &w_deq)?.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        let mag = w.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        let rel = err / (mag + 1e-10);
        assert!(rel < 0.05, "Weight reconstruction error too high: {rel:.4}");

        Ok(())
    }

    #[test]
    fn test_batched_forward() -> Result<()> {
        let device = Device::Cpu;
        let (out_f, in_f) = (32, 64);

        let w = Tensor::randn(0f32, 1f32, (out_f, in_f), &device)?;
        let b = Tensor::randn(0f32, 1f32, (out_f,), &device)?;
        let linear = Linear::new(w, Some(b));

        let pq = PolarQuantLinear::from_linear(&linear, 4, &device)?;

        // 3D input: [batch, seq, features]
        let x = Tensor::randn(0f32, 1f32, (2, 5, in_f), &device)?;
        let y = crate::Module::forward(&pq, &x)?;
        assert_eq!(y.dims(), &[2, 5, out_f]);

        // 4D input: [b1, b2, seq, features]
        let x4 = Tensor::randn(0f32, 1f32, (2, 3, 4, in_f), &device)?;
        let y4 = crate::Module::forward(&pq, &x4)?;
        assert_eq!(y4.dims(), &[2, 3, 4, out_f]);

        Ok(())
    }

    #[test]
    fn test_quality_improves_with_bits() -> Result<()> {
        let device = Device::Cpu;
        let (out_f, in_f) = (32, 128);

        let w = Tensor::randn(0f32, 1f32, (out_f, in_f), &device)?;
        let linear = Linear::new(w, None);

        let x = Tensor::randn(0f32, 1f32, (4, in_f), &device)?;
        let y_exact = crate::Module::forward(&linear, &x)?;
        let mag = y_exact.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;

        let mut prev_err = f64::MAX;
        for bits in [2, 3, 4] {
            let pq = PolarQuantLinear::from_linear(&linear, bits, &device)?;
            let y_approx = crate::Module::forward(&pq, &x)?;
            let err = (&y_exact - &y_approx)?
                .sqr()?
                .sum_all()?
                .to_scalar::<f32>()? as f64
                / (mag + 1e-10);

            assert!(
                err < prev_err,
                "{bits}-bit error {err:.6} >= prev {prev_err:.6}"
            );
            prev_err = err;
        }

        Ok(())
    }
}
