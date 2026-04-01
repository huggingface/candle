//! TurboQuantMse-compressed Neural Network Layers
//!
//! Provides drop-in replacements for standard neural network layers that store
//! weights in TurboQuantMse-compressed form. Weights are dequantized on-the-fly
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
//! - [`TurboQuantLinear`]: Drop-in replacement for [`Linear`](crate::Linear).
//!   Quantize pretrained weights with [`from_linear`](TurboQuantLinear::from_linear),
//!   or construct directly from quantized data.
//!
//! # Example
//!
//! ```no_run
//! use candle::{Device, DType, Tensor};
//! use candle_nn::{Linear, Module};
//! use candle_nn::turboquant_nn::TurboQuantLinear;
//!
//! let device = Device::Cpu;
//! // Start with a pretrained Linear layer
//! let weight = Tensor::randn(0f32, 1f32, (64, 128), &device).unwrap();
//! let linear = Linear::new(weight, None);
//!
//! // Compress to 4-bit TurboQuantMse
//! let pq_linear = TurboQuantLinear::from_linear(&linear, 4, &device).unwrap();
//!
//! // Same forward API
//! let x = Tensor::randn(0f32, 1f32, (1, 128), &device).unwrap();
//! let y = pq_linear.forward(&x).unwrap();
//! assert_eq!(y.dims(), &[1, 64]);
//! ```

use candle::{DType, Device, Result, Tensor};

use crate::turboquant_mse::{TurboMseMatMul, TurboMseQuantized, TurboQuantMse};
use crate::Linear;

/// Linear layer with TurboQuantMse-compressed weights.
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
pub struct TurboQuantLinear {
    quantizer: TurboQuantMse,
    indices: Tensor,
    norms: Tensor,
    fused_op: Option<TurboMseMatMul>,
    bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
}

impl TurboQuantLinear {
    /// Quantize a pretrained [`Linear`] layer into TurboQuantMse-compressed form.
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
    /// let pq = TurboQuantLinear::from_linear(&pretrained_layer, 4, &device)?;
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
        let quantizer = TurboQuantMse::new(in_features, bit_width, internal_dtype, device)?;
        let q = quantizer.quantize(weight)?;
        let indices = q.indices.to_dtype(DType::U8)?;
        let fused_op = TurboMseMatMul::new(&quantizer, &indices, &q.norms).ok();

        Ok(Self {
            quantizer,
            indices,
            norms: q.norms,
            fused_op,
            bias: linear.bias().cloned(),
            in_features,
            out_features,
        })
    }

    /// Quantize a pretrained [`Linear`] layer using fast Hadamard rotation.
    ///
    /// Same as [`from_linear`](Self::from_linear) but uses a randomized
    /// Walsh-Hadamard transform instead of a dense orthogonal matrix.
    /// Requires `in_features` to be a power of 2.
    pub fn from_linear_hadamard(
        linear: &Linear,
        bit_width: usize,
        device: &Device,
    ) -> Result<Self> {
        let weight = linear.weight();
        let (out_features, in_features) = weight.dims2()?;

        let dtype = weight.dtype();
        let internal_dtype = if dtype == DType::F64 {
            DType::F64
        } else {
            DType::F32
        };
        let quantizer =
            TurboQuantMse::new_hadamard(in_features, bit_width, internal_dtype, device)?;
        let q = quantizer.quantize(weight)?;
        let indices = q.indices.to_dtype(DType::U8)?;
        let fused_op = TurboMseMatMul::new(&quantizer, &indices, &q.norms).ok();

        Ok(Self {
            quantizer,
            indices,
            norms: q.norms,
            fused_op,
            bias: linear.bias().cloned(),
            in_features,
            out_features,
        })
    }

    /// Create a TurboQuantLinear from pre-quantized data.
    ///
    /// # Arguments
    /// * `quantizer` - The TurboQuantMse quantizer (holds rotation matrix + centroids).
    /// * `indices` - Centroid indices, shape `[out_features, in_features]`, dtype U8.
    /// * `norms` - Row norms, shape `[out_features]`.
    /// * `bias` - Optional bias, shape `[out_features]`.
    pub fn new(
        quantizer: TurboQuantMse,
        indices: Tensor,
        norms: Tensor,
        bias: Option<Tensor>,
    ) -> Result<Self> {
        let (out_features, in_features) = indices.dims2()?;
        let fused_op = TurboMseMatMul::new(&quantizer, &indices, &norms).ok();
        Ok(Self {
            quantizer,
            indices,
            norms,
            fused_op,
            bias,
            in_features,
            out_features,
        })
    }

    /// Dequantize and return the full-precision weight matrix.
    ///
    /// Shape: `[out_features, in_features]`.
    pub fn dequantize_weight(&self) -> Result<Tensor> {
        let q = TurboMseQuantized {
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
    /// Counts bit-packed index storage + norm storage (F32), excluding the
    /// shared rotation matrix and centroids. At 2-bit with dim=768, this is
    /// ~4× smaller than U8 index storage.
    pub fn quantized_weight_bytes(&self) -> usize {
        let bits = self.quantizer.bit_width();
        // Bit-packed indices: ceil(out * in * bits / 8) bytes
        let index_bytes = (self.out_features * self.in_features * bits).div_ceil(8);
        // F32 norms: out_features * 4 bytes
        let norm_bytes = self.out_features * 4;
        index_bytes + norm_bytes
    }

    /// Memory usage of the equivalent uncompressed F32 weight in bytes.
    pub fn uncompressed_weight_bytes(&self) -> usize {
        self.out_features * self.in_features * 4
    }

    /// Compression ratio vs F32.
    pub fn compression_ratio(&self) -> f64 {
        self.uncompressed_weight_bytes() as f64 / self.quantized_weight_bytes() as f64
    }

    /// Fallback forward using full weight dequantization.
    pub(crate) fn forward_dequantize(&self, x: &Tensor) -> Result<Tensor> {
        let w = self.dequantize_weight()?;
        let wt = w.t()?;
        match *x.dims() {
            [b1, b2, m, _k] => {
                if x.is_contiguous() {
                    x.reshape((b1 * b2 * m, ()))?
                        .matmul(&wt)?
                        .reshape((b1, b2, m, ()))
                } else {
                    let wt = w.broadcast_left((b1, b2))?.t()?;
                    x.matmul(&wt)
                }
            }
            [bsize, m, _k] => {
                if x.is_contiguous() {
                    x.reshape((bsize * m, ()))?
                        .matmul(&wt)?
                        .reshape((bsize, m, ()))
                } else {
                    let wt = w.broadcast_left(bsize)?.t()?;
                    x.matmul(&wt)
                }
            }
            [_k] => x.unsqueeze(0)?.matmul(&wt)?.squeeze(0),
            _ => x.matmul(&wt),
        }
    }
}

impl crate::Module for TurboQuantLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_result = if let Some(ref fused) = self.fused_op {
            if x.dtype() == DType::F32 {
                fused.forward(x)?
            } else {
                self.forward_dequantize(x)?
            }
        } else {
            self.forward_dequantize(x)?
        };

        match &self.bias {
            None => Ok(x_result),
            Some(bias) => x_result.broadcast_add(bias),
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

        let pq = TurboQuantLinear::from_linear(&linear, 4, &device)?;
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

        let pq = TurboQuantLinear::from_linear(&linear, 3, &device)?;
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

        let pq = TurboQuantLinear::from_linear(&linear, 4, &device)?;
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

        // 4-bit: packed indices = 64*128*4/8 = 4096 bytes, norms = 256, total = 4352
        // F32 weight = 32768, ratio = 32768/4352 ≈ 7.5x
        let pq4 = TurboQuantLinear::from_linear(&linear, 4, &device)?;
        let ratio4 = pq4.compression_ratio();
        assert!(ratio4 > 7.0, "Expected > 7x at 4-bit, got {ratio4:.2}x");

        // 2-bit: packed indices = 64*128*2/8 = 2048 bytes, norms = 256, total = 2304
        // ratio = 32768/2304 ≈ 14.2x
        let pq2 = TurboQuantLinear::from_linear(&linear, 2, &device)?;
        let ratio2 = pq2.compression_ratio();
        assert!(ratio2 > 14.0, "Expected > 14x at 2-bit, got {ratio2:.2}x");
        assert!(ratio2 > ratio4, "2-bit should compress more than 4-bit");

        Ok(())
    }

    #[test]
    fn test_dequantize_weight() -> Result<()> {
        let device = Device::Cpu;
        let (out_f, in_f) = (16, 64);

        let w = Tensor::randn(0f32, 1f32, (out_f, in_f), &device)?;
        let linear = Linear::new(w.clone(), None);

        let pq = TurboQuantLinear::from_linear(&linear, 4, &device)?;
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

        let pq = TurboQuantLinear::from_linear(&linear, 4, &device)?;

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
            let pq = TurboQuantLinear::from_linear(&linear, bits, &device)?;
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

    #[test]
    fn test_hadamard_from_linear() -> Result<()> {
        let device = Device::Cpu;
        let (out_f, in_f) = (32, 128); // in_f must be power of 2

        let w = Tensor::randn(0f32, 1f32, (out_f, in_f), &device)?;
        let linear = Linear::new(w, None);

        let pq = TurboQuantLinear::from_linear_hadamard(&linear, 4, &device)?;
        assert_eq!(pq.in_features(), in_f);
        assert_eq!(pq.out_features(), out_f);

        let x = Tensor::randn(0f32, 1f32, (2, in_f), &device)?;
        let y_exact = crate::Module::forward(&linear, &x)?;
        let y_pq = crate::Module::forward(&pq, &x)?;

        let err = (&y_exact - &y_pq)?.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        let mag = y_exact.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        let rel = err / (mag + 1e-10);
        assert!(
            rel < 0.05,
            "Hadamard 4-bit relative error too high: {rel:.4}"
        );

        Ok(())
    }

    #[test]
    fn test_fused_matches_dequantize() -> Result<()> {
        let device = Device::Cpu;
        let (out_f, in_f) = (32, 64);

        let w = Tensor::randn(0f32, 1f32, (out_f, in_f), &device)?;
        let linear = Linear::new(w, None);
        let pq = TurboQuantLinear::from_linear(&linear, 4, &device)?;

        // The fused op should produce the same result as dequantize+matmul
        let x = Tensor::randn(0f32, 1f32, (4, in_f), &device)?;

        // Fused path (via Module::forward)
        let y_fused = crate::Module::forward(&pq, &x)?;

        // Dequantize path (explicit)
        let y_deq = pq.forward_dequantize(&x)?;

        let diff = (&y_fused - &y_deq)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(
            diff < 1e-4,
            "Fused vs dequantize max diff: {diff} (should be ~0)"
        );

        Ok(())
    }
}
