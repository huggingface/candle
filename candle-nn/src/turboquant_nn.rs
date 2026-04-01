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

use crate::turboquant::TurboQuant;
use crate::turboquant_mse::{TurboMseMatMul, TurboMseQuantized, TurboQuantMse};
use crate::Linear;

/// Linear layer with TurboQuant-compressed weights.
///
/// Stores the weight matrix as quantized MSE indices + QJL sign corrections,
/// providing unbiased inner product estimation (`E[⟨x, W̃ᵢ⟩] = ⟨x, Wᵢ⟩`).
///
/// The forward pass decomposes into:
/// 1. **MSE path**: fused centroid-dot matmul (fast, from `TurboMseMatMul`)
/// 2. **QJL correction**: `√(π/2)/d · (x @ S^T) @ signs^T @ diag(residual_norms)`
///
/// This avoids materializing the full weight matrix while preserving unbiased
/// inner products from TurboQuant Algorithm 2.
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
    mse_quantizer: TurboQuantMse,
    indices: Tensor,
    norms: Tensor,
    fused_op: Option<TurboMseMatMul>,
    /// QJL projection matrix S^T: [d, d] — shared from the TurboQuant quantizer
    qjl_projection_t: Tensor,
    /// QJL sign bits: [out_features, in_features], values ±1.0
    qjl_signs: Tensor,
    /// Pre-computed √(π/2)/d · residual_norms: [out_features]
    scaled_residual_norms: Tensor,
    bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
    bit_width: usize,
}

impl TurboQuantLinear {
    /// Quantize a pretrained [`Linear`] layer into TurboQuant-compressed form.
    ///
    /// Uses the full TurboQuant algorithm (MSE + QJL) for unbiased inner products.
    ///
    /// # Arguments
    /// * `linear` - The pretrained linear layer to compress.
    /// * `bit_width` - Quantization bits per coordinate (typically 2–4).
    /// * `device` - Device for the quantizer's rotation matrix.
    pub fn from_linear(linear: &Linear, bit_width: usize, device: &Device) -> Result<Self> {
        let weight = linear.weight();
        let (out_features, in_features) = weight.dims2()?;

        let dtype = weight.dtype();
        let internal_dtype = if dtype == DType::F64 {
            DType::F64
        } else {
            DType::F32
        };
        let turbo = TurboQuant::new(in_features, bit_width, internal_dtype, device)?;
        let q = turbo.quantize(weight)?;
        let mse_quantizer = turbo.mse_quantizer().clone();
        let indices = q.polar.indices.to_dtype(DType::U8)?;
        let fused_op = TurboMseMatMul::new(&mse_quantizer, &indices, &q.polar.norms).ok();
        let qjl_scale = (std::f64::consts::FRAC_PI_2).sqrt() / in_features as f64;
        let scaled_residual_norms = (&q.residual_norms * qjl_scale)?;

        Ok(Self {
            mse_quantizer,
            indices,
            norms: q.polar.norms,
            fused_op,
            qjl_projection_t: turbo.projection().t()?.contiguous()?,
            qjl_signs: q.qjl_signs,
            scaled_residual_norms,
            bias: linear.bias().cloned(),
            in_features,
            out_features,
            bit_width,
        })
    }

    /// Quantize a pretrained [`Linear`] layer using fast Hadamard rotation.
    ///
    /// Same as [`from_linear`](Self::from_linear) but uses a randomized
    /// Walsh-Hadamard transform for the MSE stage. Requires `in_features`
    /// to be a power of 2.
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
        // Use Hadamard for the MSE stage, standard Gaussian for QJL projection
        let mse_quantizer =
            TurboQuantMse::new_hadamard(in_features, bit_width - 1, internal_dtype, device)?;
        let turbo = TurboQuant::from_mse_quantizer(mse_quantizer.clone(), internal_dtype, device)?;
        let q = turbo.quantize(weight)?;
        let indices = q.polar.indices.to_dtype(DType::U8)?;
        let fused_op = TurboMseMatMul::new(&mse_quantizer, &indices, &q.polar.norms).ok();
        let qjl_scale = (std::f64::consts::FRAC_PI_2).sqrt() / in_features as f64;
        let scaled_residual_norms = (&q.residual_norms * qjl_scale)?;

        Ok(Self {
            mse_quantizer,
            indices,
            norms: q.polar.norms,
            fused_op,
            qjl_projection_t: turbo.projection().t()?.contiguous()?,
            qjl_signs: q.qjl_signs,
            scaled_residual_norms,
            bias: linear.bias().cloned(),
            in_features,
            out_features,
            bit_width,
        })
    }

    /// Dequantize and return the full-precision weight matrix.
    ///
    /// Includes the QJL correction for unbiased reconstruction.
    pub fn dequantize_weight(&self) -> Result<Tensor> {
        // MSE reconstruction
        let q = TurboMseQuantized {
            indices: self.indices.to_dtype(DType::U32)?,
            norms: self.norms.clone(),
        };
        let w_mse = self.mse_quantizer.dequantize(&q)?;

        // QJL correction: √(π/2)/d · diag(residual_norms) · signs · S
        let w_qjl = self.qjl_signs.matmul(&self.qjl_projection_t.t()?)?;
        let w_qjl = w_qjl.broadcast_mul(&self.scaled_residual_norms.unsqueeze(1)?)?;

        (w_mse + w_qjl)?.contiguous()
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
        self.bit_width
    }

    /// Approximate memory usage of the quantized weights in bytes.
    pub fn quantized_weight_bytes(&self) -> usize {
        let mse_bits = self.bit_width - 1;
        // Bit-packed MSE indices + 1-bit QJL signs
        let index_bytes = (self.out_features * self.in_features * mse_bits).div_ceil(8);
        let sign_bytes = (self.out_features * self.in_features).div_ceil(8);
        // F32 norms (MSE) + F32 residual norms (QJL)
        let norm_bytes = self.out_features * 4 * 2;
        index_bytes + sign_bytes + norm_bytes
    }

    /// Memory usage of the equivalent uncompressed F32 weight in bytes.
    pub fn uncompressed_weight_bytes(&self) -> usize {
        self.out_features * self.in_features * 4
    }

    /// Compression ratio vs F32.
    pub fn compression_ratio(&self) -> f64 {
        self.uncompressed_weight_bytes() as f64 / self.quantized_weight_bytes() as f64
    }

    /// Forward using full weight dequantization (fallback).
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

    /// Forward with decomposed MSE fused + QJL correction.
    ///
    /// y = fused_mse(x) + √(π/2)/d · (x @ S^T) @ signs^T @ diag(residual_norms)
    fn forward_fused_with_qjl(&self, fused: &TurboMseMatMul, x: &Tensor) -> Result<Tensor> {
        let orig_dims = x.dims().to_vec();
        let d = self.in_features;
        let batch = x.elem_count() / d;

        // MSE part via fused kernel
        let y_mse = fused.forward(x)?;

        // QJL correction: x_proj @ signs^T * scaled_residual_norms
        let x_2d = if x.dims().len() == 1 {
            x.unsqueeze(0)?
        } else {
            x.reshape((batch, d))?
        };
        let x_proj = x_2d.matmul(&self.qjl_projection_t)?;
        let correction = x_proj.matmul(&self.qjl_signs.t()?)?;
        let correction = correction.broadcast_mul(&self.scaled_residual_norms.unsqueeze(0)?)?;

        // Reshape correction to match y_mse
        let correction = match orig_dims.len() {
            1 => correction.squeeze(0)?,
            _ => {
                let mut out_dims = orig_dims;
                *out_dims.last_mut().unwrap() = self.out_features;
                correction.reshape(out_dims)?
            }
        };

        y_mse + correction
    }
}

impl crate::Module for TurboQuantLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_result = if let Some(ref fused) = self.fused_op {
            if x.dtype() == DType::F32 {
                self.forward_fused_with_qjl(fused, x)?
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

        // Relative error should be reasonable for 4-bit TurboQuant
        // (uses 3-bit MSE + 1-bit QJL, so slightly more distortion than pure 4-bit MSE)
        let err = (&y_exact - &y_approx)?
            .sqr()?
            .sum_all()?
            .to_scalar::<f32>()? as f64;
        let mag = y_exact.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        let rel = err / (mag + 1e-10);
        assert!(
            rel < 0.15,
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

        // 4-bit TurboQuant: (b-1)=3-bit MSE indices + 1-bit QJL signs + norms
        let pq4 = TurboQuantLinear::from_linear(&linear, 4, &device)?;
        let ratio4 = pq4.compression_ratio();
        assert!(ratio4 > 5.0, "Expected > 5x at 4-bit, got {ratio4:.2}x");

        // 2-bit TurboQuant: 1-bit MSE + 1-bit QJL + norms
        let pq2 = TurboQuantLinear::from_linear(&linear, 2, &device)?;
        let ratio2 = pq2.compression_ratio();
        assert!(ratio2 > 10.0, "Expected > 10x at 2-bit, got {ratio2:.2}x");
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

        // Dequantized weight should be close to original (includes QJL correction)
        let err = (&w - &w_deq)?.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        let mag = w.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        let rel = err / (mag + 1e-10);
        assert!(rel < 0.15, "Weight reconstruction error too high: {rel:.4}");

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
            rel < 0.15,
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
