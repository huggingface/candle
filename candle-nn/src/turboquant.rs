//! TurboQuant: Online Vector Quantization with Near-Optimal Distortion Rate
//!
//! Implementation of the TurboQuant algorithm from:
//! "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
//! ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874))
//!
//! TurboQuant provides unbiased inner product estimation by composing two stages:
//! 1. **TurboQuantMse** (b−1 bits): MSE-optimal scalar quantization via random rotation
//!    and Lloyd-Max codebooks. See [`crate::turboquant_mse`] for standalone usage.
//! 2. **QJL** (1 bit): Quantized Johnson-Lindenstrauss correction on the residual
//!    for unbiased inner product estimation.
//!
//! # When to use TurboQuant vs TurboQuantMse
//!
//! - Use [`TurboQuantMse`](crate::turboquant_mse::TurboQuantMse) when you need MSE-optimal
//!   reconstruction (e.g., KV cache compression, embedding storage).
//! - Use [`TurboQuant`] when you need **unbiased inner product estimation**
//!   (e.g., approximate nearest neighbor search, similarity scoring).
//!
//! # Example
//!
//! ```no_run
//! use candle::{Device, DType, Tensor};
//! use candle_nn::turboquant::TurboQuant;
//!
//! let device = Device::Cpu;
//! let quantizer = TurboQuant::new(128, 4, DType::F32, &device).unwrap();
//!
//! let x = Tensor::randn(0f32, 1f32, (10, 128), &device).unwrap();
//! let quantized = quantizer.quantize(&x).unwrap();
//! let reconstructed = quantizer.dequantize(&quantized).unwrap();
//! ```

use candle::{DType, Device, Result, Tensor, D};

use crate::turboquant_mse::{TurboMseQuantized, TurboQuantMse};

/// Pack F32 ±1.0 signs into U8 bytes (8 signs per byte, LSB first).
///
/// Input: `[n, d]` F32 with values ±1.0.
/// Output: `[n, ceil(d/8)]` U8.
///
/// Bit encoding: 1 = +1.0, 0 = −1.0. Bit 0 of byte 0 holds element 0,
/// bit 1 holds element 1, etc.
pub fn pack_signs(signs: &Tensor) -> Result<Tensor> {
    let (n, d) = signs.dims2()?;
    let d_packed = d.div_ceil(8);
    let d_padded = d_packed * 8;

    // ±1.0 → {0.0, 1.0}
    let bits = signs.to_dtype(DType::F32)?.affine(0.5, 0.5)?;

    // Pad last dim to a multiple of 8
    let bits = if d_padded > d {
        let pad = Tensor::zeros((n, d_padded - d), DType::F32, signs.device())?;
        Tensor::cat(&[&bits, &pad], 1)?
    } else {
        bits
    };

    // [n, d_packed, 8] × bit-position weights → [n, d_packed]
    let bits = bits.reshape((n, d_packed, 8))?;
    let powers = Tensor::new(
        &[1.0f32, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0],
        signs.device(),
    )?;
    let packed = bits.broadcast_mul(&powers)?.sum(2)?.to_dtype(DType::U8)?;

    Ok(packed)
}

/// Unpack U8 packed signs back to F32 ±1.0.
///
/// Input: `[n, ceil(d/8)]` U8.
/// Output: `[n, d]` F32.
pub fn unpack_signs(packed: &Tensor, d: usize) -> Result<Tensor> {
    let (n, d_packed) = packed.dims2()?;

    let packed_f32 = packed.to_dtype(DType::F32)?.unsqueeze(2)?; // [n, d_packed, 1]

    // Extract each bit: bit_i = floor(x / 2^i) − 2·floor(x / 2^(i+1))
    let mut planes = Vec::with_capacity(8);
    for i in 0..8u32 {
        let div = (1u32 << i) as f64;
        let div_next = (1u32 << (i + 1)) as f64;
        let lo = packed_f32.affine(1.0 / div, 0.0)?.floor()?;
        let hi = packed_f32.affine(1.0 / div_next, 0.0)?.floor()?;
        planes.push((&lo - hi.affine(2.0, 0.0)?)?);
    }
    let bits = Tensor::cat(&planes, 2)?; // [n, d_packed, 8]
    let bits = bits.reshape((n, d_packed * 8))?;

    // Trim padding and map {0,1} → {−1.0, +1.0}
    let bits = if d_packed * 8 > d {
        bits.narrow(1, 0, d)?
    } else {
        bits
    };
    bits.affine(2.0, -1.0)
}

/// Result of TurboQuant quantization.
///
/// Stores the TurboQuantMse MSE quantization (b−1 bits), packed QJL sign bits (1 bit),
/// and residual norms needed for unbiased inner product reconstruction.
#[derive(Debug, Clone)]
pub struct TurboQuantized {
    /// TurboQuantMse quantization part (b−1 bits per coordinate).
    pub polar: TurboMseQuantized,
    /// QJL sign bits, packed. Shape: `[n, ceil(d/8)]`, dtype U8.
    /// Each byte packs 8 signs (LSB first): bit set = +1.0, unset = −1.0.
    pub qjl_signs: Tensor,
    /// Residual L2 norms. Shape: `[n]`.
    pub residual_norms: Tensor,
}

/// Inner-product-optimized vector quantizer combining TurboQuantMse and QJL.
///
/// Provides unbiased inner product estimation by:
/// 1. Quantizing with [`TurboQuantMse`](crate::turboquant_mse::TurboQuantMse) using b−1
///    bits per coordinate for MSE-optimal reconstruction
/// 2. Applying 1-bit QJL (Quantized Johnson-Lindenstrauss) to the residual
///    for unbiased inner product correction
///
/// # Properties
///
/// - **Unbiased**: E[⟨y, x̃⟩] = ⟨y, x⟩
/// - **Low distortion**: inner product error ≤ (√3·π²·‖y‖²/d) · 1/4^b
/// - For b=1,2,3,4: distortion ≈ 1.57/d, 0.56/d, 0.18/d, 0.047/d
///
/// # Example
///
/// ```no_run
/// use candle::{Device, DType, Tensor};
/// use candle_nn::turboquant::TurboQuant;
///
/// let device = Device::Cpu;
/// let quantizer = TurboQuant::new(128, 4, DType::F32, &device).unwrap();
///
/// let x = Tensor::randn(0f32, 1f32, (10, 128), &device).unwrap();
/// let quantized = quantizer.quantize(&x).unwrap();
/// let reconstructed = quantizer.dequantize(&quantized).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct TurboQuant {
    polar: TurboQuantMse,
    projection: Tensor,
    dim: usize,
    bit_width: usize,
    dtype: DType,
}

impl TurboQuant {
    /// Create a new TurboQuant quantizer.
    ///
    /// Internally creates a [`TurboQuantMse`](crate::turboquant_mse::TurboQuantMse)
    /// quantizer with b−1 bits and a random Gaussian projection matrix for QJL.
    ///
    /// # Arguments
    /// * `dim` - Vector dimension (d).
    /// * `bit_width` - Total bits per coordinate (b ≥ 1). Uses b−1 for TurboQuantMse + 1 for QJL.
    /// * `dtype` - Data type for internal computations.
    /// * `device` - Device for tensor allocation.
    pub fn new(dim: usize, bit_width: usize, dtype: DType, device: &Device) -> Result<Self> {
        if bit_width < 1 {
            candle::bail!("TurboQuant requires bit_width >= 1");
        }

        let polar = TurboQuantMse::new(dim, bit_width - 1, dtype, device)?;
        // Random Gaussian projection matrix for QJL
        let projection = Tensor::randn(0f64, 1f64, (dim, dim), &Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(device)?;

        Ok(Self {
            polar,
            projection,
            dim,
            bit_width,
            dtype,
        })
    }

    /// Quantize vectors.
    ///
    /// # Arguments
    /// * `x` - Input tensor, shape `[n, d]` or `[d]`
    ///
    /// # Returns
    /// [`TurboQuantized`] with TurboQuantMse indices, QJL signs, and norms.
    pub fn quantize(&self, x: &Tensor) -> Result<TurboQuantized> {
        let is_single = x.dims().len() == 1;
        let x = if is_single {
            x.unsqueeze(0)?
        } else {
            x.clone()
        };
        let x = x.to_dtype(self.dtype)?;

        // Stage 1: TurboQuantMse with b-1 bits
        let polar_q = self.polar.quantize(&x)?;
        let x_polar = self.polar.dequantize(&polar_q)?;

        // Stage 2: Compute residual r = x - TurboQuantMse(x)
        let residual = x.broadcast_sub(&x_polar)?;
        let residual_norms = residual.sqr()?.sum(D::Minus1)?.sqrt()?;

        // Stage 3: QJL on residual: sign(S · r), packed into U8 bits
        let sr = residual.matmul(&self.projection.t()?)?;
        let zeros = sr.zeros_like()?;
        let positive = sr.ge(&zeros)?;
        // Map U8 {0, 1} -> F32 {-1.0, +1.0}, then bit-pack into U8
        let signs_f32 = positive.to_dtype(DType::F32)?.affine(2.0, -1.0)?;
        let qjl_signs = pack_signs(&signs_f32)?;

        Ok(TurboQuantized {
            polar: polar_q,
            qjl_signs,
            residual_norms,
        })
    }

    /// Dequantize back to vectors.
    ///
    /// # Arguments
    /// * `q` - Quantized representation from [`quantize`](Self::quantize)
    ///
    /// # Returns
    /// Reconstructed tensor, shape `[n, d]`.
    pub fn dequantize(&self, q: &TurboQuantized) -> Result<Tensor> {
        // Stage 1: TurboQuantMse dequantization
        let x_polar = self.polar.dequantize(&q.polar)?;

        // Stage 2: QJL dequantization — unpack U8 packed signs to F32 ±1.0
        // x_qjl = √(π/2)/d · γ · S^T · qjl
        let signs = unpack_signs(&q.qjl_signs, self.dim)?.to_dtype(self.dtype)?;
        let x_qjl = signs.matmul(&self.projection)?;
        let scale = (std::f64::consts::FRAC_PI_2).sqrt() / self.dim as f64;
        let x_qjl = (x_qjl * scale)?;
        let x_qjl = x_qjl.broadcast_mul(&q.residual_norms.unsqueeze(D::Minus1)?)?;

        // Stage 3: Combine TurboQuantMse reconstruction + QJL correction
        x_polar.broadcast_add(&x_qjl)
    }

    /// Get the vector dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the bit width.
    pub fn bit_width(&self) -> usize {
        self.bit_width
    }

    /// Get a reference to the internal MSE quantizer.
    pub fn mse_quantizer(&self) -> &TurboQuantMse {
        &self.polar
    }

    /// Get a reference to the QJL projection matrix.
    pub fn projection(&self) -> &Tensor {
        &self.projection
    }

    /// Create a TurboQuant from an existing MSE quantizer.
    ///
    /// Generates a fresh QJL projection matrix. Useful for combining
    /// a Hadamard-based MSE quantizer with TurboQuant's QJL correction.
    pub fn from_mse_quantizer(mse: TurboQuantMse, dtype: DType, device: &Device) -> Result<Self> {
        let dim = mse.dim();
        let bit_width = mse.bit_width() + 1;
        let projection = Tensor::randn(0f64, 1f64, (dim, dim), &Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(device)?;
        Ok(Self {
            polar: mse,
            projection,
            dim,
            bit_width,
            dtype,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_dequantize() -> Result<()> {
        let device = Device::Cpu;
        let dim = 64;
        let bit_width = 4;
        let n = 8;

        let quantizer = TurboQuant::new(dim, bit_width, DType::F32, &device)?;
        let x = Tensor::randn(0f32, 1f32, (n, dim), &device)?;

        let q = quantizer.quantize(&x)?;
        let x_recon = quantizer.dequantize(&q)?;

        assert_eq!(x_recon.dims(), &[n, dim]);

        // QJL signs must be packed U8 with shape [n, ceil(dim/8)]
        assert_eq!(q.qjl_signs.dtype(), DType::U8);
        assert_eq!(q.qjl_signs.dims(), &[n, dim.div_ceil(8)]);

        // Unpack and verify all values are ±1
        let unpacked = unpack_signs(&q.qjl_signs, dim)?;
        let signs_abs = unpacked.abs()?;
        let ones = signs_abs.ones_like()?;
        let sign_err = signs_abs
            .broadcast_sub(&ones)?
            .abs()?
            .sum_all()?
            .to_scalar::<f32>()?;
        assert!(sign_err < 1e-5, "QJL signs should be ±1");

        Ok(())
    }

    #[test]
    fn test_unbiased_inner_product() -> Result<()> {
        // Verify that TurboQuant's QJL correction produces an unbiased inner product
        // estimator. We use fixed input vectors and average over many random quantizers.
        // With dim=128, 4-bit, and 200 trials the standard error is small enough that
        // a 0.5 relative tolerance never flakes.
        let device = Device::Cpu;
        let dim = 128;
        let bit_width = 4;
        let n_trials = 200;

        // Fixed unit-ish vectors for deterministic inputs
        let x_data: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.1).sin()).collect();
        let y_data: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.07).cos()).collect();
        let x = Tensor::new(x_data, &device)?.unsqueeze(0)?;
        let y = Tensor::new(y_data, &device)?.unsqueeze(0)?;
        let true_ip = (&x * &y)?.sum_all()?.to_scalar::<f32>()? as f64;

        let mut ip_sum = 0.0;
        for _ in 0..n_trials {
            let quantizer = TurboQuant::new(dim, bit_width, DType::F32, &device)?;
            let q = quantizer.quantize(&x)?;
            let x_recon = quantizer.dequantize(&q)?;
            let ip = (&x_recon * &y)?.sum_all()?.to_scalar::<f32>()? as f64;
            ip_sum += ip;
        }
        let ip_avg = ip_sum / n_trials as f64;

        // Unbiased: average should approach true inner product
        let rel_err = (ip_avg - true_ip).abs() / (true_ip.abs() + 1e-6);
        assert!(
            rel_err < 0.5,
            "Inner product bias too high: avg={ip_avg:.4}, true={true_ip:.4}, rel_err={rel_err:.4}"
        );

        Ok(())
    }

    #[test]
    fn test_bit_width_1() -> Result<()> {
        // bit_width=1 means 0 bits for TurboQuantMse + 1 bit for QJL (pure QJL)
        let device = Device::Cpu;
        let dim = 32;
        let quantizer = TurboQuant::new(dim, 1, DType::F32, &device)?;
        let x = Tensor::randn(0f32, 1f32, (4, dim), &device)?;

        let q = quantizer.quantize(&x)?;
        let x_recon = quantizer.dequantize(&q)?;

        assert_eq!(x_recon.dims(), &[4, dim]);
        Ok(())
    }
}
