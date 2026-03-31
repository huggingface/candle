//! TurboQuant: Online Vector Quantization with Near-Optimal Distortion Rate
//!
//! Implementation of the TurboQuant algorithm from:
//! "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
//! ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874))
//!
//! Two quantizers are provided:
//! - [`TurboQuantMse`]: Minimizes mean-squared error (MSE) between original and
//!   reconstructed vectors. Achieves MSE ≤ (√3·π/2) · 1/4^b for b-bit quantization.
//! - [`TurboQuantProd`]: Provides unbiased inner product estimation with low distortion.
//!   Combines MSE quantization with QJL (Quantized Johnson-Lindenstrauss) on the residual.
//!
//! # Algorithm Overview
//!
//! 1. **Random rotation**: Multiplies input by a random orthogonal matrix Π,
//!    making coordinates follow a Beta distribution (≈ Gaussian in high dimensions).
//! 2. **Scalar quantization**: Applies optimal Lloyd-Max scalar quantizers per coordinate.
//! 3. **Inner product mode** (TurboQuantProd only): Applies 1-bit QJL quantization on the
//!    MSE residual for unbiased inner product estimation.
//!
//! # Example
//!
//! ```no_run
//! use candle::{Device, DType, Tensor};
//! use candle_nn::turboquant::TurboQuantMse;
//!
//! let device = Device::Cpu;
//! let dim = 128;
//! let bit_width = 4;
//!
//! let quantizer = TurboQuantMse::new(dim, bit_width, DType::F32, &device).unwrap();
//!
//! // Quantize a batch of 10 vectors
//! let x = Tensor::randn(0f32, 1f32, (10, dim), &device).unwrap();
//! let quantized = quantizer.quantize(&x).unwrap();
//! let reconstructed = quantizer.dequantize(&quantized).unwrap();
//! ```

use candle::{DType, Device, Result, Tensor, D};
use std::fmt;

// ============================================================================
// Mathematical utilities
// ============================================================================

/// Approximation of the error function (Abramowitz & Stegun 7.1.26, ~1.5×10⁻⁷ accuracy).
fn erf_approx(x: f64) -> f64 {
    let sign = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// Standard normal PDF: φ(x) = (1/√(2π)) exp(-x²/2)
fn normal_pdf(x: f64) -> f64 {
    (-x * x / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Standard normal CDF: Φ(x) = ½(1 + erf(x/√2))
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))
}

/// Conditional expectation E[X | a ≤ X < b] for X ~ N(0, 1).
fn normal_conditional_mean(a: f64, b: f64) -> f64 {
    let prob = normal_cdf(b) - normal_cdf(a);
    if prob < 1e-15 {
        (a + b) / 2.0
    } else {
        (normal_pdf(a) - normal_pdf(b)) / prob
    }
}

// ============================================================================
// Lloyd-Max codebook computation
// ============================================================================

/// Compute optimal Lloyd-Max quantizer centroids for N(0, 1).
///
/// Uses the iterative Lloyd-Max algorithm to find `2^bit_width` centroids that
/// minimize expected squared quantization error for the standard normal
/// distribution. Returns centroids in ascending order.
fn lloyd_max_gaussian(bit_width: usize, max_iter: usize) -> Vec<f64> {
    let n_levels = 1usize << bit_width;
    if n_levels == 1 {
        return vec![0.0];
    }

    // Initialize centroids uniformly in [-3, 3] (covers >99.7% of N(0,1))
    let mut centroids: Vec<f64> = (0..n_levels)
        .map(|i| -3.0 + 6.0 * (i as f64 + 0.5) / n_levels as f64)
        .collect();

    for _ in 0..max_iter {
        let mut boundaries = Vec::with_capacity(n_levels + 1);
        boundaries.push(f64::NEG_INFINITY);
        for i in 0..n_levels - 1 {
            boundaries.push((centroids[i] + centroids[i + 1]) / 2.0);
        }
        boundaries.push(f64::INFINITY);

        let new_centroids: Vec<f64> = (0..n_levels)
            .map(|i| normal_conditional_mean(boundaries[i], boundaries[i + 1]))
            .collect();

        let max_change: f64 = centroids
            .iter()
            .zip(&new_centroids)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);

        centroids = new_centroids;
        if max_change < 1e-12 {
            break;
        }
    }

    centroids
}

// ============================================================================
// Random orthogonal matrix generation
// ============================================================================

/// Generate a random orthogonal matrix via modified Gram-Schmidt.
///
/// Generates a d×d random Gaussian matrix using candle's `randn`, pulls data
/// to CPU for orthogonalization, then returns the result as a Tensor.
fn random_orthogonal_tensor(dim: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let random_mat = Tensor::randn(0f64, 1f64, (dim, dim), &Device::Cpu)?;
    let rows: Vec<Vec<f64>> = random_mat.to_vec2()?;

    let mut orth = rows;

    for i in 0..dim {
        for j in 0..i {
            let dot: f64 = orth[i].iter().zip(&orth[j]).map(|(a, b)| a * b).sum();
            let orth_j = orth[j].clone();
            for k in 0..dim {
                orth[i][k] -= dot * orth_j[k];
            }
        }
        let norm: f64 = orth[i].iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for val in &mut orth[i] {
                *val /= norm;
            }
        }
    }

    let flat: Vec<f64> = orth.into_iter().flatten().collect();
    Tensor::from_vec(flat, (dim, dim), &Device::Cpu)?
        .to_dtype(dtype)?
        .to_device(device)
}

// ============================================================================
// Quantized representations
// ============================================================================

/// Result of MSE-optimized quantization.
///
/// Stores centroid indices (b bits per coordinate) and original vector norms.
pub struct MseQuantized {
    /// Centroid indices. Shape: `[n, d]`, dtype: `U32`.
    /// Each value is in `[0, 2^b)`.
    pub indices: Tensor,
    /// Original L2 norms. Shape: `[n]`.
    pub norms: Tensor,
}

/// Result of inner-product-optimized quantization.
///
/// Stores MSE quantization, QJL sign bits, and residual norms.
pub struct ProdQuantized {
    /// MSE quantization part (b-1 bits per coordinate).
    pub mse: MseQuantized,
    /// QJL sign bits. Shape: `[n, d]`, values in {-1.0, +1.0}.
    pub qjl_signs: Tensor,
    /// Residual L2 norms. Shape: `[n]`.
    pub residual_norms: Tensor,
}

// ============================================================================
// TurboQuantMse
// ============================================================================

/// MSE-optimized TurboQuant vector quantizer.
///
/// Quantizes d-dimensional vectors to b bits per coordinate by:
/// 1. Randomly rotating vectors (via orthogonal matrix Π) to induce near-Gaussian
///    coordinate distributions
/// 2. Applying optimal Lloyd-Max scalar quantization per coordinate
///
/// # Distortion Bounds
///
/// For unit-norm vectors and bit-width b:
/// - MSE ≤ (√3·π/2) · 1/4^b ≈ 2.72/4^b
/// - For b=1,2,3,4: MSE ≈ 0.36, 0.117, 0.03, 0.009
#[derive(Clone)]
pub struct TurboQuantMse {
    dim: usize,
    bit_width: usize,
    rotation: Tensor,
    centroids: Tensor,
    dtype: DType,
}

impl fmt::Debug for TurboQuantMse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TurboQuantMse")
            .field("dim", &self.dim)
            .field("bit_width", &self.bit_width)
            .field("dtype", &self.dtype)
            .finish()
    }
}

impl TurboQuantMse {
    /// Create a new MSE-optimized TurboQuant quantizer.
    ///
    /// # Arguments
    /// * `dim` - Vector dimension (d). Should be moderately large (≥ 32) for best results.
    /// * `bit_width` - Bits per coordinate (b). Typically 1–8.
    /// * `dtype` - Data type for internal computations (F32 or F64).
    /// * `device` - Device for tensor allocation.
    pub fn new(dim: usize, bit_width: usize, dtype: DType, device: &Device) -> Result<Self> {
        let unit_centroids = lloyd_max_gaussian(bit_width, 1000);

        // Scale by 1/√d: coordinates of rotated unit vectors ~ N(0, 1/d)
        let scale = 1.0 / (dim as f64).sqrt();
        let scaled: Vec<f64> = unit_centroids.iter().map(|&c| c * scale).collect();

        let centroids =
            Tensor::from_vec(scaled, (unit_centroids.len(),), device)?.to_dtype(dtype)?;
        let rotation = random_orthogonal_tensor(dim, dtype, device)?;

        Ok(Self {
            dim,
            bit_width,
            rotation,
            centroids,
            dtype,
        })
    }

    /// Quantize vectors.
    ///
    /// # Arguments
    /// * `x` - Input tensor, shape `[n, d]` or `[d]`
    ///
    /// # Returns
    /// [`MseQuantized`] with centroid indices and original norms.
    pub fn quantize(&self, x: &Tensor) -> Result<MseQuantized> {
        let is_single = x.dims().len() == 1;
        let x = if is_single {
            x.unsqueeze(0)?
        } else {
            x.clone()
        };
        let x = x.to_dtype(self.dtype)?;

        // Compute and store original norms
        let norms = x.sqr()?.sum(D::Minus1)?.sqrt()?;
        // Avoid division by zero for zero vectors
        let safe_norms = norms.affine(1.0, 1e-10)?;
        let x_norm = x.broadcast_div(&safe_norms.unsqueeze(D::Minus1)?)?;

        // Random rotation: y = x @ Π^T  (each row y_i = Π · x_i)
        let y = x_norm.matmul(&self.rotation.t()?)?;

        // Find nearest centroid for each coordinate
        let indices = if self.bit_width == 0 {
            Tensor::zeros(y.dims(), DType::U32, y.device())?
        } else {
            // y: [n, d] -> [n, d, 1], centroids: [2^b] -> [1, 1, 2^b]
            let y_exp = y.unsqueeze(D::Minus1)?;
            let c_exp = self.centroids.unsqueeze(0)?.unsqueeze(0)?;
            let diffs = y_exp.broadcast_sub(&c_exp)?.sqr()?;
            diffs.argmin(D::Minus1)?
        };

        Ok(MseQuantized { indices, norms })
    }

    /// Dequantize back to vectors.
    ///
    /// # Arguments
    /// * `q` - Quantized representation from [`quantize`](Self::quantize)
    ///
    /// # Returns
    /// Reconstructed tensor, shape `[n, d]`.
    pub fn dequantize(&self, q: &MseQuantized) -> Result<Tensor> {
        let (n, d) = q.indices.dims2()?;

        // Look up centroids by index
        let flat_idx = q.indices.flatten_all()?;
        let flat_vals = self.centroids.index_select(&flat_idx, 0)?;
        let y_recon = flat_vals.reshape((n, d))?;

        // Inverse rotation: x = y @ Π  (since Π^T = Π^{-1} for orthogonal Π)
        let x_recon = y_recon.matmul(&self.rotation)?;

        // Rescale by original norms
        x_recon.broadcast_mul(&q.norms.unsqueeze(D::Minus1)?)
    }

    /// Get the vector dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the bit width.
    pub fn bit_width(&self) -> usize {
        self.bit_width
    }
}

// ============================================================================
// TurboQuantProd
// ============================================================================

/// Inner-product-optimized TurboQuant vector quantizer.
///
/// Provides unbiased inner product estimation by:
/// 1. MSE-quantizing with b−1 bits per coordinate
/// 2. Applying 1-bit QJL (Quantized Johnson-Lindenstrauss) to the residual
///
/// # Properties
///
/// - **Unbiased**: E[⟨y, x̃⟩] = ⟨y, x⟩
/// - **Low distortion**: inner product error ≤ (√3·π²·‖y‖²/d) · 1/4^b
/// - For b=1,2,3,4: distortion ≈ 1.57/d, 0.56/d, 0.18/d, 0.047/d
pub struct TurboQuantProd {
    mse: TurboQuantMse,
    projection: Tensor,
    dim: usize,
    bit_width: usize,
    dtype: DType,
}

impl TurboQuantProd {
    /// Create a new inner-product-optimized TurboQuant quantizer.
    ///
    /// # Arguments
    /// * `dim` - Vector dimension (d).
    /// * `bit_width` - Total bits per coordinate (b ≥ 1). Uses b−1 for MSE + 1 for QJL.
    /// * `dtype` - Data type for internal computations.
    /// * `device` - Device for tensor allocation.
    pub fn new(dim: usize, bit_width: usize, dtype: DType, device: &Device) -> Result<Self> {
        if bit_width < 1 {
            return Err(candle::Error::Msg(
                "TurboQuantProd requires bit_width >= 1".into(),
            ));
        }

        let mse = TurboQuantMse::new(dim, bit_width - 1, dtype, device)?;
        // Random Gaussian projection matrix for QJL
        let projection = Tensor::randn(0f64, 1f64, (dim, dim), &Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(device)?;

        Ok(Self {
            mse,
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
    /// [`ProdQuantized`] with MSE indices, QJL signs, and norms.
    pub fn quantize(&self, x: &Tensor) -> Result<ProdQuantized> {
        let is_single = x.dims().len() == 1;
        let x = if is_single {
            x.unsqueeze(0)?
        } else {
            x.clone()
        };
        let x = x.to_dtype(self.dtype)?;

        // Step 1: MSE quantize with b-1 bits
        let mse_q = self.mse.quantize(&x)?;
        let x_mse = self.mse.dequantize(&mse_q)?;

        // Step 2: Compute residual r = x - DeQuant_mse(idx)
        let residual = x.broadcast_sub(&x_mse)?;
        let residual_norms = residual.sqr()?.sum(D::Minus1)?.sqrt()?;

        // Step 3: QJL on residual: sign(S · r)
        // Batch: r @ S^T gives S·r_i per row
        let sr = residual.matmul(&self.projection.t()?)?;
        let zeros = sr.zeros_like()?;
        let positive = sr.ge(&zeros)?;
        // Map U8 {0, 1} -> {-1.0, +1.0}
        let qjl_signs = positive.to_dtype(self.dtype)?.affine(2.0, -1.0)?;

        Ok(ProdQuantized {
            mse: mse_q,
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
    pub fn dequantize(&self, q: &ProdQuantized) -> Result<Tensor> {
        // Step 1: MSE dequantization
        let x_mse = self.mse.dequantize(&q.mse)?;

        // Step 2: QJL dequantization
        // x_qjl = √(π/2)/d · γ · S^T · qjl
        // Batch: qjl @ S gives S^T · qjl_i per row
        let x_qjl = q.qjl_signs.matmul(&self.projection)?;
        let scale = (std::f64::consts::FRAC_PI_2).sqrt() / self.dim as f64;
        let x_qjl = (x_qjl * scale)?;
        let x_qjl = x_qjl.broadcast_mul(&q.residual_norms.unsqueeze(D::Minus1)?)?;

        // Step 3: Combine MSE reconstruction + QJL correction
        x_mse.broadcast_add(&x_qjl)
    }

    /// Get the vector dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the bit width.
    pub fn bit_width(&self) -> usize {
        self.bit_width
    }
}

// ============================================================================
// TurboQuantKvCache
// ============================================================================

/// Concatenate a new tensor to an existing one along dim 0, or initialize.
fn cat_or_init(existing: &Option<Tensor>, new: &Tensor) -> Result<Tensor> {
    match existing {
        Some(prev) => Tensor::cat(&[prev, new], 0),
        None => Ok(new.clone()),
    }
}

/// Dequantize stored KV cache data back to a 4D tensor `[B, H, S, D]`.
fn dequantize_kv(
    quantizer: &TurboQuantMse,
    indices: &Tensor,
    norms: &Tensor,
    batch_size: usize,
    num_heads: usize,
    dtype: DType,
) -> Result<Tensor> {
    let indices_u32 = indices.to_dtype(DType::U32)?;
    let q = MseQuantized {
        indices: indices_u32,
        norms: norms.clone(),
    };
    let flat = quantizer.dequantize(&q)?;
    let n_total = flat.dims()[0];
    let head_dim = flat.dims()[1];
    let seq_len = n_total / (batch_size * num_heads);
    flat.reshape((batch_size, num_heads, seq_len, head_dim))?
        .to_dtype(dtype)
}

/// Quantized KV cache using TurboQuant for memory-efficient inference.
///
/// Drop-in replacement for [`crate::kv_cache::ConcatKvCache`] that stores
/// key/value embeddings in quantized form, reducing memory by up to 4× with
/// near-zero quality loss.
///
/// # How it works
///
/// On each [`append`](Self::append) call:
/// 1. New K/V tensors are quantized via [`TurboQuantMse`] (b bits per coordinate)
/// 2. Quantized representations (U8 indices + F32 norms) are concatenated with
///    the existing cache
/// 3. The full cache is dequantized and returned for attention computation
///
/// # Memory savings
///
/// Per vector (head_dim = D, bit_width = b):
/// - Unquantized: `4·D` bytes (F32) or `2·D` bytes (F16)
/// - Quantized: `D` bytes (U8 indices) + `4` bytes (F32 norm)
/// - With D=128: **~3.7× compression** vs F32
///
/// # Example
///
/// ```no_run
/// use candle::{Device, DType, Tensor};
/// use candle_nn::turboquant::TurboQuantKvCache;
///
/// let device = Device::Cpu;
/// let mut cache = TurboQuantKvCache::new(64, 4, DType::F32, &device).unwrap();
///
/// // Simulate attention: k/v of shape [batch, heads, seq_len, head_dim]
/// let k = Tensor::randn(0f32, 1f32, (1, 8, 10, 64), &device).unwrap();
/// let v = Tensor::randn(0f32, 1f32, (1, 8, 10, 64), &device).unwrap();
///
/// let (cached_k, cached_v) = cache.append(&k, &v).unwrap();
/// assert_eq!(cached_k.dims(), &[1, 8, 10, 64]);
/// ```
#[derive(Clone)]
pub struct TurboQuantKvCache {
    k_quantizer: TurboQuantMse,
    v_quantizer: TurboQuantMse,
    k_indices: Option<Tensor>,
    k_norms: Option<Tensor>,
    v_indices: Option<Tensor>,
    v_norms: Option<Tensor>,
    batch_size: usize,
    num_heads: usize,
    dtype: DType,
}

impl fmt::Debug for TurboQuantKvCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TurboQuantKvCache")
            .field("k_quantizer", &self.k_quantizer)
            .field("v_quantizer", &self.v_quantizer)
            .field("seq_len", &self.current_seq_len())
            .finish()
    }
}

impl TurboQuantKvCache {
    /// Create a new quantized KV cache.
    ///
    /// # Arguments
    /// * `head_dim` - Dimension of each attention head (D in `[B, H, S, D]`).
    /// * `bit_width` - Quantization bits per coordinate (typically 2–4).
    /// * `dtype` - Data type of the original K/V tensors.
    /// * `device` - Device for tensor allocation.
    pub fn new(head_dim: usize, bit_width: usize, dtype: DType, device: &Device) -> Result<Self> {
        let internal_dtype = if dtype == DType::F64 {
            DType::F64
        } else {
            DType::F32
        };
        let k_quantizer = TurboQuantMse::new(head_dim, bit_width, internal_dtype, device)?;
        let v_quantizer = TurboQuantMse::new(head_dim, bit_width, internal_dtype, device)?;

        Ok(Self {
            k_quantizer,
            v_quantizer,
            k_indices: None,
            k_norms: None,
            v_indices: None,
            v_norms: None,
            batch_size: 0,
            num_heads: 0,
            dtype,
        })
    }

    /// Append new key/value tensors and return the full dequantized cache.
    ///
    /// Input tensors must have shape `[batch, heads, seq_len, head_dim]` (the
    /// standard transformer layout with sequence on dim 2). Returns the full
    /// accumulated (dequantized) K and V tensors in the same layout.
    ///
    /// This matches the API of [`crate::kv_cache::ConcatKvCache::append`].
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let k_dims = k.dims();
        if k_dims.len() != 4 {
            return Err(candle::Error::Msg(format!(
                "TurboQuantKvCache expects 4D tensors [B, H, S, D], got {}D",
                k_dims.len()
            )));
        }
        let (b, h, s, d) = (k_dims[0], k_dims[1], k_dims[2], k_dims[3]);
        self.batch_size = b;
        self.num_heads = h;
        self.dtype = k.dtype();

        // Flatten to [B*H*S, D] for per-vector quantization
        let n = b * h * s;
        let k_flat = k.reshape((n, d))?;
        let v_flat = v.reshape((n, d))?;

        let k_q = self.k_quantizer.quantize(&k_flat)?;
        let v_q = self.v_quantizer.quantize(&v_flat)?;

        // Store indices as U8 for memory efficiency (valid for bit_width ≤ 8)
        let k_idx = k_q.indices.to_dtype(DType::U8)?;
        let v_idx = v_q.indices.to_dtype(DType::U8)?;

        self.k_indices = Some(cat_or_init(&self.k_indices, &k_idx)?);
        self.k_norms = Some(cat_or_init(&self.k_norms, &k_q.norms)?);
        self.v_indices = Some(cat_or_init(&self.v_indices, &v_idx)?);
        self.v_norms = Some(cat_or_init(&self.v_norms, &v_q.norms)?);

        // Dequantize full cache for attention
        let full_k = dequantize_kv(
            &self.k_quantizer,
            self.k_indices.as_ref().unwrap(),
            self.k_norms.as_ref().unwrap(),
            b,
            h,
            self.dtype,
        )?;
        let full_v = dequantize_kv(
            &self.v_quantizer,
            self.v_indices.as_ref().unwrap(),
            self.v_norms.as_ref().unwrap(),
            b,
            h,
            self.dtype,
        )?;

        Ok((full_k, full_v))
    }

    /// Number of cached sequence positions.
    pub fn current_seq_len(&self) -> usize {
        match &self.k_indices {
            Some(indices) if self.batch_size > 0 && self.num_heads > 0 => {
                indices.dims()[0] / (self.batch_size * self.num_heads)
            }
            _ => 0,
        }
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.k_indices.is_none()
    }

    /// Clear all cached data.
    pub fn reset(&mut self) {
        self.k_indices = None;
        self.k_norms = None;
        self.v_indices = None;
        self.v_norms = None;
    }

    /// Dequantize and return the full cached key tensor, if any.
    pub fn k(&self) -> Result<Option<Tensor>> {
        match (&self.k_indices, &self.k_norms) {
            (Some(indices), Some(norms)) => Ok(Some(dequantize_kv(
                &self.k_quantizer,
                indices,
                norms,
                self.batch_size,
                self.num_heads,
                self.dtype,
            )?)),
            _ => Ok(None),
        }
    }

    /// Dequantize and return the full cached value tensor, if any.
    pub fn v(&self) -> Result<Option<Tensor>> {
        match (&self.v_indices, &self.v_norms) {
            (Some(indices), Some(norms)) => Ok(Some(dequantize_kv(
                &self.v_quantizer,
                indices,
                norms,
                self.batch_size,
                self.num_heads,
                self.dtype,
            )?)),
            _ => Ok(None),
        }
    }
}

// ============================================================================
// KvCache: Unified cache enum for model integration
// ============================================================================

/// A KV cache that is either plain (unquantized) or TurboQuant-compressed.
///
/// This enum allows models to support both modes with the same code path.
/// When `kv_quant_bits` is specified in a model config, models can construct
/// a `KvCache::TurboQuant` variant; otherwise they use `KvCache::Plain`.
///
/// # Example (inside a model's attention layer)
///
/// ```no_run
/// use candle::{Device, DType, Tensor};
/// use candle_nn::turboquant::KvCache;
///
/// // Plain (no quantization):
/// let mut cache = KvCache::plain(2);
///
/// // Quantized (4-bit TurboQuant):
/// let mut cache = KvCache::turbo_quant(64, 4, DType::F32, &Device::Cpu).unwrap();
///
/// // Both variants use the same append API:
/// // let (k, v) = cache.append(&k, &v)?;
/// ```
#[derive(Debug, Clone)]
pub enum KvCache {
    /// Standard unquantized cache (wraps [`crate::kv_cache::ConcatKvCache`]).
    Plain(crate::kv_cache::ConcatKvCache),
    /// TurboQuant-compressed cache.
    TurboQuant(TurboQuantKvCache),
}

impl KvCache {
    /// Create an unquantized KV cache that concatenates along `dim`.
    pub fn plain(dim: usize) -> Self {
        Self::Plain(crate::kv_cache::ConcatKvCache::new(dim))
    }

    /// Create a TurboQuant-compressed KV cache.
    ///
    /// # Arguments
    /// * `head_dim` - Attention head dimension.
    /// * `bit_width` - Quantization bits per coordinate (typically 2–4).
    /// * `dtype` - Data type of K/V tensors.
    /// * `device` - Device for tensor allocation.
    pub fn turbo_quant(
        head_dim: usize,
        bit_width: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self::TurboQuant(TurboQuantKvCache::new(
            head_dim, bit_width, dtype, device,
        )?))
    }

    /// Append new K/V tensors and return the full accumulated cache.
    ///
    /// Works identically for both plain and quantized variants.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Plain(c) => c.append(k, v),
            Self::TurboQuant(c) => c.append(k, v),
        }
    }

    /// Number of cached sequence positions.
    pub fn current_seq_len(&self) -> usize {
        match self {
            Self::Plain(c) => c.current_seq_len(),
            Self::TurboQuant(c) => c.current_seq_len(),
        }
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Plain(c) => c.is_empty(),
            Self::TurboQuant(c) => c.is_empty(),
        }
    }

    /// Clear all cached data.
    pub fn reset(&mut self) {
        match self {
            Self::Plain(c) => c.reset(),
            Self::TurboQuant(c) => c.reset(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lloyd_max_1bit() {
        let centroids = lloyd_max_gaussian(1, 1000);
        assert_eq!(centroids.len(), 2);
        // For N(0,1), 1-bit optimal centroids are ±√(2/π) ≈ ±0.7979
        let expected = (2.0 / std::f64::consts::PI).sqrt();
        assert!(
            (centroids[0] + expected).abs() < 1e-4,
            "c0={}, expected={}",
            centroids[0],
            -expected
        );
        assert!(
            (centroids[1] - expected).abs() < 1e-4,
            "c1={}, expected={}",
            centroids[1],
            expected
        );
    }

    #[test]
    fn test_lloyd_max_2bit() {
        let centroids = lloyd_max_gaussian(2, 1000);
        assert_eq!(centroids.len(), 4);
        // Expected: ≈ ±0.4528, ±1.5104
        assert!((centroids[0] + 1.51).abs() < 0.01, "c0={}", centroids[0]);
        assert!((centroids[1] + 0.4528).abs() < 0.01, "c1={}", centroids[1]);
        assert!((centroids[2] - 0.4528).abs() < 0.01, "c2={}", centroids[2]);
        assert!((centroids[3] - 1.51).abs() < 0.01, "c3={}", centroids[3]);
    }

    #[test]
    fn test_lloyd_max_symmetry() {
        // Centroids should be symmetric around 0 for symmetric distributions
        for b in 1..=4 {
            let centroids = lloyd_max_gaussian(b, 1000);
            let n = centroids.len();
            for i in 0..n / 2 {
                assert!(
                    (centroids[i] + centroids[n - 1 - i]).abs() < 1e-6,
                    "Asymmetry at b={}, i={}: {} vs {}",
                    b,
                    i,
                    centroids[i],
                    centroids[n - 1 - i]
                );
            }
        }
    }

    #[test]
    fn test_mse_quantize_dequantize() -> Result<()> {
        let device = Device::Cpu;
        let dim = 64;
        let bit_width = 4;
        let n = 8;

        let quantizer = TurboQuantMse::new(dim, bit_width, DType::F32, &device)?;
        let x = Tensor::randn(0f32, 1f32, (n, dim), &device)?;

        let q = quantizer.quantize(&x)?;
        let x_recon = quantizer.dequantize(&q)?;

        // Check shapes
        assert_eq!(q.indices.dims(), &[n, dim]);
        assert_eq!(q.norms.dims(), &[n]);
        assert_eq!(x_recon.dims(), &[n, dim]);

        // Relative MSE should be small for 4-bit quantization
        let diff = x.broadcast_sub(&x_recon)?;
        let mse_total = diff.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        let norm_sq_total = x.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        let relative_mse = mse_total / norm_sq_total;
        assert!(
            relative_mse < 0.05,
            "4-bit relative MSE too high: {relative_mse}"
        );

        Ok(())
    }

    #[test]
    fn test_mse_preserves_norms() -> Result<()> {
        let device = Device::Cpu;
        let dim = 64;
        let bit_width = 3;
        let n = 4;

        let quantizer = TurboQuantMse::new(dim, bit_width, DType::F32, &device)?;

        // Create vectors with known norms
        let x = Tensor::randn(0f32, 1f32, (n, dim), &device)?;
        let q = quantizer.quantize(&x)?;
        let x_recon = quantizer.dequantize(&q)?;

        // Reconstructed norms should roughly preserve original norms
        let orig_norms: Vec<f32> = x.sqr()?.sum(D::Minus1)?.sqrt()?.to_vec1()?;
        let recon_norms: Vec<f32> = x_recon.sqr()?.sum(D::Minus1)?.sqrt()?.to_vec1()?;

        for i in 0..n {
            let ratio = recon_norms[i] / orig_norms[i];
            assert!(
                (0.5..2.0).contains(&ratio),
                "Norm ratio out of range for vec {i}: {ratio}"
            );
        }

        Ok(())
    }

    #[test]
    fn test_mse_single_vector() -> Result<()> {
        let device = Device::Cpu;
        let dim = 32;

        let quantizer = TurboQuantMse::new(dim, 2, DType::F32, &device)?;
        let x = Tensor::randn(0f32, 1f32, (dim,), &device)?;

        let q = quantizer.quantize(&x)?;
        let x_recon = quantizer.dequantize(&q)?;

        assert_eq!(q.indices.dims(), &[1, dim]);
        assert_eq!(x_recon.dims(), &[1, dim]);

        Ok(())
    }

    #[test]
    fn test_mse_index_range() -> Result<()> {
        let device = Device::Cpu;
        let dim = 32;

        for bit_width in 1..=4 {
            let quantizer = TurboQuantMse::new(dim, bit_width, DType::F32, &device)?;
            let x = Tensor::randn(0f32, 1f32, (4, dim), &device)?;
            let q = quantizer.quantize(&x)?;

            let max_idx = q.indices.max_all()?.to_scalar::<u32>()?;
            assert!(
                max_idx < (1u32 << bit_width),
                "Index {} out of range for {}-bit: max={}",
                max_idx,
                bit_width,
                (1u32 << bit_width) - 1
            );
        }

        Ok(())
    }

    #[test]
    fn test_prod_quantize_dequantize() -> Result<()> {
        let device = Device::Cpu;
        let dim = 64;
        let bit_width = 4;
        let n = 8;

        let quantizer = TurboQuantProd::new(dim, bit_width, DType::F32, &device)?;
        let x = Tensor::randn(0f32, 1f32, (n, dim), &device)?;

        let q = quantizer.quantize(&x)?;
        let x_recon = quantizer.dequantize(&q)?;

        assert_eq!(x_recon.dims(), &[n, dim]);

        // QJL signs must be exactly ±1
        let signs_abs = q.qjl_signs.abs()?;
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
    fn test_prod_unbiased() -> Result<()> {
        let device = Device::Cpu;
        let dim = 64;
        let bit_width = 3;
        let n_trials = 50;

        let x = Tensor::randn(0f32, 1f32, (1, dim), &device)?;
        let y = Tensor::randn(0f32, 1f32, (1, dim), &device)?;
        let true_ip = x.broadcast_mul(&y)?.sum_all()?.to_scalar::<f32>()? as f64;

        let mut ip_sum = 0.0;
        for _ in 0..n_trials {
            // Each trial uses a fresh random quantizer
            let quantizer = TurboQuantProd::new(dim, bit_width, DType::F32, &device)?;
            let q = quantizer.quantize(&x)?;
            let x_recon = quantizer.dequantize(&q)?;
            let ip = x_recon.broadcast_mul(&y)?.sum_all()?.to_scalar::<f32>()? as f64;
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
    fn test_prod_bit_width_1() -> Result<()> {
        // bit_width=1 means 0 bits for MSE + 1 bit for QJL (pure QJL)
        let device = Device::Cpu;
        let dim = 32;
        let quantizer = TurboQuantProd::new(dim, 1, DType::F32, &device)?;
        let x = Tensor::randn(0f32, 1f32, (4, dim), &device)?;

        let q = quantizer.quantize(&x)?;
        let x_recon = quantizer.dequantize(&q)?;

        assert_eq!(x_recon.dims(), &[4, dim]);
        Ok(())
    }

    #[test]
    fn test_mse_distortion_decreases_with_bits() -> Result<()> {
        let device = Device::Cpu;
        let dim = 128;
        let n = 16;
        let x = Tensor::randn(0f32, 1f32, (n, dim), &device)?;
        let norm_sq = x.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;

        let mut prev_mse = f64::MAX;
        for bit_width in 1..=4 {
            let quantizer = TurboQuantMse::new(dim, bit_width, DType::F32, &device)?;
            let q = quantizer.quantize(&x)?;
            let x_recon = quantizer.dequantize(&q)?;

            let mse = x
                .broadcast_sub(&x_recon)?
                .sqr()?
                .sum_all()?
                .to_scalar::<f32>()? as f64
                / norm_sq;

            assert!(
                mse < prev_mse,
                "MSE should decrease: {bit_width}-bit MSE={mse:.6} >= prev={prev_mse:.6}"
            );
            prev_mse = mse;
        }

        Ok(())
    }

    // ================================================================
    // TurboQuantKvCache tests
    // ================================================================

    #[test]
    fn test_kv_cache_append_shape() -> Result<()> {
        let device = Device::Cpu;
        let head_dim = 64;
        let (batch, heads, seq) = (1, 8, 10);

        let mut cache = TurboQuantKvCache::new(head_dim, 4, DType::F32, &device)?;
        let k = Tensor::randn(0f32, 1f32, (batch, heads, seq, head_dim), &device)?;
        let v = Tensor::randn(0f32, 1f32, (batch, heads, seq, head_dim), &device)?;

        let (ck, cv) = cache.append(&k, &v)?;
        assert_eq!(ck.dims(), &[batch, heads, seq, head_dim]);
        assert_eq!(cv.dims(), &[batch, heads, seq, head_dim]);
        assert_eq!(cache.current_seq_len(), seq);

        Ok(())
    }

    #[test]
    fn test_kv_cache_incremental_decode() -> Result<()> {
        let device = Device::Cpu;
        let head_dim = 64;
        let (batch, heads) = (1, 4);

        let mut cache = TurboQuantKvCache::new(head_dim, 4, DType::F32, &device)?;

        // Prefill: 8 tokens
        let k0 = Tensor::randn(0f32, 1f32, (batch, heads, 8, head_dim), &device)?;
        let v0 = Tensor::randn(0f32, 1f32, (batch, heads, 8, head_dim), &device)?;
        let (ck, cv) = cache.append(&k0, &v0)?;
        assert_eq!(ck.dims(), &[batch, heads, 8, head_dim]);
        assert_eq!(cache.current_seq_len(), 8);

        // Decode step 1: 1 new token
        let k1 = Tensor::randn(0f32, 1f32, (batch, heads, 1, head_dim), &device)?;
        let v1 = Tensor::randn(0f32, 1f32, (batch, heads, 1, head_dim), &device)?;
        let (ck, cv) = cache.append(&k1, &v1)?;
        assert_eq!(ck.dims(), &[batch, heads, 9, head_dim]);
        assert_eq!(cv.dims(), &[batch, heads, 9, head_dim]);
        assert_eq!(cache.current_seq_len(), 9);

        // Decode step 2: 1 more token
        let k2 = Tensor::randn(0f32, 1f32, (batch, heads, 1, head_dim), &device)?;
        let v2 = Tensor::randn(0f32, 1f32, (batch, heads, 1, head_dim), &device)?;
        let (ck, _cv) = cache.append(&k2, &v2)?;
        assert_eq!(ck.dims(), &[batch, heads, 10, head_dim]);
        assert_eq!(cache.current_seq_len(), 10);

        Ok(())
    }

    #[test]
    fn test_kv_cache_attention_quality() -> Result<()> {
        let device = Device::Cpu;
        let head_dim = 64;
        let (batch, heads, seq_len) = (1, 4, 16);

        // Generate Q, K, V
        let q = Tensor::randn(0f32, 1f32, (batch, heads, 1, head_dim), &device)?;
        let k = Tensor::randn(0f32, 1f32, (batch, heads, seq_len, head_dim), &device)?;
        let v = Tensor::randn(0f32, 1f32, (batch, heads, seq_len, head_dim), &device)?;

        // Unquantized attention scores
        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores_exact = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        // Quantized KV cache attention scores
        let mut cache = TurboQuantKvCache::new(head_dim, 4, DType::F32, &device)?;
        let (ck, _cv) = cache.append(&k, &v)?;
        let scores_quant = (q.matmul(&ck.transpose(2, 3)?)? * scale)?;

        // Attention score error should be small for 4-bit
        let score_err = scores_exact
            .broadcast_sub(&scores_quant)?
            .sqr()?
            .mean_all()?
            .to_scalar::<f32>()? as f64;
        let score_mag = scores_exact.sqr()?.mean_all()?.to_scalar::<f32>()? as f64;
        let rel_err = score_err / (score_mag + 1e-10);

        assert!(
            rel_err < 0.1,
            "Attention score relative error too high: {rel_err:.4}"
        );

        Ok(())
    }

    #[test]
    fn test_kv_cache_reset() -> Result<()> {
        let device = Device::Cpu;
        let head_dim = 32;

        let mut cache = TurboQuantKvCache::new(head_dim, 3, DType::F32, &device)?;
        let k = Tensor::randn(0f32, 1f32, (1, 2, 5, head_dim), &device)?;
        let v = Tensor::randn(0f32, 1f32, (1, 2, 5, head_dim), &device)?;

        cache.append(&k, &v)?;
        assert!(!cache.is_empty());
        assert_eq!(cache.current_seq_len(), 5);

        cache.reset();
        assert!(cache.is_empty());
        assert_eq!(cache.current_seq_len(), 0);
        assert!(cache.k()?.is_none());
        assert!(cache.v()?.is_none());

        Ok(())
    }

    #[test]
    fn test_kv_cache_k_v_accessors() -> Result<()> {
        let device = Device::Cpu;
        let head_dim = 32;
        let (batch, heads, seq) = (1, 2, 4);

        let mut cache = TurboQuantKvCache::new(head_dim, 4, DType::F32, &device)?;

        assert!(cache.k()?.is_none());
        assert!(cache.v()?.is_none());

        let k = Tensor::randn(0f32, 1f32, (batch, heads, seq, head_dim), &device)?;
        let v = Tensor::randn(0f32, 1f32, (batch, heads, seq, head_dim), &device)?;
        cache.append(&k, &v)?;

        let cached_k = cache.k()?.unwrap();
        let cached_v = cache.v()?.unwrap();
        assert_eq!(cached_k.dims(), &[batch, heads, seq, head_dim]);
        assert_eq!(cached_v.dims(), &[batch, heads, seq, head_dim]);

        Ok(())
    }
}
