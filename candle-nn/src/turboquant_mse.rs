//! TurboQuant MSE: MSE-Optimal Vector Quantization via Random Rotation
//!
//! Implementation of Algorithm 1 (TurboQuant_mse) from:
//! "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
//! ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874))
//!
//! Inspired by PolarQuant ([arXiv:2502.02617](https://arxiv.org/abs/2502.02617)),
//! which introduced random preconditioning for KV cache quantization.
//!
//! TurboQuantMse achieves near-optimal MSE distortion for vector quantization by:
//! 1. **Random rotation**: Multiplies input by a random orthogonal matrix Π,
//!    spreading vector energy evenly across coordinates so each follows an
//!    approximately Gaussian (Beta) distribution.
//! 2. **Scalar quantization**: Applies optimal Lloyd-Max scalar quantizers
//!    independently per coordinate, eliminating the need for learned codebooks.
//!
//! This makes TurboQuantMse **data-oblivious** — no training or calibration is
//! needed. It is suitable for online, streaming, or one-shot quantization of
//! embeddings, activations, KV cache vectors, or any high-dimensional data.
//!
//! # Relationship to TurboQuant
//!
//! TurboQuantMse is the foundational MSE stage used by
//! [`TurboQuant`](crate::turboquant::TurboQuant), which adds a 1-bit QJL
//! correction for unbiased inner product estimation. TurboQuantMse can be used
//! independently when MSE-optimal reconstruction is sufficient.
//!
//! # Example
//!
//! ```no_run
//! use candle::{Device, DType, Tensor};
//! use candle_nn::turboquant_mse::TurboQuantMse;
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

/// Generate a random orthogonal matrix via Householder QR decomposition.
///
/// Generates a d×d random Gaussian matrix, computes its QR factorization
/// using Householder reflections, and returns Q. This matches the TurboQuant
/// paper's specification and is both faster and more numerically stable than
/// Modified Gram-Schmidt.
fn random_orthogonal_tensor(dim: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let random_mat = Tensor::randn(0f64, 1f64, (dim, dim), &Device::Cpu)?;
    let a: Vec<Vec<f64>> = random_mat.to_vec2()?;

    // Column-major storage for in-place Householder QR
    let mut col_major = vec![0f64; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            col_major[j * dim + i] = a[i][j];
        }
    }

    // Store Householder vectors' essential parts (below diagonal) in-place
    let mut tau = vec![0f64; dim];

    for k in 0..dim {
        // Extract column k below the diagonal
        let mut norm_sq = 0f64;
        for i in k..dim {
            norm_sq += col_major[k * dim + i] * col_major[k * dim + i];
        }
        let norm = norm_sq.sqrt();

        if norm < 1e-15 {
            tau[k] = 0.0;
            continue;
        }

        // Choose sign to avoid cancellation
        let alpha = col_major[k * dim + k];
        let beta = if alpha >= 0.0 { -norm } else { norm };

        // Compute Householder vector v with v[k] = 1 (implicit)
        // tau = 2 / (v^T v)
        let denom = alpha - beta;
        if denom.abs() < 1e-15 {
            tau[k] = 0.0;
            col_major[k * dim + k] = beta;
            continue;
        }

        let inv_denom = 1.0 / denom;
        for i in (k + 1)..dim {
            col_major[k * dim + i] *= inv_denom;
        }
        col_major[k * dim + k] = beta;

        // tau[k] = (beta - alpha) / beta = -(alpha - beta) / beta
        tau[k] = (beta - alpha) / beta;

        // Apply H_k to remaining columns: A[k:, k+1:] -= tau * v * (v^T * A[k:, k+1:])
        for j in (k + 1)..dim {
            // dot = v^T * A[k:, j] (v[k] = 1 implicitly)
            let mut dot = col_major[j * dim + k];
            for i in (k + 1)..dim {
                dot += col_major[k * dim + i] * col_major[j * dim + i];
            }
            dot *= tau[k];
            // A[k:, j] -= dot * v
            col_major[j * dim + k] -= dot;
            for i in (k + 1)..dim {
                col_major[j * dim + i] -= dot * col_major[k * dim + i];
            }
        }
    }

    // Accumulate Q = H_0 * H_1 * ... * H_{d-1} by backward accumulation
    // Start with Q = I, then apply reflectors from right to left
    let mut q = vec![0f64; dim * dim];
    for i in 0..dim {
        q[i * dim + i] = 1.0;
    }

    for k in (0..dim).rev() {
        if tau[k].abs() < 1e-15 {
            continue;
        }
        // Apply H_k to Q[k:, k:]: Q[k:, j] -= tau * v * (v^T * Q[k:, j])
        for j in k..dim {
            let mut dot = q[j * dim + k]; // v[k] = 1
            for i in (k + 1)..dim {
                dot += col_major[k * dim + i] * q[j * dim + i];
            }
            dot *= tau[k];
            q[j * dim + k] -= dot;
            for i in (k + 1)..dim {
                q[j * dim + i] -= dot * col_major[k * dim + i];
            }
        }
    }

    // Convert column-major Q to row-major flat array
    let mut flat = vec![0f64; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            flat[i * dim + j] = q[j * dim + i];
        }
    }

    Tensor::from_vec(flat, (dim, dim), &Device::Cpu)?
        .to_dtype(dtype)?
        .to_device(device)
}

/// Apply an in-place Walsh-Hadamard transform to `data` of length `n` (must be power of 2).
///
/// The transform is unnormalized — caller is responsible for scaling by 1/√n.
fn hadamard_transform_inplace(data: &mut [f64], n: usize) {
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let x = data[j];
                let y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
        }
        h *= 2;
    }
}

/// Generate a randomized Hadamard rotation matrix as a Tensor.
///
/// Uses a Walsh-Hadamard matrix H_d (O(d log d) to apply) with random sign
/// flips D (diagonal ±1 matrix) to form the rotation Π = (1/√d) · H · D.
/// This is an orthogonal transformation that is O(d log d) to apply, compared
/// to O(d²) for a dense orthogonal matrix.
///
/// `dim` must be a power of 2.
/// Generate a randomized Hadamard rotation matrix and its sign vector.
///
/// Returns `(matrix_tensor, signs_f32)` where signs can be used for fast
/// O(d log d) application without the dense matrix.
fn randomized_hadamard_matrix(
    dim: usize,
    dtype: DType,
    device: &Device,
) -> Result<(Tensor, Vec<f32>)> {
    if !dim.is_power_of_two() {
        candle::bail!(
            "Hadamard rotation requires dim to be a power of 2, got {}",
            dim
        );
    }

    // Generate random signs for D: each ±1 with equal probability
    let sign_tensor = Tensor::randn(0f64, 1f64, (dim,), &Device::Cpu)?;
    let signs: Vec<f64> = sign_tensor.to_vec1()?;
    let signs: Vec<f64> = signs
        .iter()
        .map(|&s| if s >= 0.0 { 1.0 } else { -1.0 })
        .collect();

    // Build the matrix row by row: row_i = (1/√d) * H * D * e_i
    // This is equivalent to column i of (1/√d) * H * D
    let scale = 1.0 / (dim as f64).sqrt();
    let mut matrix = vec![0f64; dim * dim];

    for i in 0..dim {
        // Start with e_i scaled by sign
        let mut col = vec![0f64; dim];
        col[i] = signs[i];

        // Apply Hadamard transform
        hadamard_transform_inplace(&mut col, dim);

        // Scale and store as row i of the transposed matrix
        // We want rotation[i][j] = scale * H[j][i] * signs[i]
        // but since H is symmetric, H[j][i] = H[i][j], so we store as column
        for j in 0..dim {
            matrix[j * dim + i] = col[j] * scale;
        }
    }

    let signs_f32: Vec<f32> = signs.iter().map(|&s| s as f32).collect();

    Tensor::from_vec(matrix, (dim, dim), &Device::Cpu)?
        .to_dtype(dtype)?
        .to_device(device)
        .map(|t| (t, signs_f32))
}

/// Result of TurboQuantMse quantization.
///
/// Stores centroid indices (b bits per coordinate) and original vector norms.
/// The indices select from Lloyd-Max centroids computed for the standard normal
/// distribution, scaled by 1/√d.
#[derive(Debug, Clone)]
pub struct TurboMseQuantized {
    /// Centroid indices. Shape: `[n, d]`, dtype: `U32`.
    /// Each value is in `[0, 2^b)`.
    pub indices: Tensor,
    /// Original L2 norms. Shape: `[n]`.
    pub norms: Tensor,
}

/// MSE-optimal vector quantizer using random rotation and Lloyd-Max codebooks.
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
///
/// # Example
///
/// ```no_run
/// use candle::{Device, DType, Tensor};
/// use candle_nn::turboquant_mse::TurboQuantMse;
///
/// let device = Device::Cpu;
/// let quantizer = TurboQuantMse::new(128, 4, DType::F32, &device).unwrap();
///
/// let x = Tensor::randn(0f32, 1f32, (10, 128), &device).unwrap();
/// let quantized = quantizer.quantize(&x).unwrap();
/// let reconstructed = quantizer.dequantize(&quantized).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct TurboQuantMse {
    dim: usize,
    bit_width: usize,
    rotation: Tensor,
    centroids: Tensor,
    dtype: DType,
    /// Random sign vector for fast Hadamard path. None for dense rotation.
    hadamard_signs: Option<Vec<f32>>,
}

impl TurboQuantMse {
    /// Create a new TurboQuantMse quantizer.
    ///
    /// Generates a random orthogonal rotation matrix and computes Lloyd-Max
    /// centroids for the standard normal distribution, scaled by 1/√d.
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

        let centroids = Tensor::from_vec(scaled, (unit_centroids.len(),), &Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(device)?;
        let rotation = random_orthogonal_tensor(dim, dtype, device)?;

        Ok(Self {
            dim,
            bit_width,
            rotation,
            centroids,
            dtype,
            hadamard_signs: None,
        })
    }

    /// Create a new TurboQuantMse quantizer using fast Hadamard rotation.
    ///
    /// Uses a randomized Walsh-Hadamard transform instead of a dense orthogonal
    /// matrix. Construction is O(d²) but the resulting rotation matrix enables
    /// the fused matmul kernel to apply inverse rotation in O(d log d) per row.
    ///
    /// Requires `dim` to be a power of 2 (32, 64, 128, 256, ...).
    ///
    /// # Arguments
    /// * `dim` - Vector dimension (d). Must be a power of 2.
    /// * `bit_width` - Bits per coordinate (b). Typically 1–8.
    /// * `dtype` - Data type for internal computations (F32 or F64).
    /// * `device` - Device for tensor allocation.
    pub fn new_hadamard(
        dim: usize,
        bit_width: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let unit_centroids = lloyd_max_gaussian(bit_width, 1000);
        let scale = 1.0 / (dim as f64).sqrt();
        let scaled: Vec<f64> = unit_centroids.iter().map(|&c| c * scale).collect();

        let centroids = Tensor::from_vec(scaled, (unit_centroids.len(),), &Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(device)?;
        let (rotation, signs) = randomized_hadamard_matrix(dim, dtype, device)?;

        Ok(Self {
            dim,
            bit_width,
            rotation,
            centroids,
            dtype,
            hadamard_signs: Some(signs),
        })
    }

    /// Quantize vectors.
    ///
    /// # Arguments
    /// * `x` - Input tensor, shape `[n, d]` or `[d]`
    ///
    /// # Returns
    /// [`TurboMseQuantized`] with centroid indices and original norms.
    pub fn quantize(&self, x: &Tensor) -> Result<TurboMseQuantized> {
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

        // Random rotation: y = x_norm @ Π^T
        // For Hadamard: Π^T = (1/√d) · D · H, applied row-wise in O(d log d) on CPU
        let y = if let Some(ref signs) = self.hadamard_signs {
            let d = self.dim;
            let n_rows = x_norm.dims()[0];
            let scale = 1.0 / (d as f32).sqrt();
            // Pull to CPU for the in-place butterfly transform
            let mut data: Vec<f32> = x_norm
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1()?;
            for row in 0..n_rows {
                let r = &mut data[row * d..(row + 1) * d];
                for (val, &sign) in r.iter_mut().zip(signs.iter()) {
                    *val *= sign;
                }
                hadamard_transform_inplace_f32(r, d);
                for val in r.iter_mut() {
                    *val *= scale;
                }
            }
            Tensor::from_vec(data, (n_rows, d), &Device::Cpu)?
                .to_dtype(self.dtype)?
                .to_device(x_norm.device())?
        } else {
            x_norm.matmul(&self.rotation.t()?)?
        };

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

        Ok(TurboMseQuantized { indices, norms })
    }

    /// Dequantize back to vectors.
    ///
    /// # Arguments
    /// * `q` - Quantized representation from [`quantize`](Self::quantize)
    ///
    /// # Returns
    /// Reconstructed tensor, shape `[n, d]`.
    pub fn dequantize(&self, q: &TurboMseQuantized) -> Result<Tensor> {
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

    /// Get a reference to the centroid values.
    pub fn centroids(&self) -> &Tensor {
        &self.centroids
    }

    /// Get a reference to the rotation matrix.
    pub fn rotation(&self) -> &Tensor {
        &self.rotation
    }

    /// Get the centroids as a CPU f32 slice (for fused ops).
    pub fn centroids_f32(&self) -> Result<Vec<f32>> {
        self.centroids.to_dtype(DType::F32)?.to_vec1()
    }

    /// Get the rotation matrix as a CPU f32 slice (for fused ops).
    pub fn rotation_f32(&self) -> Result<Vec<f32>> {
        self.rotation.to_dtype(DType::F32)?.flatten_all()?.to_vec1()
    }

    /// Get the Hadamard sign vector, if this quantizer uses Hadamard rotation.
    pub fn hadamard_signs(&self) -> Option<&[f32]> {
        self.hadamard_signs.as_deref()
    }
}

/// Fused codebook-matmul operation for TurboQuantMse-compressed weights.
///
/// Stores weights as bit-packed indices + norms + a pre-rotated codebook.
/// The codebook contains `2^b` vectors of dimension `d`, pre-multiplied by
/// the inverse rotation matrix so inference needs no rotation step.
///
/// Forward pass: `output[j] = norms[j] * Σ_i x[i] * codebook[idx[j][i]][i]`
/// — a single fused kernel with no rotation, no full-weight materialization.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TurboMseMatMul {
    /// Rotation matrix transpose: `[d, d]` F32 — on target device (fallback path).
    rotation_t: Tensor,
    /// Scalar centroid table: `[2^b]` F32 — on target device.
    centroids: Tensor,
    /// Bit-packed weight indices: `[packed_len]` U8 — on target device.
    packed_indices: Tensor,
    /// Per-row weight norms: `[n_out]` F32 — on target device.
    norms: Tensor,
    /// Hadamard sign vector: `[d]` F32 — on target device. None for dense rotation.
    signs: Option<Tensor>,
    /// CPU copies for CPU path.
    packed_indices_cpu: Vec<u8>,
    norms_cpu: Vec<f32>,
    centroids_cpu: Vec<f32>,
    /// Bits per index.
    bit_width: usize,
    /// Number of output features.
    out_features: usize,
    /// Number of input features (= dim).
    in_features: usize,
}

/// Pack b-bit indices into bytes, little-endian within each byte.
fn pack_indices(indices: &[u8], bit_width: usize) -> Vec<u8> {
    if bit_width >= 8 {
        return indices.to_vec();
    }
    let mask = (1u8 << bit_width) - 1;
    let indices_per_byte = 8 / bit_width;
    let packed_len = indices.len().div_ceil(indices_per_byte);
    let mut packed = vec![0u8; packed_len];
    for (i, &idx) in indices.iter().enumerate() {
        let byte_pos = i / indices_per_byte;
        let bit_offset = (i % indices_per_byte) * bit_width;
        packed[byte_pos] |= (idx & mask) << bit_offset;
    }
    packed
}

/// Unpack a single b-bit index from packed storage.
#[inline(always)]
fn unpack_index(packed: &[u8], pos: usize, bit_width: usize) -> usize {
    let indices_per_byte = 8 / bit_width;
    let byte_pos = pos / indices_per_byte;
    let bit_offset = (pos % indices_per_byte) * bit_width;
    let mask = (1u8 << bit_width) - 1;
    ((packed[byte_pos] >> bit_offset) & mask) as usize
}

impl TurboMseMatMul {
    /// Create a fused matmul op with compressed GPU storage.
    ///
    /// Stores 4-bit packed indices, norms, centroids, and rotation matrix
    /// directly on the target device. Forward does:
    /// 1. `x_rot = x @ Π^T` (BLAS matmul, rotate input)
    /// 2. `output[j] = norms[j] * Σ_i x_rot[i] * centroids[idx[j][i]]` (from compressed indices)
    ///
    /// No F32 weight matrix is ever materialized in GPU memory.
    pub fn new(quantizer: &TurboQuantMse, indices: &Tensor, norms: &Tensor) -> Result<Self> {
        let bit_width = quantizer.bit_width();
        if bit_width == 0 || bit_width > 7 {
            candle::bail!(
                "TurboMseMatMul requires bit_width in 1..=7, got {}",
                bit_width
            );
        }
        let (out_features, in_features) = indices.dims2()?;
        let indices_u8: Vec<u8> = indices.to_dtype(DType::U8)?.flatten_all()?.to_vec1()?;
        let packed_cpu = pack_indices(&indices_u8, bit_width);
        let norms_cpu: Vec<f32> = norms.to_dtype(DType::F32)?.to_vec1()?;
        let centroids_cpu = quantizer.centroids_f32()?;

        let device = quantizer.rotation().device();
        let rotation_t = quantizer
            .rotation()
            .t()?
            .contiguous()?
            .to_dtype(DType::F32)?
            .to_device(device)?;
        let centroids_tensor =
            Tensor::from_vec(centroids_cpu.clone(), (centroids_cpu.len(),), &Device::Cpu)?
                .to_device(device)?;
        let packed_tensor =
            Tensor::from_vec(packed_cpu.clone(), (packed_cpu.len(),), &Device::Cpu)?
                .to_device(device)?;
        let norms_tensor = Tensor::from_vec(norms_cpu.clone(), (norms_cpu.len(),), &Device::Cpu)?
            .to_device(device)?;

        // Store Hadamard signs on device for fused kernel
        let signs = quantizer
            .hadamard_signs()
            .map(|s| {
                Tensor::from_vec(s.to_vec(), (s.len(),), &Device::Cpu)
                    .and_then(|t| t.to_device(device))
            })
            .transpose()?;

        Ok(Self {
            rotation_t,
            centroids: centroids_tensor,
            packed_indices: packed_tensor,
            norms: norms_tensor,
            signs,
            packed_indices_cpu: packed_cpu,
            norms_cpu,
            centroids_cpu,
            bit_width,
            out_features,
            in_features,
        })
    }

    /// Number of output features.
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Packed index storage size in bytes.
    pub fn packed_index_bytes(&self) -> usize {
        self.packed_indices_cpu.len()
    }
}

/// Apply an in-place Walsh-Hadamard transform to f32 data.
#[inline]
fn hadamard_transform_inplace_f32(data: &mut [f32], n: usize) {
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let x = data[j];
                let y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
        }
        h *= 2;
    }
}

impl TurboMseMatMul {
    /// Compute `x @ W^T` using the pre-computed rotated weight matrix.
    ///
    /// Forward: `x @ W^T` from compressed 4-bit storage.
    ///
    /// 1. Rotate input: `x_rot = x @ Π^T` (BLAS matmul)
    /// 2. Centroid dot: `output[j] = norms[j] * Σ_i x_rot[i] * centroids[idx[j][i]]`
    ///    - Metal: custom shader reads packed indices from GPU memory
    ///    - CPU: scalar loop with bit unpacking
    ///
    /// No F32 weight matrix is materialized. GPU memory holds only the compressed
    /// indices (~8× smaller than F32 weights).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dims = x.dims().to_vec();
        let d = self.in_features;
        let n = self.out_features;

        let batch = x.elem_count() / d;
        let x_2d = if x.dims().len() == 1 {
            x.unsqueeze(0)?
        } else {
            x.reshape((batch, d))?
        };

        let y_2d = if x.device().is_cpu() {
            // CPU path: Hadamard rotation + scalar centroid dot
            let x_rot = x_2d.matmul(&self.rotation_t)?;
            let x_rot_data: Vec<f32> = x_rot.flatten_all()?.to_vec1()?;
            let mut output = vec![0f32; batch * n];
            for row in 0..n {
                let idx_start = row * d;
                let norm = self.norms_cpu[row];
                for b in 0..batch {
                    let xr = &x_rot_data[b * d..(b + 1) * d];
                    let mut dot = 0f32;
                    for (i, &xr_val) in xr.iter().enumerate() {
                        let idx =
                            unpack_index(&self.packed_indices_cpu, idx_start + i, self.bit_width);
                        dot += xr_val * self.centroids_cpu[idx];
                    }
                    output[b * n + row] = dot * norm;
                }
            }
            Tensor::from_vec(output, (batch, n), &Device::Cpu)?
        } else {
            // GPU path: try fused Hadamard kernel, fallback to rotation + centroid dot
            #[cfg(feature = "metal")]
            {
                use candle::{backend::BackendStorage, MetalStorage, Storage};

                if let Some(ref signs_tensor) = self.signs {
                    // Fused path: Hadamard in shared memory, no rotation matmul
                    let x_dev = x_2d.to_device(x.device())?.contiguous()?;
                    let scale = 1.0f32 / (d as f32).sqrt();

                    let (xs, xl) = x_dev.storage_and_layout();
                    let (ps, _) = self.packed_indices.storage_and_layout();
                    let (ns, _) = self.norms.storage_and_layout();
                    let (cs, _) = self.centroids.storage_and_layout();
                    let (ss, _) = signs_tensor.storage_and_layout();

                    let out_metal = match (&*xs, &*ps, &*ns, &*cs, &*ss) {
                        (
                            Storage::Metal(xm),
                            Storage::Metal(pm),
                            Storage::Metal(nm),
                            Storage::Metal(cm),
                            Storage::Metal(sm),
                        ) => {
                            let dev = xm.device();
                            let out_buf = dev.new_buffer(batch * n, DType::F32, "pq_fused")?;
                            let enc = dev.command_encoder()?;
                            candle_metal_kernels::call_turboquant_mse_fused_hadamard(
                                dev.device(),
                                &enc,
                                dev.kernels(),
                                self.bit_width,
                                d,
                                n,
                                batch,
                                xm.buffer(),
                                xl.start_offset() * DType::F32.size_in_bytes(),
                                pm.buffer(),
                                nm.buffer(),
                                cm.buffer(),
                                sm.buffer(),
                                scale,
                                &out_buf,
                            )
                            .map_err(candle::Error::wrap)?;
                            Ok::<_, candle::Error>(MetalStorage::new(
                                out_buf,
                                dev.clone(),
                                batch * n,
                                DType::F32,
                            ))
                        }
                        _ => candle::bail!("Expected Metal storage"),
                    }?;

                    drop(xs);
                    drop(ps);
                    drop(ns);
                    drop(cs);
                    drop(ss);

                    Tensor::from_storage(
                        Storage::Metal(out_metal),
                        (batch, n),
                        candle::op::BackpropOp::none(),
                        false,
                    )
                } else {
                    // Fallback: rotation matmul + centroid dot shader
                    let x_rot = x_2d.matmul(&self.rotation_t.to_device(x.device())?)?;

                    let (xs, xl) = x_rot.storage_and_layout();
                    let (ps, _) = self.packed_indices.storage_and_layout();
                    let (ns, _) = self.norms.storage_and_layout();
                    let (cs, _) = self.centroids.storage_and_layout();

                    let out_metal = match (&*xs, &*ps, &*ns, &*cs) {
                        (
                            Storage::Metal(xm),
                            Storage::Metal(pm),
                            Storage::Metal(nm),
                            Storage::Metal(cm),
                        ) => {
                            let dev = xm.device();
                            let out_buf = dev.new_buffer(batch * n, DType::F32, "pq_out")?;
                            let enc = dev.command_encoder()?;
                            candle_metal_kernels::call_turboquant_mse_centroid_dot(
                                dev.device(),
                                &enc,
                                dev.kernels(),
                                self.bit_width,
                                d,
                                n,
                                batch,
                                xm.buffer(),
                                xl.start_offset() * DType::F32.size_in_bytes(),
                                pm.buffer(),
                                nm.buffer(),
                                cm.buffer(),
                                &out_buf,
                            )
                            .map_err(candle::Error::wrap)?;
                            Ok::<_, candle::Error>(MetalStorage::new(
                                out_buf,
                                dev.clone(),
                                batch * n,
                                DType::F32,
                            ))
                        }
                        _ => candle::bail!("Expected Metal storage"),
                    }?;

                    drop(xs);
                    drop(ps);
                    drop(ns);
                    drop(cs);

                    Tensor::from_storage(
                        Storage::Metal(out_metal),
                        (batch, n),
                        candle::op::BackpropOp::none(),
                        false,
                    )
                }
            }
            #[cfg(not(feature = "metal"))]
            {
                candle::bail!("GPU TurboMseMatMul requires metal feature")
            }
        };

        match orig_dims.len() {
            1 => y_2d.squeeze(0),
            _ => {
                let mut out_dims = orig_dims;
                *out_dims.last_mut().unwrap() = n;
                y_2d.reshape(out_dims)
            }
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
    fn test_quantize_dequantize() -> Result<()> {
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
    fn test_preserves_norms() -> Result<()> {
        let device = Device::Cpu;
        let dim = 64;
        let bit_width = 3;
        let n = 4;

        let quantizer = TurboQuantMse::new(dim, bit_width, DType::F32, &device)?;

        let x = Tensor::randn(0f32, 1f32, (n, dim), &device)?;
        let q = quantizer.quantize(&x)?;
        let x_recon = quantizer.dequantize(&q)?;

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
    fn test_single_vector() -> Result<()> {
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
    fn test_index_range() -> Result<()> {
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
    fn test_distortion_decreases_with_bits() -> Result<()> {
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
}
