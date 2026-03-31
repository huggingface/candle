//! PolarQuant: MSE-Optimal Vector Quantization via Random Rotation
//!
//! Implementation of the PolarQuant algorithm from:
//! "PolarQuant: Quantizing with Random Rotations for Sharp Rate-Distortion"
//! ([arXiv:2502.02617](https://arxiv.org/abs/2502.02617))
//!
//! PolarQuant achieves near-optimal MSE distortion for vector quantization by:
//! 1. **Random rotation**: Multiplies input by a random orthogonal matrix Π,
//!    spreading vector energy evenly across coordinates so each follows an
//!    approximately Gaussian distribution.
//! 2. **Scalar quantization**: Applies optimal Lloyd-Max scalar quantizers
//!    independently per coordinate, eliminating the need for learned codebooks.
//!
//! This makes PolarQuant **data-oblivious** — no training or calibration is
//! needed. It is suitable for online, streaming, or one-shot quantization of
//! embeddings, activations, KV cache vectors, or any high-dimensional data.
//!
//! # Relationship to TurboQuant
//!
//! PolarQuant is the foundational MSE stage used by
//! [`TurboQuant`](crate::turboquant::TurboQuant), which adds a 1-bit QJL
//! correction for unbiased inner product estimation. PolarQuant can be used
//! independently when MSE-optimal reconstruction is sufficient.
//!
//! # Example
//!
//! ```no_run
//! use candle::{Device, DType, Tensor};
//! use candle_nn::polarquant::PolarQuant;
//!
//! let device = Device::Cpu;
//! let dim = 128;
//! let bit_width = 4;
//!
//! let quantizer = PolarQuant::new(dim, bit_width, DType::F32, &device).unwrap();
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
fn randomized_hadamard_matrix(dim: usize, dtype: DType, device: &Device) -> Result<Tensor> {
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

    Tensor::from_vec(matrix, (dim, dim), &Device::Cpu)?
        .to_dtype(dtype)?
        .to_device(device)
}

/// Result of PolarQuant quantization.
///
/// Stores centroid indices (b bits per coordinate) and original vector norms.
/// The indices select from Lloyd-Max centroids computed for the standard normal
/// distribution, scaled by 1/√d.
#[derive(Debug, Clone)]
pub struct PolarQuantized {
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
/// use candle_nn::polarquant::PolarQuant;
///
/// let device = Device::Cpu;
/// let quantizer = PolarQuant::new(128, 4, DType::F32, &device).unwrap();
///
/// let x = Tensor::randn(0f32, 1f32, (10, 128), &device).unwrap();
/// let quantized = quantizer.quantize(&x).unwrap();
/// let reconstructed = quantizer.dequantize(&quantized).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct PolarQuant {
    dim: usize,
    bit_width: usize,
    rotation: Tensor,
    centroids: Tensor,
    dtype: DType,
}

impl PolarQuant {
    /// Create a new PolarQuant quantizer.
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

    /// Create a new PolarQuant quantizer using fast Hadamard rotation.
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

        let centroids =
            Tensor::from_vec(scaled, (unit_centroids.len(),), device)?.to_dtype(dtype)?;
        let rotation = randomized_hadamard_matrix(dim, dtype, device)?;

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
    /// [`PolarQuantized`] with centroid indices and original norms.
    pub fn quantize(&self, x: &Tensor) -> Result<PolarQuantized> {
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

        Ok(PolarQuantized { indices, norms })
    }

    /// Dequantize back to vectors.
    ///
    /// # Arguments
    /// * `q` - Quantized representation from [`quantize`](Self::quantize)
    ///
    /// # Returns
    /// Reconstructed tensor, shape `[n, d]`.
    pub fn dequantize(&self, q: &PolarQuantized) -> Result<Tensor> {
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
}

/// Fused centroid-lookup and dot-product operation for PolarQuant-compressed weights.
///
/// Operates on **pre-rotated** input: given `x_rot = x @ Π^T`, computes:
///   `output[b][row] = norms[row] * Σ_i x_rot[b][i] * centroids[indices[row][i]]`
///
/// The rotation `x @ Π^T` is done outside this op using BLAS-accelerated tensor
/// matmul. This op handles only the cheap centroid-lookup + dot-product step,
/// which is O(batch × out × d) with no large matrix allocations.
#[derive(Debug, Clone)]
pub struct PolarQuantMatMul {
    /// Pre-computed `Π^T` tensor for BLAS-accelerated rotation. Shape: `[d, d]`.
    rotation_t: Tensor,
    /// Centroid values, length `2^b`.
    centroids: Vec<f32>,
    /// Weight indices, `[out_features * in_features]` as U8.
    indices: Vec<u8>,
    /// Weight norms, `[out_features]` as F32.
    norms: Vec<f32>,
    /// Number of output features.
    out_features: usize,
    /// Number of input features (= dim).
    in_features: usize,
}

impl PolarQuantMatMul {
    /// Create a fused matmul op from PolarQuant components.
    pub fn new(quantizer: &PolarQuant, indices: &Tensor, norms: &Tensor) -> Result<Self> {
        let (out_features, in_features) = indices.dims2()?;
        let indices_u8: Vec<u8> = indices.to_dtype(DType::U8)?.flatten_all()?.to_vec1()?;
        let norms_f32: Vec<f32> = norms.to_dtype(DType::F32)?.to_vec1()?;
        let centroids = quantizer.centroids_f32()?;
        let rotation_t = quantizer
            .rotation()
            .t()?
            .contiguous()?
            .to_dtype(DType::F32)?;

        Ok(Self {
            rotation_t,
            centroids,
            indices: indices_u8,
            norms: norms_f32,
            out_features,
            in_features,
        })
    }

    /// Rotation matrix transpose tensor, for BLAS-accelerated `x @ Π^T`.
    pub fn rotation_t(&self) -> &Tensor {
        &self.rotation_t
    }

    /// Number of output features.
    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

impl PolarQuantMatMul {
    /// Compute `x @ W^T` using BLAS rotation + BLAS centroid matmul.
    ///
    /// Two BLAS matmuls, no scalar loops:
    /// 1. `x_rot = x @ Π^T` — rotate input (shared `[d, d]` matmul)
    /// 2. Build `W_rot[row][i] = centroids[indices[row][i]] * norms[row]` (cheap lookups)
    /// 3. `output = x_rot @ W_rot^T` — standard BLAS matmul
    ///
    /// This avoids allocating the full `[out, in]` dequantized weight (which would
    /// require an additional `[out, d] @ [d, d]` rotation matmul). Instead, the
    /// rotated-space weight `W_rot` is just centroid lookups — O(out × d) with no
    /// matrix multiply.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dims = x.dims().to_vec();
        let d = self.in_features;
        let n = self.out_features;

        // Flatten to 2D
        let batch = x.elem_count() / d;
        let x_2d = if x.dims().len() == 1 {
            x.unsqueeze(0)?
        } else {
            x.reshape((batch, d))?
        };
        let x_2d = x_2d.contiguous()?;

        // Step 1: BLAS-accelerated rotation: x_rot = x @ Π^T  [batch, d]
        let x_rot = x_2d.matmul(&self.rotation_t)?;

        // Step 2: Build rotated-space weight matrix from centroid lookups
        // W_rot[row][i] = centroids[indices[row*d + i]] * norms[row]
        let mut w_rot_data = vec![0f32; n * d];
        for row in 0..n {
            let idx_start = row * d;
            let norm = self.norms[row];
            let w_row = &mut w_rot_data[row * d..(row + 1) * d];
            for (i, w) in w_row.iter_mut().enumerate() {
                *w = self.centroids[self.indices[idx_start + i] as usize] * norm;
            }
        }
        let w_rot = Tensor::from_vec(w_rot_data, (n, d), x.device())?;

        // Step 3: BLAS matmul: output = x_rot @ W_rot^T  [batch, n]
        let y_2d = x_rot.matmul(&w_rot.t()?)?;

        // Reshape back
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

        let quantizer = PolarQuant::new(dim, bit_width, DType::F32, &device)?;
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

        let quantizer = PolarQuant::new(dim, bit_width, DType::F32, &device)?;

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

        let quantizer = PolarQuant::new(dim, 2, DType::F32, &device)?;
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
            let quantizer = PolarQuant::new(dim, bit_width, DType::F32, &device)?;
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
            let quantizer = PolarQuant::new(dim, bit_width, DType::F32, &device)?;
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
