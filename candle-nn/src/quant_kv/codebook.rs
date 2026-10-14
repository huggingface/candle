//! Lloyd-Max scalar quantization codebooks for the Beta distribution.
//!
//! ## Background
//!
//! After applying the Subsampled Randomized Hadamard Transform (SRHT) to a unit-sphere vector,
//! each coordinate follows an approximately Beta marginal distribution. Specifically:
//!
//! - For a d-dimensional unit-sphere vector after rotation, each coordinate follows
//!   Beta(1/2, (d-1)/2) remapped to [-1, 1].
//! - For PolarQuant level-1 angles (atan2 of 2D subvectors), after SRHT the angle is
//!   approximately **uniform** on [-π, π] (since each 2D subvector direction is uniform).
//! - For PolarQuant level-ℓ angles (ℓ ≥ 2), the distribution is:
//!   f(ψ) ∝ sin^(2^(ℓ-1) - 1)(2ψ)  on [0, π/2]
//!
//! The Lloyd-Max quantizer is the optimal scalar quantizer that minimizes Mean Squared Error
//! (MSE) given a known source distribution. It alternates between:
//!   1. Computing new boundaries as midpoints between adjacent centroids
//!   2. Computing new centroids as the mean of each cell under the source distribution
//!
//! ## Data-Oblivious Property
//!
//! Because the angle distribution is fully determined by the SRHT preconditioning (it depends
//! only on d and the level ℓ, not on the actual data), the codebook can be computed once and
//! reused for all vectors. This is the key advantage of PolarQuant/TurboQuant over calibrated
//! methods like KIVI that require per-model profiling.

use std::f64::consts::PI;

// ============================================================================
// Math utilities for Lloyd-Max Gaussian codebook computation
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

/// Compute optimal Lloyd-Max quantizer centroids for N(0, 1).
///
/// Uses the iterative Lloyd-Max algorithm to find `2^bit_width` centroids that
/// minimize expected squared quantization error for the standard normal distribution.
/// Returns centroids in ascending order.
pub fn lloyd_max_gaussian(bit_width: usize, max_iter: usize) -> Vec<f64> {
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

/// A scalar quantization codebook with `2^bits` levels.
///
/// - `boundaries`: the `2^bits - 1` decision boundaries (thresholds) between cells, sorted
///   in ascending order. A value x is assigned to cell k if `boundaries[k-1] ≤ x < boundaries[k]`.
/// - `centroids`: the `2^bits` reconstruction values, one per cell.
#[derive(Debug, Clone, PartialEq)]
pub struct Codebook {
    pub boundaries: Vec<f32>,
    pub centroids: Vec<f32>,
}

impl Codebook {
    /// The number of quantization levels (2^bits).
    pub fn num_levels(&self) -> usize {
        self.centroids.len()
    }

    /// The number of bits per quantized value (log2(num_levels)).
    pub fn bits(&self) -> u32 {
        self.num_levels().trailing_zeros()
    }

    /// Quantize a scalar value to the index of its nearest cell (0-indexed).
    ///
    /// Uses binary search on the boundaries array. O(log(2^bits)) = O(bits) per call.
    pub fn quantize(&self, x: f32) -> u8 {
        if self.boundaries.is_empty() {
            return 0;
        }
        // Binary search: find first boundary that x is less than
        let idx = self.boundaries.partition_point(|&b| x >= b);
        idx as u8
    }

    /// Reconstruct a scalar value from a quantization index.
    pub fn dequantize(&self, idx: u8) -> f32 {
        let idx = idx as usize;
        if idx < self.centroids.len() {
            self.centroids[idx]
        } else {
            *self.centroids.last().unwrap_or(&0.0)
        }
    }

    /// Create a uniform quantization codebook covering [lo, hi] with `2^bits` equal-width cells.
    ///
    /// This is optimal for a uniform source distribution, which applies to PolarQuant level-1
    /// angles (approximately uniform on [-π, π] after SRHT preconditioning).
    pub fn uniform(bits: u32, lo: f32, hi: f32) -> Self {
        let n = 1usize << bits;
        let width = (hi - lo) as f64 / n as f64;
        let boundaries: Vec<f32> = (1..n)
            .map(|i| (lo as f64 + i as f64 * width) as f32)
            .collect();
        let centroids: Vec<f32> = (0..n)
            .map(|i| (lo as f64 + (i as f64 + 0.5) * width) as f32)
            .collect();
        Self {
            boundaries,
            centroids,
        }
    }
}

/// The probability density function for PolarQuant level-ℓ angles (ℓ ≥ 2).
///
/// For level ℓ, the density is f(ψ) = C · sin^(2^(ℓ-1) - 1)(2ψ) on [0, π/2],
/// where C is the normalizing constant.
///
/// Returns f(ψ) (unnormalized; the Lloyd-Max algorithm only needs relative densities).
fn polar_angle_pdf(psi: f64, level: u32) -> f64 {
    // Exponent = 2^(level-1) - 1
    let exp = (1usize << (level - 1)) as f64 - 1.0;
    let sin_val = (2.0 * psi).sin();
    if sin_val <= 0.0 {
        return 0.0;
    }
    sin_val.powf(exp)
}

/// Compute the normalizing constant for `polar_angle_pdf` on [0, π/2] using numerical
/// integration (composite Simpson's rule with 1024 intervals).
fn polar_angle_normalizer(level: u32) -> f64 {
    let n = 1024usize;
    let a = 0.0f64;
    let b = PI / 2.0;
    let h = (b - a) / n as f64;
    let mut sum = polar_angle_pdf(a, level) + polar_angle_pdf(b, level);
    for i in 1..n {
        let x = a + i as f64 * h;
        let weight = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum += weight * polar_angle_pdf(x, level);
    }
    sum * h / 3.0
}

/// Compute the expected value (centroid) of ψ under `polar_angle_pdf` over [lo, hi].
///
/// Returns ∫_{lo}^{hi} ψ · f(ψ) dψ / ∫_{lo}^{hi} f(ψ) dψ
fn polar_angle_centroid(lo: f64, hi: f64, level: u32) -> f64 {
    let n = 256usize;
    let h = (hi - lo) / n as f64;

    // Numerator: ∫ ψ · f(ψ) dψ
    let mut num = (lo * polar_angle_pdf(lo, level) + hi * polar_angle_pdf(hi, level)) / 2.0;
    // Denominator: ∫ f(ψ) dψ
    let mut den = (polar_angle_pdf(lo, level) + polar_angle_pdf(hi, level)) / 2.0;

    for i in 1..n {
        let x = lo + i as f64 * h;
        let f = polar_angle_pdf(x, level);
        num += x * f;
        den += f;
    }

    if den == 0.0 {
        (lo + hi) / 2.0
    } else {
        num / den
    }
}

/// Compute the CDF of `polar_angle_pdf` at x, normalized so that CDF(π/2) = 1.
fn polar_angle_cdf(x: f64, level: u32, normalizer: f64) -> f64 {
    let n = 512usize;
    let h = x / n as f64;
    let a = 0.0f64;
    let mut sum = polar_angle_pdf(a, level) + polar_angle_pdf(x, level);
    for i in 1..n {
        let t = a + i as f64 * h;
        let weight = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum += weight * polar_angle_pdf(t, level);
    }
    let integral = sum * h / 3.0;
    if normalizer == 0.0 {
        0.5
    } else {
        (integral / normalizer).clamp(0.0, 1.0)
    }
}

/// Compute a Lloyd-Max codebook for PolarQuant level-ℓ (ℓ ≥ 2) angles on [0, π/2].
///
/// The algorithm:
/// 1. Initialize boundaries by equal-probability quantiles of the CDF
/// 2. Iterate: compute centroids as means of each cell, then update boundaries as midpoints
/// 3. Repeat until convergence (50 iterations is sufficient for machine precision)
///
/// # Arguments
/// * `bits` — number of bits per quantized angle (typically 2 for levels ≥ 2)
/// * `level` — the PolarQuant level (1-indexed; this function is for levels ≥ 2)
pub fn compute_polar_codebook(bits: u32, level: u32) -> Codebook {
    assert!(
        level >= 2,
        "use Codebook::uniform for level 1 (uniform distribution)"
    );
    assert!(bits <= 8, "bits must be ≤ 8");

    let n = 1usize << bits; // number of levels
    let normalizer = polar_angle_normalizer(level);

    // Initialize boundaries by equal-probability quantiles
    // Invert CDF using binary search
    let mut boundaries: Vec<f64> = (1..n)
        .map(|k| {
            let target = k as f64 / n as f64;
            // Binary search on [0, π/2] for CDF^{-1}(target)
            let mut lo = 0.0f64;
            let mut hi = PI / 2.0;
            for _ in 0..64 {
                let mid = (lo + hi) / 2.0;
                if polar_angle_cdf(mid, level, normalizer) < target {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            (lo + hi) / 2.0
        })
        .collect();

    // Lloyd-Max iteration: alternate centroid computation and boundary update
    for _ in 0..60 {
        // Compute centroids as conditional means
        let mut centroids: Vec<f64> = Vec::with_capacity(n);
        let full_boundaries: Vec<f64> = std::iter::once(0.0f64)
            .chain(boundaries.iter().copied())
            .chain(std::iter::once(PI / 2.0))
            .collect();

        for k in 0..n {
            let lo = full_boundaries[k];
            let hi = full_boundaries[k + 1];
            centroids.push(polar_angle_centroid(lo, hi, level));
        }

        // Update boundaries as midpoints between adjacent centroids
        for k in 0..n - 1 {
            boundaries[k] = (centroids[k] + centroids[k + 1]) / 2.0;
        }
    }

    // Final centroid computation
    let full_boundaries: Vec<f64> = std::iter::once(0.0f64)
        .chain(boundaries.iter().copied())
        .chain(std::iter::once(PI / 2.0))
        .collect();
    let centroids: Vec<f32> = (0..n)
        .map(|k| polar_angle_centroid(full_boundaries[k], full_boundaries[k + 1], level) as f32)
        .collect();

    Codebook {
        boundaries: boundaries.iter().map(|&b| b as f32).collect(),
        centroids,
    }
}

/// Lazy-computed codebook cache for PolarQuant levels.
///
/// Returns a `Codebook` for the given `(bits, level)` combination.
/// - Level 1: uniform distribution on [-π, π] → use `Codebook::uniform(bits, -PI, PI)`
/// - Level ≥ 2: Beta-derived distribution → use `compute_polar_codebook(bits, level)`
///
/// This function recomputes every time it is called. For production use, store the result in
/// your `PolarQuantConfig` struct so it is computed once at construction time.
pub fn get_polar_codebook(bits: u32, level: u32) -> Codebook {
    if level == 1 {
        Codebook::uniform(bits, -PI as f32, PI as f32)
    } else {
        compute_polar_codebook(bits, level)
    }
}

/// Compute an optimal Lloyd-Max codebook for the marginal distribution of a unit-sphere
/// coordinate after random rotation (SRHT preconditioning).
///
/// After SRHT, each coordinate of a unit-sphere vector follows approximately N(0, 1/d).
/// This function computes the optimal Lloyd-Max codebook for N(0,1) and scales all
/// centroids and boundaries by 1/√d, giving lower MSE than a uniform approximation.
///
/// This is used by TurboQuant Stage 1 (PolarQuant MSE component).
pub fn get_sphere_marginal_codebook(bits: u32, dim: usize) -> Codebook {
    let unit_centroids = lloyd_max_gaussian(bits as usize, 1000);
    let scale = 1.0 / (dim as f64).sqrt();
    let scaled: Vec<f64> = unit_centroids.iter().map(|&c| c * scale).collect();
    let boundaries: Vec<f32> = scaled
        .windows(2)
        .map(|w| ((w[0] + w[1]) / 2.0) as f32)
        .collect();
    let centroids: Vec<f32> = scaled.iter().map(|&c| c as f32).collect();
    Codebook {
        boundaries,
        centroids,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that codebook boundaries are strictly monotone increasing.
    #[test]
    fn monotone_boundaries() {
        // Test uniform codebook
        let cb = Codebook::uniform(4, -1.0, 1.0);
        for w in cb.boundaries.windows(2) {
            assert!(w[0] < w[1], "boundaries not monotone: {} >= {}", w[0], w[1]);
        }

        // Test polar codebook level 2
        let cb = compute_polar_codebook(2, 2);
        for w in cb.boundaries.windows(2) {
            assert!(w[0] < w[1], "boundaries not monotone: {} >= {}", w[0], w[1]);
        }
    }

    /// Verify that centroids are inside their respective cells.
    #[test]
    fn centroids_in_cells() {
        let cb = compute_polar_codebook(2, 2);
        let full_lo: Vec<f64> = std::iter::once(0.0f64)
            .chain(cb.boundaries.iter().map(|&b| b as f64))
            .collect();
        let full_hi: Vec<f64> = cb
            .boundaries
            .iter()
            .map(|&b| b as f64)
            .chain(std::iter::once(PI / 2.0))
            .collect();

        for (k, (&c, (lo, hi))) in cb
            .centroids
            .iter()
            .zip(full_lo.iter().zip(full_hi.iter()))
            .enumerate()
        {
            assert!(
                c as f64 >= *lo && c as f64 <= *hi,
                "centroid[{k}]={c} not in [{lo}, {hi}]"
            );
        }
    }

    /// Verify that quantize then dequantize gives a value close to the original.
    #[test]
    fn round_trip_error() {
        let cb = Codebook::uniform(4, -std::f32::consts::PI, std::f32::consts::PI);
        let n = 1000;
        let mut total_mse = 0.0f64;
        for i in 0..n {
            let x = -std::f32::consts::PI + 2.0 * std::f32::consts::PI * i as f32 / n as f32;
            let idx = cb.quantize(x);
            let x_hat = cb.dequantize(idx);
            total_mse += (x - x_hat) as f64 * (x - x_hat) as f64;
        }
        let mse = total_mse / n as f64;
        // For uniform quantizer on [-π, π] with 16 levels: MSE = (2π/16)² / 12 ≈ 0.102
        let cell_width = 2.0 * PI / 16.0;
        let theoretical_mse = cell_width * cell_width / 12.0;
        assert!(
            mse <= theoretical_mse * 1.5,
            "MSE={mse} > 1.5 × theoretical={theoretical_mse}"
        );
    }

    /// Verify correct number of levels for each bit-width.
    #[test]
    fn level_count() {
        for bits in 1u32..=4 {
            let cb = Codebook::uniform(bits, -1.0, 1.0);
            assert_eq!(cb.num_levels(), 1 << bits);
            assert_eq!(cb.boundaries.len(), (1 << bits) - 1);
            assert_eq!(cb.bits(), bits);
        }
    }

    /// For 1-bit Lloyd-Max on N(0,1), optimal centroids are ±√(2/π) ≈ ±0.7979.
    #[test]
    fn lloyd_max_gaussian_1bit_known_value() {
        let centroids = lloyd_max_gaussian(1, 1000);
        assert_eq!(centroids.len(), 2);
        let expected = (2.0 / std::f64::consts::PI).sqrt();
        assert!(
            (centroids[0] + expected).abs() < 1e-4,
            "c0={}, expected ≈ {}",
            centroids[0],
            -expected
        );
        assert!(
            (centroids[1] - expected).abs() < 1e-4,
            "c1={}, expected ≈ {}",
            centroids[1],
            expected
        );
    }

    /// For 2-bit Lloyd-Max on N(0,1), optimal centroids are ≈ ±0.4528, ±1.5104.
    #[test]
    fn lloyd_max_gaussian_2bit_known_value() {
        let centroids = lloyd_max_gaussian(2, 1000);
        assert_eq!(centroids.len(), 4);
        assert!((centroids[0] + 1.51).abs() < 0.01, "c0={}", centroids[0]);
        assert!((centroids[1] + 0.4528).abs() < 0.01, "c1={}", centroids[1]);
        assert!((centroids[2] - 0.4528).abs() < 0.01, "c2={}", centroids[2]);
        assert!((centroids[3] - 1.51).abs() < 0.01, "c3={}", centroids[3]);
    }

    /// Lloyd-Max centroids for N(0,1) should be symmetric around 0.
    #[test]
    fn lloyd_max_gaussian_symmetry() {
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

    /// MSE should strictly decrease as bit-width increases for sphere marginal codebook.
    #[test]
    fn sphere_marginal_distortion_decreases_with_bits() {
        let dim = 128usize;
        let std = 1.0 / (dim as f32).sqrt();
        // Generate deterministic test points from N(0, 1/d) via simple linear congruential samples
        let n = 256usize;
        let mut prev_mse = f64::MAX;
        for bits in 1u32..=4 {
            let cb = get_sphere_marginal_codebook(bits, dim);
            let mut mse = 0.0f64;
            for i in 0..n {
                // Evenly-spaced samples over [-4σ, 4σ] — exercises the full codebook range
                let x = -4.0 * std + 8.0 * std * i as f32 / n as f32;
                let idx = cb.quantize(x);
                let x_hat = cb.dequantize(idx);
                let err = (x - x_hat) as f64;
                mse += err * err;
            }
            mse /= n as f64;
            assert!(
                mse < prev_mse,
                "{}-bit MSE={mse:.6} should be < {}-bit MSE={prev_mse:.6}",
                bits,
                bits - 1
            );
            prev_mse = mse;
        }
    }

    /// All quantize() outputs must be valid indices for the codebook.
    #[test]
    fn sphere_marginal_index_range() {
        let dim = 64usize;
        let std = 1.0 / (dim as f32).sqrt();
        for bits in 1u32..=4 {
            let cb = get_sphere_marginal_codebook(bits, dim);
            let max_valid = (1u8 << bits) - 1;
            for i in 0..200 {
                let x = -5.0 * std + 10.0 * std * i as f32 / 200.0;
                let idx = cb.quantize(x);
                assert!(
                    idx <= max_valid,
                    "{}-bit index {} out of range [0, {}]",
                    bits,
                    idx,
                    max_valid
                );
            }
        }
    }
}
