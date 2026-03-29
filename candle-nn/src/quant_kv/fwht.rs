//! Fast Walsh-Hadamard Transform (FWHT) and Subsampled Randomized Hadamard Transform (SRHT).
//!
//! ## Mathematical Background
//!
//! The Walsh-Hadamard matrix H_n is defined recursively:
//!   H_1 = [1]
//!   H_{2n} = H_n ⊗ [1  1; 1 -1]
//!
//! Key properties:
//! - H_n^T · H_n = n · I  (orthogonality up to scaling)
//! - H_n is its own inverse: H_n · H_n = n · I, so H_n^{-1} = H_n / n
//! - Can be applied in O(n log n) via the butterfly algorithm
//!
//! ## Role in TurboQuant / PolarQuant
//!
//! Before quantizing key vectors, we apply the Subsampled Randomized Hadamard Transform (SRHT):
//!   Π = D · H / √n
//! where D is a diagonal matrix of random ±1 signs drawn from a seeded PRNG.
//!
//! This rotation has two important effects:
//! 1. **Concentration**: After rotation, each coordinate of a unit-sphere vector follows an
//!    approximately Beta(1/2, (d-1)/2) marginal distribution, which is the same for all input
//!    vectors. This "data-oblivious" property means we can pre-compute the optimal codebook
//!    without seeing any actual data.
//! 2. **Efficiency**: O(d log d) instead of O(d²) for a dense random rotation.
//!
//! The inverse rotation is D · H / √n (since D^{-1} = D for diagonal ±1 matrices, and
//! H^{-1} = H / n), giving:
//!   Π^{-1} = (H · D) / √n

use super::prng::Prng;

/// Returns the smallest power of 2 that is ≥ n.
///
/// # Examples
/// ```
/// use candle_nn::quant_kv::fwht::next_pow2;
/// assert_eq!(next_pow2(1), 1);
/// assert_eq!(next_pow2(3), 4);
/// assert_eq!(next_pow2(64), 64);
/// ```
pub fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}

/// Apply the Fast Walsh-Hadamard Transform in-place on a power-of-2 length slice.
///
/// After this call, `data` holds `H · data` (unnormalized). To get the normalized (orthonormal)
/// transform, divide each element by `sqrt(data.len())` afterward.
///
/// **Panics** if `data.len()` is not a power of 2 or is zero.
///
/// # Algorithm
///
/// The butterfly algorithm makes log2(n) passes. In each pass, adjacent pairs separated by
/// stride h are combined:
///   x[j]   ← x[j] + x[j+h]
///   x[j+h] ← x[j] - x[j+h]   (using the original x[j])
pub fn fwht_inplace(data: &mut [f32]) {
    let n = data.len();
    debug_assert!(n.is_power_of_two() && n > 0, "fwht_inplace requires power-of-2 length");

    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let x = data[j];
                let y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
            i += 2 * h;
        }
        h *= 2;
    }
}

/// Apply the Walsh-Hadamard Transform to an input slice of arbitrary length.
///
/// If `input.len()` is not a power of 2, the input is zero-padded to the next power of 2,
/// the transform is applied, and only the first `input.len()` elements are written to `output`.
///
/// The result is **normalized** (divided by `sqrt(padded_len)`), making it equivalent to an
/// approximate orthonormal rotation. This normalization ensures that norms are approximately
/// preserved: `||Hx||₂ ≈ ||x||₂`.
///
/// `output` must have the same length as `input`.
pub fn fwht_padded(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    let d = input.len();
    let padded = next_pow2(d);

    // Use a temporary buffer if padding is needed, or work in output directly
    if padded == d {
        output.copy_from_slice(input);
        fwht_inplace(output);
        let scale = 1.0 / (d as f32).sqrt();
        for x in output.iter_mut() {
            *x *= scale;
        }
    } else {
        let mut buf = vec![0.0f32; padded];
        buf[..d].copy_from_slice(input);
        fwht_inplace(&mut buf);
        let scale = 1.0 / (padded as f32).sqrt();
        for i in 0..d {
            output[i] = buf[i] * scale;
        }
    }
}

/// Apply the Subsampled Randomized Hadamard Transform (SRHT) to `input`, writing result to `out`.
///
/// The transform is: `out = (H · D · input) / √n`
/// where D is a diagonal matrix of random ±1 values drawn from `seed`.
///
/// After this rotation, a unit-sphere vector's coordinates follow an approximately
/// Beta(1/2, (d-1)/2) marginal distribution, which enables data-oblivious codebook design.
///
/// `input` and `out` must have the same length. Pads to next power of 2 internally if needed.
pub fn srht_precondition(input: &[f32], seed: u64, out: &mut [f32]) {
    assert_eq!(input.len(), out.len());
    let d = input.len();
    let padded = next_pow2(d);

    // Step 1: Apply random sign flip D (Rademacher diagonal matrix)
    let mut rng = Prng::new(seed);
    let mut signs = vec![0.0f32; d];
    rng.fill_signs(&mut signs);

    if padded == d {
        // Copy with sign flip directly into out
        for i in 0..d {
            out[i] = input[i] * signs[i];
        }
        // Step 2: Apply FWHT
        fwht_inplace(out);
        // Step 3: Normalize
        let scale = 1.0 / (d as f32).sqrt();
        for x in out.iter_mut() {
            *x *= scale;
        }
    } else {
        let mut buf = vec![0.0f32; padded];
        for i in 0..d {
            buf[i] = input[i] * signs[i];
        }
        // buf[d..] stays zero (zero-padding)
        fwht_inplace(&mut buf);
        let scale = 1.0 / (padded as f32).sqrt();
        for i in 0..d {
            out[i] = buf[i] * scale;
        }
    }
}

/// Apply the inverse SRHT: `out = (D · H · input) · √n`
///
/// Since D^{-1} = D (±1 diagonal is self-inverse) and H^{-1} = H/n:
///   Π^{-1} = D · H / √n = D · (H · input / n) · √n = D · H · input / √n
///
/// In practice: apply H, normalize, then re-apply the same sign flip D.
/// Note: we apply D AFTER H here, giving (D·H·input)/√n which equals Π^T = Π^{-1}
/// for orthogonal Π = H·D/√n, since (H·D)^T = D·H and D^{-1} = D.
pub fn srht_inverse(input: &[f32], seed: u64, out: &mut [f32]) {
    assert_eq!(input.len(), out.len());
    let d = input.len();
    let padded = next_pow2(d);

    // Step 1: Apply FWHT (H is symmetric: H^T = H)
    if padded == d {
        out.copy_from_slice(input);
        fwht_inplace(out);
        let scale = 1.0 / (d as f32).sqrt();
        for x in out.iter_mut() {
            *x *= scale;
        }
    } else {
        let mut buf = vec![0.0f32; padded];
        buf[..d].copy_from_slice(input);
        fwht_inplace(&mut buf);
        let scale = 1.0 / (padded as f32).sqrt();
        for i in 0..d {
            out[i] = buf[i] * scale;
        }
    }

    // Step 2: Re-apply the same sign flip D (D^{-1} = D)
    let mut rng = Prng::new(seed);
    let mut signs = vec![0.0f32; d];
    rng.fill_signs(&mut signs);
    for i in 0..d {
        out[i] *= signs[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that FWHT(FWHT(x)) = n * x for a power-of-2 length vector.
    #[test]
    fn round_trip() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let n = input.len() as f32;
        let mut data = input.clone();
        fwht_inplace(&mut data);
        fwht_inplace(&mut data);
        for (a, b) in data.iter().zip(input.iter()) {
            assert!(
                (a - b * n).abs() < 1e-4,
                "round_trip: got {a}, expected {}",
                b * n
            );
        }
    }

    /// Verify that SRHT followed by SRHT inverse recovers the original vector.
    #[test]
    fn srht_round_trip() {
        let d = 64;
        let seed = 42u64;
        let input: Vec<f32> = (0..d).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut rotated = vec![0.0f32; d];
        let mut recovered = vec![0.0f32; d];

        srht_precondition(&input, seed, &mut rotated);
        srht_inverse(&rotated, seed, &mut recovered);

        for (orig, rec) in input.iter().zip(recovered.iter()) {
            assert!(
                (orig - rec).abs() < 1e-4,
                "srht_round_trip: orig={orig}, rec={rec}"
            );
        }
    }

    /// Verify that SRHT approximately preserves norms (orthogonality property).
    ///
    /// For a random unit vector x, ||SRHT(x)||₂ ≈ 1.0 (within numerical precision).
    #[test]
    fn srht_norm_preservation() {
        let d = 128;
        let seed = 77u64;

        // Create a random unit vector
        let mut rng = Prng::new(seed.wrapping_add(1));
        let mut input = vec![0.0f32; d];
        rng.fill_normal(&mut input);
        let norm_sq: f32 = input.iter().map(|x| x * x).sum();
        let norm = norm_sq.sqrt();
        for x in input.iter_mut() {
            *x /= norm;
        }

        let mut rotated = vec![0.0f32; d];
        srht_precondition(&input, seed, &mut rotated);
        let rotated_norm_sq: f32 = rotated.iter().map(|x| x * x).sum();

        // Norm should be preserved: ||SRHT(x)||² ≈ 1.0
        assert!(
            (rotated_norm_sq - 1.0).abs() < 0.1,
            "norm not preserved: ||SRHT(x)||² = {rotated_norm_sq}"
        );
    }

    /// Verify that fwht_padded works for non-power-of-2 input (d=6, padded to 8).
    #[test]
    fn non_pow2_input() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // d=6, pads to 8
        let mut output = vec![0.0f32; 6];
        fwht_padded(&input, &mut output);

        // Energy should be approximately preserved
        let input_energy: f32 = input.iter().map(|x| x * x).sum();
        let output_energy: f32 = output.iter().map(|x| x * x).sum();
        // Due to zero-padding, output energy ≤ input_energy (some energy in the truncated dims)
        assert!(
            output_energy <= input_energy * 1.01,
            "energy not bounded: output={output_energy}, input={input_energy}"
        );
    }

    /// Verify that SRHT with different seeds produces different outputs.
    #[test]
    fn different_seeds_differ() {
        let input = vec![1.0f32; 32];
        let mut out1 = vec![0.0f32; 32];
        let mut out2 = vec![0.0f32; 32];
        srht_precondition(&input, 1, &mut out1);
        srht_precondition(&input, 2, &mut out2);
        let diff: f32 = out1.iter().zip(out2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.0, "different seeds should produce different rotations");
    }
}
