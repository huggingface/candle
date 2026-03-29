//! Seeded pseudo-random number generator for on-the-fly projection matrix generation.
//!
//! This module provides a lightweight, dependency-free PRNG based on the xorshift64 algorithm,
//! extended with a Box-Muller transform to produce standard-normal (N(0,1)) samples.
//!
//! ## Why a custom PRNG?
//!
//! The QJL and TurboQuant algorithms rely on a random projection matrix S ∈ ℝ^{d×d} where
//! S[i,j] ~ N(0,1). Storing this O(d²) matrix would negate most of the memory savings. Instead,
//! we regenerate rows of S on-the-fly during both encoding and decoding using the same seed,
//! ensuring deterministic reproducibility without storage overhead.
//!
//! The xorshift64 generator has period 2^64 − 1, which is more than sufficient for the dimensions
//! used in practice (d ≤ 512, requiring at most 512² = 262,144 random values per cache entry).

use std::f64::consts::PI;

/// A seeded xorshift64 pseudo-random number generator.
///
/// This generator produces a deterministic stream of pseudo-random 64-bit integers, which can
/// be used to derive f32 uniform samples and N(0,1) normal samples via Box-Muller transform.
///
/// The same seed always produces the same sequence, which is essential for regenerating the
/// random projection matrix S during attention score computation after encoding.
///
/// # Example
/// ```
/// use candle_nn::quant_kv::prng::Prng;
/// let mut rng = Prng::new(42);
/// let mut buf = vec![0.0f32; 64];
/// rng.fill_normal(&mut buf);
/// ```
#[derive(Debug, Clone)]
pub struct Prng {
    state: u64,
}

impl Prng {
    /// Create a new PRNG with the given seed. Seed must be non-zero; if zero is passed, 1 is
    /// used instead (xorshift64 has a degenerate fixed point at 0).
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Advance the state and return the next pseudo-random u64.
    ///
    /// Uses the xorshift64 algorithm with shifts (13, 7, 17), which passes all BigCrush tests
    /// and has period 2^64 − 1.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Return a uniform float in [0, 1) derived from the next u64.
    #[inline]
    pub fn next_f64_unit(&mut self) -> f64 {
        // Use the top 53 bits for full double precision
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Return a uniform f32 in [0, 1).
    #[inline]
    pub fn next_f32_unit(&mut self) -> f32 {
        // Use the top 24 bits for full single precision
        (self.next_u64() >> 40) as f32 * (1.0 / (1u32 << 24) as f32)
    }

    /// Return a pair of independent standard normal (N(0,1)) samples using the Box-Muller
    /// transform.
    ///
    /// Given two uniform samples u1, u2 ∈ (0, 1], the transform produces:
    ///   z0 = sqrt(-2 ln u1) · cos(2π u2)
    ///   z1 = sqrt(-2 ln u1) · sin(2π u2)
    ///
    /// Both z0 and z1 are independent N(0,1) variates.
    pub fn next_normal_pair(&mut self) -> (f32, f32) {
        // Ensure u1 > 0 to avoid log(0); clamp away from zero
        let u1 = (self.next_f64_unit() + 1e-10_f64).min(1.0);
        let u2 = self.next_f64_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        ((r * theta.cos()) as f32, (r * theta.sin()) as f32)
    }

    /// Fill the given buffer with independent N(0,1) samples.
    ///
    /// Uses Box-Muller in pairs; if `buf.len()` is odd, the last element is filled from the
    /// first of a new pair (the second is discarded).
    pub fn fill_normal(&mut self, buf: &mut [f32]) {
        let mut i = 0;
        while i + 1 < buf.len() {
            let (z0, z1) = self.next_normal_pair();
            buf[i] = z0;
            buf[i + 1] = z1;
            i += 2;
        }
        if i < buf.len() {
            let (z0, _) = self.next_normal_pair();
            buf[i] = z0;
        }
    }

    /// Fill the buffer with random ±1.0 values (Rademacher distribution), derived from
    /// the sign of N(0,1) samples.
    pub fn fill_signs(&mut self, buf: &mut [f32]) {
        let mut i = 0;
        while i + 1 < buf.len() {
            let (z0, z1) = self.next_normal_pair();
            buf[i] = if z0 >= 0.0 { 1.0 } else { -1.0 };
            buf[i + 1] = if z1 >= 0.0 { 1.0 } else { -1.0 };
            i += 2;
        }
        if i < buf.len() {
            let (z0, _) = self.next_normal_pair();
            buf[i] = if z0 >= 0.0 { 1.0 } else { -1.0 };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that `fill_normal` produces samples with mean ≈ 0 and variance ≈ 1.
    /// With 100,000 samples, the Central Limit Theorem guarantees this within tight tolerances.
    #[test]
    fn normal_statistics() {
        let mut rng = Prng::new(12345);
        let n = 100_000;
        let mut buf = vec![0.0f32; n];
        rng.fill_normal(&mut buf);

        let mean = buf.iter().sum::<f32>() / n as f32;
        let variance = buf.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;

        // With 100k samples, standard error of mean ≈ 1/sqrt(100k) ≈ 0.003
        assert!(
            mean.abs() < 0.05,
            "mean = {mean} (expected ≈ 0)"
        );
        // Variance of variance estimator is 2/(n-1) ≈ 0.00002, so std ≈ 0.004
        assert!(
            (variance - 1.0).abs() < 0.05,
            "variance = {variance} (expected ≈ 1)"
        );
    }

    /// Verify determinism: same seed, same sequence.
    #[test]
    fn determinism() {
        let mut rng1 = Prng::new(999);
        let mut rng2 = Prng::new(999);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    /// Verify seed 0 is handled safely (no fixed-point degenerate sequence).
    #[test]
    fn zero_seed_safe() {
        let mut rng = Prng::new(0);
        // Should not produce all-zero sequence
        let first = rng.next_u64();
        assert_ne!(first, 0);
    }

    /// Verify fill_signs produces roughly equal ±1 counts.
    #[test]
    fn signs_balanced() {
        let mut rng = Prng::new(42);
        let n = 10_000;
        let mut buf = vec![0.0f32; n];
        rng.fill_signs(&mut buf);
        let positive = buf.iter().filter(|&&x| x > 0.0).count();
        // Binomial(10000, 0.5): std ≈ 50. 3σ interval: [4850, 5150]
        assert!(
            positive >= 4800 && positive <= 5200,
            "signs balance: {positive}/{n} positive"
        );
    }
}
