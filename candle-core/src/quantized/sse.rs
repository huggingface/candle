//! SSSE3/SSE4.1 fallback for the quantized matmul kernels.
//!
//! Only `vec_dot_q6k_q8k` is implemented here. This is a straight port of
//! `super::avx::vec_dot_q6k_q8k`, splitting every 256-bit (`__m256i`/`__m256`)
//! operation into a pair of 128-bit (`__m128i`/`__m128`) operations acting on
//! the low and high halves of the original register. The element-to-scale
//! pairing is preserved exactly (see the `get_scale_shuffle` table, which
//! replicates each of the 16 scale bytes 8 times - the low 8 words produced by
//! `maddubs_epi16` on a 16-byte half line up with the first scale byte of that
//! half, and the high 8 words line up with the second).
//!
//! This path exists for x86_64 CPUs that have SSSE3 + SSE4.1 but lack AVX2
//! (e.g. pre-Sandy-Bridge-refresh / pre-Haswell-era Xeons), where candle-core
//! upstream has no vectorized fallback and silently drops to a pure-scalar
//! generic implementation. It intentionally does not use FMA (`_mm_fmadd_ps`)
//! since SSE4.1-only CPUs typically lack it too; a plain multiply + add is
//! used instead.
//!
//! `vec_dot_q4k_q8k` is ported the same way, since the tutelary-ai Mid/Low
//! tiers' production GGUF files use Q4_K_M (see `tutelary-ai`'s
//! `docs/research/base-models.md`), not Q6K.
//!
//! No other quant formats (q2k/q3k/q5k/q4_0/q8_0/...) are ported: see
//! `docs/ops/test-server-runbook.md` in the `tutelary-ai` repo for the
//! empirical analysis (GGUF header tensor-type scan) showing that Q6K/Q4K are
//! the only quantized weight formats present in the models this fork targets.

use byteorder::{ByteOrder, LittleEndian};

use super::k_quants::{BlockQ4K, BlockQ6K, BlockQ8K, QK_K};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[inline(always)]
unsafe fn get_scale_shuffle(i: usize) -> __m128i {
    const K_SHUFFLE: [u8; 128] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
        3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7,
        7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10,
        11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13,
        13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15,
    ];
    _mm_loadu_si128((K_SHUFFLE.as_ptr() as *const __m128i).add(i))
}

#[inline(always)]
unsafe fn hsum_float_4(x: __m128) -> f32 {
    let res = _mm_add_ps(x, _mm_movehl_ps(x, x));
    let res = _mm_add_ss(res, _mm_movehdup_ps(res));
    _mm_cvtss_f32(res)
}

/// Widens each byte of `v` into the low 8 or high 8 words of a 128-bit
/// register, matching how `_mm256_cvtepi8_epi16` widens a full 16-byte
/// `__m128i` into a 256-bit 16-word register (low 8 bytes -> low 8 words,
/// high 8 bytes -> high 8 words).
#[inline(always)]
unsafe fn cvtepi8_epi16_lo(v: __m128i) -> __m128i {
    _mm_cvtepi8_epi16(v)
}

#[inline(always)]
unsafe fn cvtepi8_epi16_hi(v: __m128i) -> __m128i {
    _mm_cvtepi8_epi16(_mm_srli_si128(v, 8))
}

pub(crate) fn vec_dot_q6k_q8k(n: usize, xs: &[BlockQ6K], ys: &[BlockQ8K]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_q6k_q8k (sse): {n} is not divisible by {QK_K}"
    );

    unsafe {
        let m4 = _mm_set1_epi8(0xF);
        let m2 = _mm_set1_epi8(3);
        let m32s = _mm_set1_epi8(32);
        let mut acc_lo = _mm_setzero_ps();
        let mut acc_hi = _mm_setzero_ps();

        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = y.d * x.d.to_f32();
            let d_bcast = _mm_set1_ps(d);
            let mut q4 = x.ql.as_ptr();
            let mut qh = x.qh.as_ptr();
            let mut q8 = y.qs.as_ptr();

            let scales = _mm_loadu_si128(x.scales.as_ptr() as *const __m128i);
            let mut sumi_lo = _mm_setzero_si128();
            let mut sumi_hi = _mm_setzero_si128();

            for j in 0..QK_K / 128 {
                let is = j * 4;
                let scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle(is));
                let scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
                let scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
                let scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));

                // Each "256-bit" load below is replaced by a pair of 128-bit
                // loads (lo = first 16 bytes, hi = next 16 bytes).
                let q4bits1_lo = _mm_loadu_si128(q4 as *const __m128i);
                let q4bits1_hi = _mm_loadu_si128(q4.add(16) as *const __m128i);
                q4 = q4.add(32);
                let q4bits2_lo = _mm_loadu_si128(q4 as *const __m128i);
                let q4bits2_hi = _mm_loadu_si128(q4.add(16) as *const __m128i);
                q4 = q4.add(32);
                let q4bits_h_lo = _mm_loadu_si128(qh as *const __m128i);
                let q4bits_h_hi = _mm_loadu_si128(qh.add(16) as *const __m128i);
                qh = qh.add(32);

                let (q4h_0_lo, q4h_0_hi) = (
                    _mm_slli_epi16(_mm_and_si128(q4bits_h_lo, m2), 4),
                    _mm_slli_epi16(_mm_and_si128(q4bits_h_hi, m2), 4),
                );
                let (q4h_1_lo, q4h_1_hi) = (
                    _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bits_h_lo, 2), m2), 4),
                    _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bits_h_hi, 2), m2), 4),
                );
                let (q4h_2_lo, q4h_2_hi) = (
                    _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bits_h_lo, 4), m2), 4),
                    _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bits_h_hi, 4), m2), 4),
                );
                let (q4h_3_lo, q4h_3_hi) = (
                    _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bits_h_lo, 6), m2), 4),
                    _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bits_h_hi, 6), m2), 4),
                );

                let (q4_0_lo, q4_0_hi) = (
                    _mm_or_si128(_mm_and_si128(q4bits1_lo, m4), q4h_0_lo),
                    _mm_or_si128(_mm_and_si128(q4bits1_hi, m4), q4h_0_hi),
                );
                let (q4_1_lo, q4_1_hi) = (
                    _mm_or_si128(_mm_and_si128(q4bits2_lo, m4), q4h_1_lo),
                    _mm_or_si128(_mm_and_si128(q4bits2_hi, m4), q4h_1_hi),
                );
                let (q4_2_lo, q4_2_hi) = (
                    _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits1_lo, 4), m4), q4h_2_lo),
                    _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits1_hi, 4), m4), q4h_2_hi),
                );
                let (q4_3_lo, q4_3_hi) = (
                    _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits2_lo, 4), m4), q4h_3_lo),
                    _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits2_hi, 4), m4), q4h_3_hi),
                );

                let (q8_0_lo, q8_0_hi) = (
                    _mm_loadu_si128(q8 as *const __m128i),
                    _mm_loadu_si128(q8.add(16) as *const __m128i),
                );
                q8 = q8.add(32);
                let (q8_1_lo, q8_1_hi) = (
                    _mm_loadu_si128(q8 as *const __m128i),
                    _mm_loadu_si128(q8.add(16) as *const __m128i),
                );
                q8 = q8.add(32);
                let (q8_2_lo, q8_2_hi) = (
                    _mm_loadu_si128(q8 as *const __m128i),
                    _mm_loadu_si128(q8.add(16) as *const __m128i),
                );
                q8 = q8.add(32);
                let (q8_3_lo, q8_3_hi) = (
                    _mm_loadu_si128(q8 as *const __m128i),
                    _mm_loadu_si128(q8.add(16) as *const __m128i),
                );
                q8 = q8.add(32);

                let q8s_0_lo = _mm_maddubs_epi16(m32s, q8_0_lo);
                let q8s_0_hi = _mm_maddubs_epi16(m32s, q8_0_hi);
                let q8s_1_lo = _mm_maddubs_epi16(m32s, q8_1_lo);
                let q8s_1_hi = _mm_maddubs_epi16(m32s, q8_1_hi);
                let q8s_2_lo = _mm_maddubs_epi16(m32s, q8_2_lo);
                let q8s_2_hi = _mm_maddubs_epi16(m32s, q8_2_hi);
                let q8s_3_lo = _mm_maddubs_epi16(m32s, q8_3_lo);
                let q8s_3_hi = _mm_maddubs_epi16(m32s, q8_3_hi);

                let p16_0_lo = _mm_maddubs_epi16(q4_0_lo, q8_0_lo);
                let p16_0_hi = _mm_maddubs_epi16(q4_0_hi, q8_0_hi);
                let p16_1_lo = _mm_maddubs_epi16(q4_1_lo, q8_1_lo);
                let p16_1_hi = _mm_maddubs_epi16(q4_1_hi, q8_1_hi);
                let p16_2_lo = _mm_maddubs_epi16(q4_2_lo, q8_2_lo);
                let p16_2_hi = _mm_maddubs_epi16(q4_2_hi, q8_2_hi);
                let p16_3_lo = _mm_maddubs_epi16(q4_3_lo, q8_3_lo);
                let p16_3_hi = _mm_maddubs_epi16(q4_3_hi, q8_3_hi);

                let p16_0_lo = _mm_sub_epi16(p16_0_lo, q8s_0_lo);
                let p16_0_hi = _mm_sub_epi16(p16_0_hi, q8s_0_hi);
                let p16_1_lo = _mm_sub_epi16(p16_1_lo, q8s_1_lo);
                let p16_1_hi = _mm_sub_epi16(p16_1_hi, q8s_1_hi);
                let p16_2_lo = _mm_sub_epi16(p16_2_lo, q8s_2_lo);
                let p16_2_hi = _mm_sub_epi16(p16_2_hi, q8s_2_hi);
                let p16_3_lo = _mm_sub_epi16(p16_3_lo, q8s_3_lo);
                let p16_3_hi = _mm_sub_epi16(p16_3_hi, q8s_3_hi);

                let p16_0_lo = _mm_madd_epi16(cvtepi8_epi16_lo(scale_0), p16_0_lo);
                let p16_0_hi = _mm_madd_epi16(cvtepi8_epi16_hi(scale_0), p16_0_hi);
                let p16_1_lo = _mm_madd_epi16(cvtepi8_epi16_lo(scale_1), p16_1_lo);
                let p16_1_hi = _mm_madd_epi16(cvtepi8_epi16_hi(scale_1), p16_1_hi);
                let p16_2_lo = _mm_madd_epi16(cvtepi8_epi16_lo(scale_2), p16_2_lo);
                let p16_2_hi = _mm_madd_epi16(cvtepi8_epi16_hi(scale_2), p16_2_hi);
                let p16_3_lo = _mm_madd_epi16(cvtepi8_epi16_lo(scale_3), p16_3_lo);
                let p16_3_hi = _mm_madd_epi16(cvtepi8_epi16_hi(scale_3), p16_3_hi);

                sumi_lo = _mm_add_epi32(sumi_lo, _mm_add_epi32(p16_0_lo, p16_1_lo));
                sumi_hi = _mm_add_epi32(sumi_hi, _mm_add_epi32(p16_0_hi, p16_1_hi));
                sumi_lo = _mm_add_epi32(sumi_lo, _mm_add_epi32(p16_2_lo, p16_3_lo));
                sumi_hi = _mm_add_epi32(sumi_hi, _mm_add_epi32(p16_2_hi, p16_3_hi));
            }

            acc_lo = _mm_add_ps(acc_lo, _mm_mul_ps(d_bcast, _mm_cvtepi32_ps(sumi_lo)));
            acc_hi = _mm_add_ps(acc_hi, _mm_mul_ps(d_bcast, _mm_cvtepi32_ps(sumi_hi)));
        }
        hsum_float_4(_mm_add_ps(acc_lo, acc_hi))
    }
}

/// 128-bit analogue of `avx::get_scale_shuffle_k4`: each row `i` (0..8)
/// packs the byte pair `(2i, 2i+1)` repeated 8 times (16 bytes), so that
/// `_mm_shuffle_epi8(scales, get_scale_shuffle_k4(i))` reinterprets the two
/// selected scale bytes as 8 duplicate `i16` lanes directly, matching the
/// AVX2 version's trick of avoiding a separate sign/zero-extend step.
#[inline(always)]
unsafe fn get_scale_shuffle_k4(i: usize) -> __m128i {
    const K_SHUFFLE: [u8; 128] = [
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
        2, 3, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
        6, 7, 6, 7, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 10, 11, 10, 11, 10, 11, 10, 11,
        10, 11, 10, 11, 10, 11, 10, 11, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12,
        13, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
    ];
    _mm_loadu_si128((K_SHUFFLE.as_ptr() as *const __m128i).add(i))
}

/// Straight port of `super::avx::vec_dot_q4k_q8k`, splitting every 256-bit
/// operation into a pair of 128-bit operations on the low/high halves of the
/// original register - same technique as `vec_dot_q6k_q8k` above. The one
/// exception is `mins_and_scales`: the AVX2 version widens a *single*
/// 128-bit source (the packed 6-bit scale/min table) into one 256-bit
/// register via `_mm256_cvtepu8_epi16`, where the low 128 bits become the
/// "scales" half and the high 128 bits become the "mins" half. Since there is
/// no 128-bit source to split here, the SSE version instead widens the low
/// 8 bytes of the 16-byte source into `scales` and the high 8 bytes into
/// `mins` directly, producing the same two 8-lane `i16` registers.
pub(crate) fn vec_dot_q4k_q8k(n: usize, xs: &[BlockQ4K], ys: &[BlockQ8K]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_q4k_q8k (sse): {n} is not divisible by {QK_K}"
    );
    let mut utmp = [0u32; 4];
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    unsafe {
        let m4 = _mm_set1_epi8(0xF);

        let mut acc_lo = _mm_setzero_ps();
        let mut acc_hi = _mm_setzero_ps();
        let mut acc_m = _mm_setzero_ps();

        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = y.d * x.d.to_f32();
            let dmin = -y.d * x.dmin.to_f32();

            LittleEndian::read_u32_into(&x.scales, &mut utmp[0..3]);

            utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
            let uaux = utmp[1] & KMASK1;
            utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
            utmp[2] = uaux;
            utmp[0] &= KMASK1;

            let mut q4 = x.qs.as_ptr();
            let mut q8 = y.qs.as_ptr();

            let packed = _mm_set_epi32(
                utmp[3] as i32,
                utmp[2] as i32,
                utmp[1] as i32,
                utmp[0] as i32,
            );
            // Low 8 bytes (utmp[0], utmp[1]) -> scales; high 8 bytes
            // (utmp[2], utmp[3]) -> mins. See the function doc comment.
            let scales = _mm_cvtepu8_epi16(packed);
            let mins = _mm_cvtepu8_epi16(_mm_srli_si128(packed, 8));

            let q8sums_lo = _mm_loadu_si128(y.bsums.as_ptr() as *const __m128i);
            let q8sums_hi = _mm_loadu_si128((y.bsums.as_ptr() as *const __m128i).add(1));
            let q8s = _mm_hadd_epi16(q8sums_lo, q8sums_hi);
            let prod = _mm_madd_epi16(mins, q8s);
            acc_m = _mm_add_ps(acc_m, _mm_mul_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod)));

            let mut sumi_lo = _mm_setzero_si128();
            let mut sumi_hi = _mm_setzero_si128();

            for j in 0..QK_K / 64 {
                let scale_l = _mm_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j));
                let scale_h = _mm_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));

                let q4bits_lo = _mm_loadu_si128(q4 as *const __m128i);
                let q4bits_hi = _mm_loadu_si128(q4.add(16) as *const __m128i);
                q4 = q4.add(32);
                let q4l_lo = _mm_and_si128(q4bits_lo, m4);
                let q4l_hi = _mm_and_si128(q4bits_hi, m4);
                let q4h_lo = _mm_and_si128(_mm_srli_epi16(q4bits_lo, 4), m4);
                let q4h_hi = _mm_and_si128(_mm_srli_epi16(q4bits_hi, 4), m4);

                let q8l_lo = _mm_loadu_si128(q8 as *const __m128i);
                let q8l_hi = _mm_loadu_si128(q8.add(16) as *const __m128i);
                q8 = q8.add(32);
                let p16l_lo = _mm_maddubs_epi16(q4l_lo, q8l_lo);
                let p16l_hi = _mm_maddubs_epi16(q4l_hi, q8l_hi);
                let p16l_lo = _mm_madd_epi16(scale_l, p16l_lo);
                let p16l_hi = _mm_madd_epi16(scale_l, p16l_hi);
                sumi_lo = _mm_add_epi32(sumi_lo, p16l_lo);
                sumi_hi = _mm_add_epi32(sumi_hi, p16l_hi);

                let q8h_lo = _mm_loadu_si128(q8 as *const __m128i);
                let q8h_hi = _mm_loadu_si128(q8.add(16) as *const __m128i);
                q8 = q8.add(32);
                let p16h_lo = _mm_maddubs_epi16(q4h_lo, q8h_lo);
                let p16h_hi = _mm_maddubs_epi16(q4h_hi, q8h_hi);
                let p16h_lo = _mm_madd_epi16(scale_h, p16h_lo);
                let p16h_hi = _mm_madd_epi16(scale_h, p16h_hi);
                sumi_lo = _mm_add_epi32(sumi_lo, p16h_lo);
                sumi_hi = _mm_add_epi32(sumi_hi, p16h_hi);
            }

            let vd = _mm_set1_ps(d);
            acc_lo = _mm_add_ps(acc_lo, _mm_mul_ps(vd, _mm_cvtepi32_ps(sumi_lo)));
            acc_hi = _mm_add_ps(acc_hi, _mm_mul_ps(vd, _mm_cvtepi32_ps(sumi_hi)));
        }

        hsum_float_4(_mm_add_ps(acc_lo, acc_hi)) + hsum_float_4(acc_m)
    }
}

#[cfg(test)]
mod tests {
    use super::super::k_quants::GgmlType;
    use super::*;

    /// Directly exercises the SSE kernels (bypassing the runtime
    /// `has_avx2_fma()` dispatch in `GgmlType::vec_dot`, which would skip
    /// them entirely on a dev machine that happens to have AVX2) and checks
    /// they agree with the portable scalar reference implementation within
    /// quantization-noise tolerance.
    fn check_vec_dot_q4k_q8k_matches_scalar(seed: u64) {
        let n = QK_K * 4;
        let mut state = seed;
        let mut next = move || {
            // Small xorshift PRNG so this test has no extra dev-dependency.
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let xs: Vec<f32> = (0..n).map(|_| next()).collect();
        let ys: Vec<f32> = (0..n).map(|_| next()).collect();

        let mut x_blocks = vec![BlockQ4K::zeros(); n / QK_K];
        BlockQ4K::from_float(&xs, &mut x_blocks);
        let mut y_blocks = vec![BlockQ8K::zeros(); n / QK_K];
        BlockQ8K::from_float(&ys, &mut y_blocks);

        let expected = BlockQ4K::vec_dot_unopt(n, &x_blocks, &y_blocks);
        let actual = vec_dot_q4k_q8k(n, &x_blocks, &y_blocks);

        let tol = expected.abs().max(1.0) * 1e-2;
        assert!(
            (actual - expected).abs() <= tol,
            "sse vec_dot_q4k_q8k mismatch (seed {seed}): actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn vec_dot_q4k_q8k_matches_scalar_reference() {
        for seed in [1u64, 42, 1_000_003, 0xdead_beef] {
            check_vec_dot_q4k_q8k_matches_scalar(seed);
        }
    }
}
