use super::k_quants::{BlockQ4_0, BlockQ6K, BlockQ8K, BlockQ8_0, QK8_0, QK_K};
use crate::Result;
use half::f16;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[inline(always)]
pub(crate) unsafe fn sum_i16_pairs_float(x: __m256i) -> __m256 {
    let ones = _mm256_set1_epi16(1);
    let summed_pairs = _mm256_madd_epi16(ones, x);
    _mm256_cvtepi32_ps(summed_pairs)
}

#[inline(always)]
pub(crate) unsafe fn mul_sum_us8_pairs_float(ax: __m256i, sy: __m256i) -> __m256 {
    let dot = _mm256_maddubs_epi16(ax, sy);
    sum_i16_pairs_float(dot)
}

#[inline(always)]
pub(crate) unsafe fn hsum_float_8(x: __m256) -> f32 {
    let mut res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    _mm_cvtss_f32(res)
}

#[inline(always)]
pub(crate) unsafe fn bytes_from_nibbles_32(rsi: *const u8) -> __m256i {
    let tmp = _mm_loadu_si128(rsi as *const __m128i);
    let bytes = _mm256_insertf128_si256::<1>(_mm256_castsi128_si256(tmp), _mm_srli_epi16(tmp, 4));
    let low_mask = _mm256_set1_epi8(0xF);
    _mm256_and_si256(low_mask, bytes)
}

#[inline(always)]
pub(crate) unsafe fn mul_sum_i8_pairs_float(x: __m256i, y: __m256i) -> __m256 {
    let ax = _mm256_sign_epi8(x, x);
    let sy = _mm256_sign_epi8(y, x);
    mul_sum_us8_pairs_float(ax, sy)
}

#[inline(always)]
pub(crate) fn vec_dot_q4_0_q8_0(n: usize, xs: &[BlockQ4_0], ys: &[BlockQ8_0]) -> Result<f32> {
    let qk = QK8_0;
    let nb = n / qk;
    if n % QK8_0 != 0 {
        crate::bail!("vec_dot_q4_0_q8_0: {n} is not divisible by {qk}")
    }
    if nb % 2 != 0 {
        crate::bail!("vec_dot_q4_0_q8_0: {nb} is not even")
    }

    unsafe {
        let mut acc = _mm256_setzero_ps();
        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = _mm256_set1_ps(f16::to_f32(x.d) * f16::to_f32(y.d));
            let bx = bytes_from_nibbles_32(x.qs.as_ptr());
            let off = _mm256_set1_epi8(8);
            let bx = _mm256_sub_epi8(bx, off);
            let by = _mm256_loadu_si256(y.qs.as_ptr() as *const __m256i);
            let q = mul_sum_i8_pairs_float(bx, by);
            acc = _mm256_fmadd_ps(d, q, acc);
        }
        Ok(hsum_float_8(acc))
    }
}

#[inline(always)]
pub(crate) fn vec_dot_q8_0_q8_0(n: usize, xs: &[BlockQ8_0], ys: &[BlockQ8_0]) -> Result<f32> {
    let qk = QK8_0;
    if n % QK8_0 != 0 {
        crate::bail!("vec_dot_q8_0_q8_0: {n} is not divisible by {qk}")
    }
    unsafe {
        let mut acc = _mm256_setzero_ps();
        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = _mm256_set1_ps(f16::to_f32(x.d) * f16::to_f32(y.d));
            let bx = _mm256_loadu_si256(x.qs.as_ptr() as *const __m256i);
            let by = _mm256_loadu_si256(y.qs.as_ptr() as *const __m256i);
            let q = mul_sum_i8_pairs_float(bx, by);
            acc = _mm256_fmadd_ps(d, q, acc);
        }
        Ok(hsum_float_8(acc))
    }
}

const K_SHUFFLE: [u8; 128] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
    8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11,
    11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14,
    14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15,
];

unsafe fn get_scale_shuffle(i: usize) -> __m128i {
    _mm_loadu_si128((K_SHUFFLE.as_ptr() as *const __m128i).add(i))
}
#[inline(always)]
pub(crate) fn vec_dot_q6k_q8k(n: usize, xs: &[BlockQ6K], ys: &[BlockQ8K]) -> Result<f32> {
    let qk = QK_K;
    if n % qk != 0 {
        crate::bail!("vec_dot_q6k_8k: {n} is not divisible by {qk}")
    }

    unsafe {
        let m4 = _mm256_set1_epi8(0xF);
        let m2 = _mm256_set1_epi8(3);
        let m32s = _mm256_set1_epi8(32);
        let mut acc = _mm256_setzero_ps();
        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = y.d * x.d.to_f32();
            let mut q4 = x.ql.as_ptr();
            let mut qh = x.qh.as_ptr();
            let mut q8 = y.qs.as_ptr();

            let scales = _mm_loadu_si128(x.scales.as_ptr() as *const __m128i);
            let mut sumi = _mm256_setzero_si256();

            for j in 0..QK_K / 128 {
                let is = j * 4;
                let scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle(is));
                let scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
                let scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
                let scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));

                let q4bits1 = _mm256_loadu_si256(q4 as *const __m256i);
                q4 = q4.add(32);
                let q4bits2 = _mm256_loadu_si256(q4 as *const __m256i);
                q4 = q4.add(32);
                let q4bits_h = _mm256_loadu_si256(qh as *const __m256i);
                qh = qh.add(32);

                let q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bits_h, m2), 4);
                let q4h_1 =
                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bits_h, 2), m2), 4);
                let q4h_2 =
                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bits_h, 4), m2), 4);
                let q4h_3 =
                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bits_h, 6), m2), 4);

                let q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
                let q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
                let q4_2 =
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_2);
                let q4_3 =
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4), q4h_3);

                let q8_0 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let q8_1 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let q8_2 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let q8_3 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);

                let q8s_0 = _mm256_maddubs_epi16(m32s, q8_0);
                let q8s_1 = _mm256_maddubs_epi16(m32s, q8_1);
                let q8s_2 = _mm256_maddubs_epi16(m32s, q8_2);
                let q8s_3 = _mm256_maddubs_epi16(m32s, q8_3);

                let p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
                let p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
                let p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
                let p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

                let p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
                let p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
                let p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
                let p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

                let p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
                let p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
                let p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
                let p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
            }
            acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
        }
        Ok(hsum_float_8(acc))
    }
}
