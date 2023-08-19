use super::k_quants::{BlockQ4_0, BlockQ6K, BlockQ8K, BlockQ8_0, QK8_0, QK_K};
use crate::Result;
use half::f16;

#[allow(unused_imports)]
#[cfg(target_arch = "arm")]
use core::arch::arm::*;

#[allow(unused_imports)]
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

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

    // Generic implementation.
    let mut sumf = 0f32;
    for (xs, ys) in xs.iter().zip(ys.iter()) {
        let mut sum_i = 0;
        for j in 0..qk / 2 {
            let v0 = (xs.qs[j] & 0x0F) as i32 - 8;
            let v1 = (xs.qs[j] >> 4) as i32 - 8;
            sum_i += v0 * ys.qs[j] as i32 + v1 * ys.qs[j + qk / 2] as i32
        }
        sumf += sum_i as f32 * f16::to_f32(xs.d) * f16::to_f32(ys.d)
    }
    Ok(sumf)
}

#[inline(always)]
pub(crate) fn vec_dot_q6k_q8k(n: usize, xs: &[BlockQ6K], ys: &[BlockQ8K]) -> Result<f32> {
    if n % QK_K != 0 {
        crate::bail!("vec_dot_q6k_q8k: {n} is not divisible by {QK_K}")
    }

    let mut aux8 = [0i8; QK_K];
    let mut aux16 = [0i16; 8];
    let mut sums = [0f32; 8];
    let mut aux32 = [0f32; 8];

    for (x, y) in xs.iter().zip(ys.iter()) {
        let q4 = &x.ql;
        let qh = &x.qh;
        let q8 = &y.qs;
        aux32.fill(0f32);

        for j in (0..QK_K).step_by(128) {
            let aux8 = &mut aux8[j..];
            let q4 = &q4[j / 2..];
            let qh = &qh[j / 4..];
            for l in 0..32 {
                aux8[l] = (((q4[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32) as i8;
                aux8[l + 32] = (((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32) as i8;
                aux8[l + 64] = (((q4[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32) as i8;
                aux8[l + 96] = (((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32) as i8;
            }
        }

        for (j, &scale) in x.scales.iter().enumerate() {
            let scale = scale as f32;
            let q8 = &q8[16 * j..];
            let aux8 = &aux8[16 * j..];
            for l in 0..8 {
                aux16[l] = q8[l] as i16 * aux8[l] as i16;
            }
            for l in 0..8 {
                aux32[l] += scale * aux16[l] as f32
            }
            let q8 = &q8[8..];
            let aux8 = &aux8[8..];
            for l in 0..8 {
                aux16[l] = q8[l] as i16 * aux8[l] as i16;
            }
            for l in 0..8 {
                aux32[l] += scale * aux16[l] as f32
            }
        }

        let d = x.d.to_f32() * y.d;
        for (sum, &a) in sums.iter_mut().zip(aux32.iter()) {
            *sum += a * d;
        }
    }
    Ok(sums.iter().sum())
}
