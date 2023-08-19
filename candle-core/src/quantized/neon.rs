use super::k_quants::{BlockQ4_0, BlockQ6K, BlockQ8K, BlockQ8_0, QK8_0, QK_K};
use crate::Result;

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

    unsafe {
        let mut sumv0 = vdupq_n_f32(0.0f32);
        let mut sumv1 = vdupq_n_f32(0.0f32);
        for i in (0..nb).step_by(2) {
            let x0 = &xs[i];
            let x1 = &xs[i + 1];
            let y0 = &ys[i];
            let y1 = &ys[i + 1];

            let m4b = vdupq_n_u8(0x0F);
            let s8b = vdupq_n_s8(0x8);

            let v0_0 = vld1q_u8(x0.qs.as_ptr());
            let v0_1 = vld1q_u8(x1.qs.as_ptr());

            // 4-bit -> 8-bit
            let v0_0l = vreinterpretq_s8_u8(vandq_u8(v0_0, m4b));
            let v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
            let v0_1l = vreinterpretq_s8_u8(vandq_u8(v0_1, m4b));
            let v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

            // sub 8
            let v0_0ls = vsubq_s8(v0_0l, s8b);
            let v0_0hs = vsubq_s8(v0_0h, s8b);
            let v0_1ls = vsubq_s8(v0_1l, s8b);
            let v0_1hs = vsubq_s8(v0_1h, s8b);

            // load y
            let v1_0l = vld1q_s8(y0.qs.as_ptr());
            let v1_0h = vld1q_s8(y0.qs.as_ptr().add(16));
            let v1_1l = vld1q_s8(y1.qs.as_ptr());
            let v1_1h = vld1q_s8(y1.qs.as_ptr().add(16));

            // TODO: Support dotprod when it's available outside of nightly.
            let pl0l = vmull_s8(vget_low_s8(v0_0ls), vget_low_s8(v1_0l));
            let pl0h = vmull_s8(vget_high_s8(v0_0ls), vget_high_s8(v1_0l));
            let ph0l = vmull_s8(vget_low_s8(v0_0hs), vget_low_s8(v1_0h));
            let ph0h = vmull_s8(vget_high_s8(v0_0hs), vget_high_s8(v1_0h));

            let pl1l = vmull_s8(vget_low_s8(v0_1ls), vget_low_s8(v1_1l));
            let pl1h = vmull_s8(vget_high_s8(v0_1ls), vget_high_s8(v1_1l));
            let ph1l = vmull_s8(vget_low_s8(v0_1hs), vget_low_s8(v1_1h));
            let ph1h = vmull_s8(vget_high_s8(v0_1hs), vget_high_s8(v1_1h));

            let pl0 = vaddq_s32(vpaddlq_s16(pl0l), vpaddlq_s16(pl0h));
            let ph0 = vaddq_s32(vpaddlq_s16(ph0l), vpaddlq_s16(ph0h));
            let pl1 = vaddq_s32(vpaddlq_s16(pl1l), vpaddlq_s16(pl1h));
            let ph1 = vaddq_s32(vpaddlq_s16(ph1l), vpaddlq_s16(ph1h));

            sumv0 = vmlaq_n_f32(
                sumv0,
                vcvtq_f32_s32(vaddq_s32(pl0, ph0)),
                x0.d.to_f32() * y0.d.to_f32(),
            );
            sumv1 = vmlaq_n_f32(
                sumv1,
                vcvtq_f32_s32(vaddq_s32(pl1, ph1)),
                x1.d.to_f32() * y1.d.to_f32(),
            );
        }
        Ok(vaddvq_f32(sumv0) + vaddvq_f32(sumv1))
    }
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
