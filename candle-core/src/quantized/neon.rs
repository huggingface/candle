use super::k_quants::{BlockQ4K, BlockQ4_0, BlockQ6K, BlockQ8K, BlockQ8_0, QK8_0, QK_K};
use crate::Result;
use byteorder::{ByteOrder, LittleEndian};

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
pub(crate) fn vec_dot_q8_0_q8_0(n: usize, xs: &[BlockQ8_0], ys: &[BlockQ8_0]) -> Result<f32> {
    let qk = QK8_0;
    if n % QK8_0 != 0 {
        crate::bail!("vec_dot_q8_0_q8_0: {n} is not divisible by {qk}")
    }
    let nb = n / QK8_0;
    if nb % 2 != 0 {
        crate::bail!("vec_dot_q8_0_q8_0: {nb} is not even")
    }
    unsafe {
        let mut sumv0 = vdupq_n_f32(0.0f32);
        let mut sumv1 = vdupq_n_f32(0.0f32);
        for i in (0..nb).step_by(2) {
            let x0 = &xs[i];
            let x1 = &xs[i + 1];
            let y0 = &ys[i];
            let y1 = &ys[i + 1];

            let x0_0 = vld1q_s8(x0.qs.as_ptr());
            let x0_1 = vld1q_s8(x0.qs.as_ptr().add(16));
            let x1_0 = vld1q_s8(x1.qs.as_ptr());
            let x1_1 = vld1q_s8(x1.qs.as_ptr().add(16));

            // load y
            let y0_0 = vld1q_s8(y0.qs.as_ptr());
            let y0_1 = vld1q_s8(y0.qs.as_ptr().add(16));
            let y1_0 = vld1q_s8(y1.qs.as_ptr());
            let y1_1 = vld1q_s8(y1.qs.as_ptr().add(16));

            // TODO dotprod once this is the intrinsics are.
            let p0_0 = vmull_s8(vget_low_s8(x0_0), vget_low_s8(y0_0));
            let p0_1 = vmull_s8(vget_high_s8(x0_0), vget_high_s8(y0_0));
            let p0_2 = vmull_s8(vget_low_s8(x0_1), vget_low_s8(y0_1));
            let p0_3 = vmull_s8(vget_high_s8(x0_1), vget_high_s8(y0_1));

            let p1_0 = vmull_s8(vget_low_s8(x1_0), vget_low_s8(y1_0));
            let p1_1 = vmull_s8(vget_high_s8(x1_0), vget_high_s8(y1_0));
            let p1_2 = vmull_s8(vget_low_s8(x1_1), vget_low_s8(y1_1));
            let p1_3 = vmull_s8(vget_high_s8(x1_1), vget_high_s8(y1_1));

            let p0 = vaddq_s32(vpaddlq_s16(p0_0), vpaddlq_s16(p0_1));
            let p1 = vaddq_s32(vpaddlq_s16(p0_2), vpaddlq_s16(p0_3));
            let p2 = vaddq_s32(vpaddlq_s16(p1_0), vpaddlq_s16(p1_1));
            let p3 = vaddq_s32(vpaddlq_s16(p1_2), vpaddlq_s16(p1_3));

            sumv0 = vmlaq_n_f32(
                sumv0,
                vcvtq_f32_s32(vaddq_s32(p0, p1)),
                x0.d.to_f32() * y0.d.to_f32(),
            );
            sumv1 = vmlaq_n_f32(
                sumv1,
                vcvtq_f32_s32(vaddq_s32(p2, p3)),
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
    let mut sum = 0f32;
    unsafe {
        let m4b = vdupq_n_u8(0xF);

        let mone = vdupq_n_u8(3);

        for (x, y) in xs.iter().zip(ys.iter()) {
            let d_all = x.d.to_f32();

            let mut q6 = x.ql.as_ptr();
            let mut qh = x.qh.as_ptr();
            let mut q8 = y.qs.as_ptr();

            let mut scale = x.scales.as_ptr();

            let q8sums = vld1q_s16_x2(y.bsums.as_ptr());
            let scales = vld1q_s8(scale);
            let q6scales = int16x8x2_t(
                vmovl_s8(vget_low_s8(scales)),
                vmovl_s8(vget_high_s8(scales)),
            );

            let prod = vaddq_s32(
                vaddq_s32(
                    vmull_s16(vget_low_s16(q8sums.0), vget_low_s16(q6scales.0)),
                    vmull_s16(vget_high_s16(q8sums.0), vget_high_s16(q6scales.0)),
                ),
                vaddq_s32(
                    vmull_s16(vget_low_s16(q8sums.1), vget_low_s16(q6scales.1)),
                    vmull_s16(vget_high_s16(q8sums.1), vget_high_s16(q6scales.1)),
                ),
            );
            let isum_mins = vaddvq_s32(prod);

            let mut isum = 0i32;

            for _j in 0..QK_K / 128 {
                let qhbits = vld1q_u8_x2(qh);
                qh = qh.add(32);
                let q6bits = vld1q_u8_x4(q6);
                q6 = q6.add(64);
                let q8bytes = vld1q_s8_x4(q8);
                q8 = q8.add(64);

                let q6h_0 = vshlq_n_u8(vandq_u8(mone, qhbits.0), 4);
                let q6h_1 = vshlq_n_u8(vandq_u8(mone, qhbits.1), 4);
                let shifted = vshrq_n_u8(qhbits.0, 2);
                let q6h_2 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
                let shifted = vshrq_n_u8(qhbits.1, 2);
                let q6h_3 = vshlq_n_u8(vandq_u8(mone, shifted), 4);

                let q6bytes_0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.0, m4b), q6h_0));
                let q6bytes_1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.1, m4b), q6h_1));
                let q6bytes_2 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.2, m4b), q6h_2));
                let q6bytes_3 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.3, m4b), q6h_3));

                // TODO: dotprod

                let p0 = vaddq_s16(
                    vmull_s8(vget_low_s8(q6bytes_0), vget_low_s8(q8bytes.0)),
                    vmull_s8(vget_high_s8(q6bytes_0), vget_high_s8(q8bytes.0)),
                );
                let p1 = vaddq_s16(
                    vmull_s8(vget_low_s8(q6bytes_1), vget_low_s8(q8bytes.1)),
                    vmull_s8(vget_high_s8(q6bytes_1), vget_high_s8(q8bytes.1)),
                );
                let (scale0, scale1) = (*scale as i32, *scale.add(1) as i32);
                isum += vaddvq_s16(p0) as i32 * scale0 + vaddvq_s16(p1) as i32 * scale1;
                scale = scale.add(2);

                let p2 = vaddq_s16(
                    vmull_s8(vget_low_s8(q6bytes_2), vget_low_s8(q8bytes.2)),
                    vmull_s8(vget_high_s8(q6bytes_2), vget_high_s8(q8bytes.2)),
                );
                let p3 = vaddq_s16(
                    vmull_s8(vget_low_s8(q6bytes_3), vget_low_s8(q8bytes.3)),
                    vmull_s8(vget_high_s8(q6bytes_3), vget_high_s8(q8bytes.3)),
                );
                let (scale0, scale1) = (*scale as i32, *scale.add(1) as i32);
                isum += vaddvq_s16(p2) as i32 * scale0 + vaddvq_s16(p3) as i32 * scale1;
                scale = scale.add(2);

                let q8bytes = vld1q_s8_x4(q8);
                q8 = q8.add(64);

                let shifted = vshrq_n_u8(qhbits.0, 4);
                let q6h_0 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
                let shifted = vshrq_n_u8(qhbits.1, 4);
                let q6h_1 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
                let shifted = vshrq_n_u8(qhbits.0, 6);
                let q6h_2 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
                let shifted = vshrq_n_u8(qhbits.1, 6);
                let q6h_3 = vshlq_n_u8(vandq_u8(mone, shifted), 4);

                let q6bytes_0 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.0, 4), q6h_0));
                let q6bytes_1 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.1, 4), q6h_1));
                let q6bytes_2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.2, 4), q6h_2));
                let q6bytes_3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.3, 4), q6h_3));

                // TODO: dotprod case.
                let p0 = vaddq_s16(
                    vmull_s8(vget_low_s8(q6bytes_0), vget_low_s8(q8bytes.0)),
                    vmull_s8(vget_high_s8(q6bytes_0), vget_high_s8(q8bytes.0)),
                );
                let p1 = vaddq_s16(
                    vmull_s8(vget_low_s8(q6bytes_1), vget_low_s8(q8bytes.1)),
                    vmull_s8(vget_high_s8(q6bytes_1), vget_high_s8(q8bytes.1)),
                );
                let (scale0, scale1) = (*scale as i32, *scale.add(1) as i32);
                isum += vaddvq_s16(p0) as i32 * scale0 + vaddvq_s16(p1) as i32 * scale1;
                scale = scale.add(2);

                let p2 = vaddq_s16(
                    vmull_s8(vget_low_s8(q6bytes_2), vget_low_s8(q8bytes.2)),
                    vmull_s8(vget_high_s8(q6bytes_2), vget_high_s8(q8bytes.2)),
                );
                let p3 = vaddq_s16(
                    vmull_s8(vget_low_s8(q6bytes_3), vget_low_s8(q8bytes.3)),
                    vmull_s8(vget_high_s8(q6bytes_3), vget_high_s8(q8bytes.3)),
                );
                let (scale0, scale1) = (*scale as i32, *scale.add(1) as i32);
                isum += vaddvq_s16(p2) as i32 * scale0 + vaddvq_s16(p3) as i32 * scale1;
                scale = scale.add(2);
            }
            sum += d_all * y.d * ((isum - 32 * isum_mins) as f32);
        }
    }
    Ok(sum)
}

#[inline(always)]
pub(crate) fn vec_dot_q4k_q8k(n: usize, xs: &[BlockQ4K], ys: &[BlockQ8K]) -> Result<f32> {
    if n % QK_K != 0 {
        crate::bail!("vec_dot_q4k_q8k: {n} is not divisible by {QK_K}")
    }
    let mut sumf = 0f32;
    let mut utmp = [0u32; 4];
    let mut scales = [0u8; 16];
    let kmask1: u32 = 0x3f3f3f3f;
    let kmask2: u32 = 0x0f0f0f0f;
    let kmask3: u32 = 0x03030303;

    unsafe {
        let m4b = vdupq_n_u8(0xF);

        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = y.d * x.d.to_f32();
            let dmin = y.d * x.dmin.to_f32();

            let q8sums = vpaddq_s16(
                vld1q_s16(y.bsums.as_ptr()),
                vld1q_s16(y.bsums.as_ptr().add(8)),
            );

            LittleEndian::read_u32_into(&x.scales, &mut utmp[0..3]);

            let mins8 = vld1_u32(
                [
                    utmp[1] & kmask1,
                    ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4),
                ]
                .as_ptr(),
            );
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[0] &= kmask1;

            let mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)));
            let prod = vaddq_s32(
                vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)),
                vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)),
            );
            sumf -= dmin * vaddvq_s32(prod) as f32;

            LittleEndian::write_u32_into(&utmp, &mut scales);

            let mut q4 = x.qs.as_ptr();
            let mut q8 = y.qs.as_ptr();

            let mut sumi1 = 0i32;
            let mut sumi2 = 0i32;

            for j in 0..QK_K / 64 {
                let q4bits = vld1q_u8_x2(q4);
                q4 = q4.add(32);
                // TODO: dotprod
                let q8bytes = vld1q_s8_x2(q8);
                q8 = q8.add(32);
                let q4bytes = int8x16x2_t(
                    vreinterpretq_s8_u8(vandq_u8(q4bits.0, m4b)),
                    vreinterpretq_s8_u8(vandq_u8(q4bits.1, m4b)),
                );
                let p0 = vaddq_s16(
                    vmull_s8(vget_low_s8(q4bytes.0), vget_low_s8(q8bytes.0)),
                    vmull_s8(vget_high_s8(q4bytes.0), vget_high_s8(q8bytes.0)),
                );
                let p1 = vaddq_s16(
                    vmull_s8(vget_low_s8(q4bytes.1), vget_low_s8(q8bytes.1)),
                    vmull_s8(vget_high_s8(q4bytes.1), vget_high_s8(q8bytes.1)),
                );
                sumi1 += vaddvq_s16(vaddq_s16(p0, p1)) as i32 * scales[2 * j] as i32;

                let q8bytes = vld1q_s8_x2(q8);
                q8 = q8.add(32);
                let q4bytes = int8x16x2_t(
                    vreinterpretq_s8_u8(vshrq_n_u8(q4bits.0, 4)),
                    vreinterpretq_s8_u8(vshrq_n_u8(q4bits.1, 4)),
                );
                let p2 = vaddq_s16(
                    vmull_s8(vget_low_s8(q4bytes.0), vget_low_s8(q8bytes.0)),
                    vmull_s8(vget_high_s8(q4bytes.0), vget_high_s8(q8bytes.0)),
                );
                let p3 = vaddq_s16(
                    vmull_s8(vget_low_s8(q4bytes.1), vget_low_s8(q8bytes.1)),
                    vmull_s8(vget_high_s8(q4bytes.1), vget_high_s8(q8bytes.1)),
                );
                sumi2 += vaddvq_s16(vaddq_s16(p2, p3)) as i32 * scales[2 * j + 1] as i32;
            }
            sumf += d * (sumi1 + sumi2) as f32;
        }
    }
    Ok(sumf)
}
