use super::k_quants::{
    BlockQ2K, BlockQ3K, BlockQ4K, BlockQ4_0, BlockQ5K, BlockQ6K, BlockQ8K, BlockQ8_0, QK8_0, QK_K,
};
use crate::Result;
use byteorder::{ByteOrder, LittleEndian};
use itertools::izip;

#[allow(unused_imports)]
#[cfg(target_arch = "arm")]
use core::arch::arm::*;

#[allow(unused_imports)]
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
#[cfg(feature = "arm-nightly-feat")]
use std::arch::is_aarch64_feature_detected;

#[inline(always)]
pub(crate) fn vec_dot_q4_0_q8_0(n: usize, xs: &[BlockQ4_0], ys: &[BlockQ8_0]) -> Result<f32> {
    let qk = QK8_0;
    let nb = n / qk;
    if n % QK8_0 != 0 {
        crate::bail!("vec_dot_q4_0_q8_0: {n} is not divisible by {qk}")
    }

    unsafe {
        let mut sumv0 = vdupq_n_f32(0.0f32);
        for i in 0..nb {
            let x0 = &xs[i];
            let y0 = &ys[i];

            let m4b = vdupq_n_u8(0x0F);
            let s8b = vdupq_n_s8(0x8);

            let v0_0 = vld1q_u8(x0.qs.as_ptr());

            // 4-bit -> 8-bit
            let v0_0l = vreinterpretq_s8_u8(vandq_u8(v0_0, m4b));
            let v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));

            // sub 8
            let v0_0ls = vsubq_s8(v0_0l, s8b);
            let v0_0hs = vsubq_s8(v0_0h, s8b);

            // load y
            let v1_0l = vld1q_s8(y0.qs.as_ptr());
            let v1_0h = vld1q_s8(y0.qs.as_ptr().add(16));

            let pl0 = vdotq_s32_local(vdupq_n_s32(0), v0_0ls, v1_0l);
            let ph0 = vdotq_s32_local(vdupq_n_s32(0), v0_0hs, v1_0h);
            sumv0 = vmlaq_n_f32(
                sumv0,
                vcvtq_f32_s32(vaddq_s32(pl0, ph0)),
                x0.d.to_f32() * y0.d.to_f32(),
            );
        }
        Ok(vaddvq_f32(sumv0))
    }
}
#[inline(always)]
#[cfg(feature = "arm-nightly-feat")]
pub(crate) fn i8mm_q4_0_q8_0(
    n: usize,
    xs_0: &[BlockQ4_0],
    xs_1: &[BlockQ4_0],
    ys_0: &[BlockQ8_0],
    ys_1: &[BlockQ8_0],
) -> Result<[f32; 4]> {
    let qk = QK8_0;
    let nb = n / qk;
    if n % QK8_0 != 0 {
        crate::bail!("i8mm_q4_0_q8_0: {n} is not divisible by {qk}")
    }
    //let (xs_0, xs_1) = xs.split_at_mut(xs.len() / 2);
    //let (ys_0, ys_1) = ys.split_at_mut(ys.len() / 2);
    assert_eq!(xs_0.len(), xs_1.len());
    assert_eq!(ys_0.len(), ys_1.len());
    assert_eq!(xs_0.len(), ys_0.len());

    unsafe {
        let mut sum_f32 = vdupq_n_f32(0.0);

        let m4b = vdupq_n_u8(0x0F);
        let s8b = vdupq_n_s8(0x8);

        for i in 0..nb {
            let x0 = &xs_0[i];
            let x1 = &xs_1[i];
            let y0 = &ys_0[i];
            let y1 = &ys_1[i];

            let factor_00: f32 = x0.d.to_f32() * y0.d.to_f32();
            let factor_01: f32 = x1.d.to_f32() * y0.d.to_f32();
            let factor_10: f32 = x0.d.to_f32() * y1.d.to_f32();
            let factor_11: f32 = x1.d.to_f32() * y1.d.to_f32();

            let xv0 = vld1q_u8(x0.qs.as_ptr()); //16xu8
            let xv1 = vld1q_u8(x1.qs.as_ptr()); //16xu8

            // convert u8s to i4s so we have equal amount of row elements
            // and columns elements to multiply
            let xv0_0 = vreinterpretq_s8_u8(vandq_u8(xv0, m4b));
            let xv0_1 = vreinterpretq_s8_u8(vshrq_n_u8(xv0, 4));
            let xv1_0 = vreinterpretq_s8_u8(vandq_u8(xv1, m4b));
            let xv1_1 = vreinterpretq_s8_u8(vshrq_n_u8(xv1, 4));

            // sub 8
            let xv0_0s = vsubq_s8(xv0_0, s8b);
            let xv0_1s = vsubq_s8(xv0_1, s8b);
            let xv1_0s = vsubq_s8(xv1_0, s8b);
            let xv1_1s = vsubq_s8(xv1_1, s8b);
            //end of conversion

            let yv0_0 = vld1q_s8(y0.qs.as_ptr()); //16xi8
            let yv0_1 = vld1q_s8(y0.qs.as_ptr().add(16)); // 16xi8
            let yv1_0 = vld1q_s8(y1.qs.as_ptr()); //16xi8
            let yv1_1 = vld1q_s8(y1.qs.as_ptr().add(16)); // 16xi8

            let i8mm = i8mm_params::new(xv0_0s, xv0_1s, xv1_0s, xv1_1s, yv0_0, yv0_1, yv1_0, yv1_1);
            let loop_sum_s32 = i8mm.calculate(vdupq_n_s32(0));

            // scaling
            let factor_elems: [f32; 4] = [factor_00, factor_01, factor_10, factor_11];
            let rawptr = &factor_elems as *const f32;
            let factor: float32x4_t = vld1q_f32(rawptr);
            let loop_sum_f32 = vcvtq_f32_s32(loop_sum_s32);

            sum_f32 = vmlaq_f32(sum_f32, loop_sum_f32, factor);
        }
        // extract elements of the vector register
        let f0 = vgetq_lane_f32(sum_f32, 0);
        let f1 = vgetq_lane_f32(sum_f32, 1);
        let f2 = vgetq_lane_f32(sum_f32, 2);
        let f3 = vgetq_lane_f32(sum_f32, 3);
        let res: [f32; 4] = [f0, f1, f2, f3];
        Ok(res)
    }
}
#[inline(always)]
pub(crate) fn vec_dot_q8_0_q8_0(n: usize, xs: &[BlockQ8_0], ys: &[BlockQ8_0]) -> Result<f32> {
    let qk = QK8_0;
    if n % QK8_0 != 0 {
        crate::bail!("vec_dot_q8_0_q8_0: {n} is not divisible by {qk}")
    }
    let nb = n / QK8_0;
    unsafe {
        let mut sumv0 = vdupq_n_f32(0.0f32);
        for i in 0..nb {
            let x0 = &xs[i];
            let y0 = &ys[i];

            let x0_0 = vld1q_s8(x0.qs.as_ptr());
            let x0_1 = vld1q_s8(x0.qs.as_ptr().add(16));

            // load y
            let y0_0 = vld1q_s8(y0.qs.as_ptr());
            let y0_1 = vld1q_s8(y0.qs.as_ptr().add(16));

            let p0 = vdotq_s32_local(vdupq_n_s32(0), x0_0, y0_0);
            let p1 = vdotq_s32_local(vdupq_n_s32(0), x0_1, y0_1);

            sumv0 = vmlaq_n_f32(
                sumv0,
                vcvtq_f32_s32(vaddq_s32(p0, p1)),
                x0.d.to_f32() * y0.d.to_f32(),
            );
        }
        Ok(vaddvq_f32(sumv0))
    }
}
#[inline(always)]
#[cfg(feature = "arm-nightly-feat")]
pub(crate) fn i8mm_q8_0_q8_0(
    n: usize,
    xs_0: &[BlockQ8_0],
    xs_1: &[BlockQ8_0],
    ys_0: &[BlockQ8_0],
    ys_1: &[BlockQ8_0],
) -> Result<[f32; 4]> {
    assert_eq!(xs_0.len(), xs_1.len());
    assert_eq!(ys_0.len(), ys_1.len());
    assert_eq!(xs_0.len(), ys_0.len());
    let qk = QK8_0;
    if n % QK8_0 != 0 {
        crate::bail!("i8mm_q8_0_q8_0: {n} is not divisible by {qk}")
    }
    let nb = n / QK8_0;
    unsafe {
        let mut sum_f32 = vdupq_n_f32(0.0);

        for i in 0..nb {
            let x0 = &xs_0[i];
            let x1 = &xs_1[i];
            let y0 = &ys_0[i];
            let y1 = &ys_1[i];

            let factor_00: f32 = x0.d.to_f32() * y0.d.to_f32();
            let factor_01: f32 = x1.d.to_f32() * y0.d.to_f32();
            let factor_10: f32 = x0.d.to_f32() * y1.d.to_f32();
            let factor_11: f32 = x1.d.to_f32() * y1.d.to_f32();

            let xv0_0 = vld1q_s8(x0.qs.as_ptr());
            let xv0_1 = vld1q_s8(x0.qs.as_ptr().add(16));
            let xv1_0 = vld1q_s8(x1.qs.as_ptr());
            let xv1_1 = vld1q_s8(x1.qs.as_ptr().add(16));

            let yv0_0 = vld1q_s8(y0.qs.as_ptr());
            let yv0_1 = vld1q_s8(y0.qs.as_ptr().add(16));
            let yv1_0 = vld1q_s8(y1.qs.as_ptr());
            let yv1_1 = vld1q_s8(y1.qs.as_ptr().add(16));

            let i8mm = i8mm_params::new(xv0_0, xv0_1, xv1_0, xv1_1, yv0_0, yv0_1, yv1_0, yv1_1);
            let loop_sum_s32 = i8mm.calculate(vdupq_n_s32(0));

            // scaling
            let factor_elems: [f32; 4] = [factor_00, factor_01, factor_10, factor_11];
            let rawptr = &factor_elems as *const f32;
            let factor: float32x4_t = vld1q_f32(rawptr);
            let loop_sum_f32 = vcvtq_f32_s32(loop_sum_s32);

            sum_f32 = vmlaq_f32(sum_f32, loop_sum_f32, factor);
        }
        // extract elements of the vector register
        let f0 = vgetq_lane_f32(sum_f32, 0);
        let f1 = vgetq_lane_f32(sum_f32, 1);
        let f2 = vgetq_lane_f32(sum_f32, 2);
        let f3 = vgetq_lane_f32(sum_f32, 3);
        let res: [f32; 4] = [f0, f1, f2, f3];
        Ok(res)
    }
}

#[inline(always)]
pub(crate) fn vec_dot_q8k_q8k(n: usize, xs: &[BlockQ8K], ys: &[BlockQ8K]) -> Result<f32> {
    let qk = QK_K;
    if n % QK_K != 0 {
        crate::bail!("vec_dot_q8k_q8k: {n} is not divisible by {qk}")
    }

    let mut sumf = 0f32;
    for (xs, ys) in xs.iter().zip(ys.iter()) {
        unsafe {
            let mut sum_i = vdupq_n_s32(0);
            let scale = xs.d * ys.d;
            let xs = xs.qs.as_ptr();
            let ys = ys.qs.as_ptr();
            for i in (0..QK_K).step_by(16) {
                let xs = vld1q_s8(xs.add(i));
                let ys = vld1q_s8(ys.add(i));
                let xy = vdotq_s32(xs, ys);
                sum_i = vaddq_s32(sum_i, xy)
            }
            sumf += vaddvq_s32(sum_i) as f32 * scale
        }
    }
    Ok(sumf)
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
        let mzero = vdupq_n_s32(0);

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

                let p0 = vdotq_s32_local(mzero, q6bytes_0, q8bytes.0);
                let p1 = vdotq_s32_local(mzero, q6bytes_1, q8bytes.1);
                let (scale0, scale1) = (*scale as i32, *scale.add(1) as i32);
                isum += vaddvq_s32(p0) * scale0 + vaddvq_s32(p1) * scale1;
                scale = scale.add(2);

                let p2 = vdotq_s32_local(mzero, q6bytes_2, q8bytes.2);
                let p3 = vdotq_s32_local(mzero, q6bytes_3, q8bytes.3);
                let (scale0, scale1) = (*scale as i32, *scale.add(1) as i32);
                isum += vaddvq_s32(p2) * scale0 + vaddvq_s32(p3) * scale1;
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

                let p0 = vdotq_s32_local(mzero, q6bytes_0, q8bytes.0);
                let p1 = vdotq_s32_local(mzero, q6bytes_1, q8bytes.1);
                let (scale0, scale1) = (*scale as i32, *scale.add(1) as i32);
                isum += vaddvq_s32(p0) * scale0 + vaddvq_s32(p1) * scale1;
                scale = scale.add(2);

                let p2 = vdotq_s32_local(mzero, q6bytes_2, q8bytes.2);
                let p3 = vdotq_s32_local(mzero, q6bytes_3, q8bytes.3);
                let (scale0, scale1) = (*scale as i32, *scale.add(1) as i32);
                isum += vaddvq_s32(p2) * scale0 + vaddvq_s32(p3) * scale1;
                scale = scale.add(2);
            }
            sum += d_all * y.d * ((isum - 32 * isum_mins) as f32);
        }
    }
    Ok(sum)
}
// QK_K = 256
#[inline(always)]
#[cfg(feature = "arm-nightly-feat")]
pub(crate) fn i8mm_q6k_q8k(
    _n: usize,
    xs_0: &[BlockQ6K],
    xs_1: &[BlockQ6K],
    ys_0: &[BlockQ8K],
    ys_1: &[BlockQ8K],
) -> Result<[f32; 4]> {
    unsafe {
        let mut fsum = vdupq_n_f32(0.0);
        let m4b = vdupq_n_u8(0xF);
        let mone = vdupq_n_u8(3);
        for (x0, x1, y0, y1) in izip!(xs_0, xs_1, ys_0, ys_1) {
            let d_00: f32 = x0.d.to_f32() * y0.d;
            let d_01: f32 = x1.d.to_f32() * y0.d;
            let d_10: f32 = x0.d.to_f32() * y1.d;
            let d_11: f32 = x1.d.to_f32() * y1.d;

            let mut q6_0 = x0.ql.as_ptr();
            let mut q6_1 = x1.ql.as_ptr();
            let mut qh_0 = x0.qh.as_ptr();
            let mut qh_1 = x1.qh.as_ptr();
            let mut q8_0 = y0.qs.as_ptr();
            let mut q8_1 = y1.qs.as_ptr();

            let mut scale_0 = x0.scales.as_ptr();
            let mut scale_1 = x1.scales.as_ptr();

            let q8sums_0 = vld1q_s16_x2(y0.bsums.as_ptr());
            let q8sums_1 = vld1q_s16_x2(y1.bsums.as_ptr());
            let scales_0 = vld1q_s8(scale_0);
            let scales_1 = vld1q_s8(scale_1);

            let q6scales_0 = int16x8x2_t(
                vmovl_s8(vget_low_s8(scales_0)),
                vmovl_s8(vget_high_s8(scales_0)),
            );
            let q6scales_1 = int16x8x2_t(
                vmovl_s8(vget_low_s8(scales_1)),
                vmovl_s8(vget_high_s8(scales_1)),
            );

            // y0 x0
            let prod_00 = vaddq_s32(
                vaddq_s32(
                    vmull_s16(vget_low_s16(q8sums_0.0), vget_low_s16(q6scales_0.0)),
                    vmull_s16(vget_high_s16(q8sums_0.0), vget_high_s16(q6scales_0.0)),
                ),
                vaddq_s32(
                    vmull_s16(vget_low_s16(q8sums_0.1), vget_low_s16(q6scales_0.1)),
                    vmull_s16(vget_high_s16(q8sums_0.1), vget_high_s16(q6scales_0.1)),
                ),
            );
            // y0 x1
            let prod_01 = vaddq_s32(
                vaddq_s32(
                    vmull_s16(vget_low_s16(q8sums_0.0), vget_low_s16(q6scales_1.0)),
                    vmull_s16(vget_high_s16(q8sums_0.0), vget_high_s16(q6scales_1.0)),
                ),
                vaddq_s32(
                    vmull_s16(vget_low_s16(q8sums_0.1), vget_low_s16(q6scales_1.1)),
                    vmull_s16(vget_high_s16(q8sums_0.1), vget_high_s16(q6scales_1.1)),
                ),
            );
            // y1 x0
            let prod_10 = vaddq_s32(
                vaddq_s32(
                    vmull_s16(vget_low_s16(q8sums_1.0), vget_low_s16(q6scales_0.0)),
                    vmull_s16(vget_high_s16(q8sums_1.0), vget_high_s16(q6scales_0.0)),
                ),
                vaddq_s32(
                    vmull_s16(vget_low_s16(q8sums_1.1), vget_low_s16(q6scales_0.1)),
                    vmull_s16(vget_high_s16(q8sums_1.1), vget_high_s16(q6scales_0.1)),
                ),
            );
            // y1 x1
            let prod_11 = vaddq_s32(
                vaddq_s32(
                    vmull_s16(vget_low_s16(q8sums_1.0), vget_low_s16(q6scales_1.0)),
                    vmull_s16(vget_high_s16(q8sums_1.0), vget_high_s16(q6scales_1.0)),
                ),
                vaddq_s32(
                    vmull_s16(vget_low_s16(q8sums_1.1), vget_low_s16(q6scales_1.1)),
                    vmull_s16(vget_high_s16(q8sums_1.1), vget_high_s16(q6scales_1.1)),
                ),
            );
            let sumi_mins_00 = vaddvq_s32(prod_00);
            let sumi_mins_01 = vaddvq_s32(prod_01);
            let sumi_mins_10 = vaddvq_s32(prod_10);
            let sumi_mins_11 = vaddvq_s32(prod_11);

            let mut isum = vdupq_n_s32(0);
            for _j in 0..QK_K / 128 {
                let qhbits_0 = vld1q_u8_x2(qh_0);
                let qhbits_1 = vld1q_u8_x2(qh_1);
                qh_0 = qh_0.add(32);
                qh_1 = qh_1.add(32);

                let q6bits_0 = vld1q_u8_x4(q6_0);
                let q6bits_1 = vld1q_u8_x4(q6_1);
                q6_0 = q6_0.add(64);
                q6_1 = q6_1.add(64);

                let q8bytes0_0 = vld1q_s8_x4(q8_0);
                let q8bytes1_0 = vld1q_s8_x4(q8_1);
                q8_0 = q8_0.add(64);
                q8_1 = q8_1.add(64);

                let q8bytes0_1 = vld1q_s8_x4(q8_0);
                let q8bytes1_1 = vld1q_s8_x4(q8_1);
                q8_0 = q8_0.add(64);
                q8_1 = q8_1.add(64);

                let q6h0_0 = vshlq_n_u8(vandq_u8(mone, qhbits_0.0), 4);
                let q6h0_1 = vshlq_n_u8(vandq_u8(mone, qhbits_0.1), 4);
                let shifted = vshrq_n_u8(qhbits_0.0, 2);
                let q6h0_2 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
                let shifted = vshrq_n_u8(qhbits_0.1, 2);
                let q6h0_3 = vshlq_n_u8(vandq_u8(mone, shifted), 4);

                let q6h1_0 = vshlq_n_u8(vandq_u8(mone, qhbits_1.0), 4);
                let q6h1_1 = vshlq_n_u8(vandq_u8(mone, qhbits_1.1), 4);
                let shifted = vshrq_n_u8(qhbits_1.0, 2);
                let q6h1_2 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
                let shifted = vshrq_n_u8(qhbits_1.1, 2);
                let q6h1_3 = vshlq_n_u8(vandq_u8(mone, shifted), 4);

                let q6bytes0_0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits_0.0, m4b), q6h0_0));
                let q6bytes0_1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits_0.1, m4b), q6h0_1));
                let q6bytes0_2 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits_0.2, m4b), q6h0_2));
                let q6bytes0_3 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits_0.3, m4b), q6h0_3));

                let q6bytes1_0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits_1.0, m4b), q6h1_0));
                let q6bytes1_1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits_1.1, m4b), q6h1_1));
                let q6bytes1_2 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits_1.2, m4b), q6h1_2));
                let q6bytes1_3 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits_1.3, m4b), q6h1_3));

                let sc = i8mm_x_scales::new(&x_scales {
                    x00: *scale_0.add(0) as i32,
                    x01: *scale_0.add(1) as i32,
                    x10: *scale_1.add(0) as i32,
                    x11: *scale_1.add(1) as i32,
                });
                let i8mm = i8mm_params::new(
                    q6bytes0_0,
                    q6bytes0_1,
                    q6bytes1_0,
                    q6bytes1_1,
                    q8bytes0_0.0,
                    q8bytes0_0.1,
                    q8bytes1_0.0,
                    q8bytes1_0.1,
                );
                isum = i8mm.calculate_with_scales(isum, sc);

                let sc = i8mm_x_scales::new(&x_scales {
                    x00: *scale_0.add(2) as i32,
                    x01: *scale_0.add(3) as i32,
                    x10: *scale_1.add(2) as i32,
                    x11: *scale_1.add(3) as i32,
                });
                let i8mm = i8mm_params::new(
                    q6bytes0_2,
                    q6bytes0_3,
                    q6bytes1_2,
                    q6bytes1_3,
                    q8bytes0_0.2,
                    q8bytes0_0.3,
                    q8bytes1_0.2,
                    q8bytes1_0.3,
                );
                isum = i8mm.calculate_with_scales(isum, sc);

                scale_0 = scale_0.add(4);
                scale_1 = scale_1.add(4);

                let shifted = vshrq_n_u8(qhbits_0.0, 4);
                let q6h0_0 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
                let shifted = vshrq_n_u8(qhbits_0.1, 4);
                let q6h0_1 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
                let shifted = vshrq_n_u8(qhbits_0.0, 6);
                let q6h0_2 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
                let shifted = vshrq_n_u8(qhbits_0.1, 6);
                let q6h0_3 = vshlq_n_u8(vandq_u8(mone, shifted), 4);

                let shifted = vshrq_n_u8(qhbits_1.0, 4);
                let q6h1_0 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
                let shifted = vshrq_n_u8(qhbits_1.1, 4);
                let q6h1_1 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
                let shifted = vshrq_n_u8(qhbits_1.0, 6);
                let q6h1_2 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
                let shifted = vshrq_n_u8(qhbits_1.1, 6);
                let q6h1_3 = vshlq_n_u8(vandq_u8(mone, shifted), 4);

                let q6bytes0_0 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits_0.0, 4), q6h0_0));
                let q6bytes0_1 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits_0.1, 4), q6h0_1));
                let q6bytes0_2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits_0.2, 4), q6h0_2));
                let q6bytes0_3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits_0.3, 4), q6h0_3));

                let q6bytes1_0 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits_1.0, 4), q6h1_0));
                let q6bytes1_1 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits_1.1, 4), q6h1_1));
                let q6bytes1_2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits_1.2, 4), q6h1_2));
                let q6bytes1_3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits_1.3, 4), q6h1_3));

                let sc = i8mm_x_scales::new(&x_scales {
                    x00: *scale_0.add(0) as i32,
                    x01: *scale_0.add(1) as i32,
                    x10: *scale_1.add(0) as i32,
                    x11: *scale_1.add(1) as i32,
                });
                let i8mm = i8mm_params::new(
                    q6bytes0_0,
                    q6bytes0_1,
                    q6bytes1_0,
                    q6bytes1_1,
                    q8bytes0_1.0,
                    q8bytes0_1.1,
                    q8bytes1_1.0,
                    q8bytes1_1.1,
                );
                isum = i8mm.calculate_with_scales(isum, sc);

                let sc = i8mm_x_scales::new(&x_scales {
                    x00: *scale_0.add(2) as i32,
                    x01: *scale_0.add(3) as i32,
                    x10: *scale_1.add(2) as i32,
                    x11: *scale_1.add(3) as i32,
                });
                let i8mm = i8mm_params::new(
                    q6bytes0_2,
                    q6bytes0_3,
                    q6bytes1_2,
                    q6bytes1_3,
                    q8bytes0_1.2,
                    q8bytes0_1.3,
                    q8bytes1_1.2,
                    q8bytes1_1.3,
                );
                isum = i8mm.calculate_with_scales(isum, sc);

                scale_0 = scale_0.add(4);
                scale_1 = scale_1.add(4);
            }
            let factor_elems: [f32; 4] = [d_00, d_01, d_10, d_11];
            let rawptr = &factor_elems as *const f32;
            let factor: float32x4_t = vld1q_f32(rawptr);
            //sum += d_all * y.d * ((isum - 32 * isum_mins) as f32);
            let sumi_mins_arr: [i32; 4] = [
                -sumi_mins_00 * 32,
                -sumi_mins_01 * 32,
                -sumi_mins_10 * 32,
                -sumi_mins_11 * 32,
            ];
            let rawptr = &sumi_mins_arr as *const i32;
            let sumi_minsv: int32x4_t = vld1q_s32(rawptr);
            fsum = vmlaq_f32(fsum, factor, vcvtq_f32_s32(vaddq_s32(sumi_minsv, isum)));
        }
        // extract elements of the vector register
        let f0 = vgetq_lane_f32(fsum, 0);
        let f1 = vgetq_lane_f32(fsum, 1);
        let f2 = vgetq_lane_f32(fsum, 2);
        let f3 = vgetq_lane_f32(fsum, 3);
        let res: [f32; 4] = [f0, f1, f2, f3];
        Ok(res)
    }
}

#[inline(always)]
pub(crate) fn vec_dot_q5k_q8k(n: usize, xs: &[BlockQ5K], ys: &[BlockQ8K]) -> Result<f32> {
    if n % QK_K != 0 {
        crate::bail!("vec_dot_q5k_q8k: {n} is not divisible by {QK_K}")
    }
    let mut sumf = 0f32;
    let mut utmp = [0u32; 4];
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    unsafe {
        let m4b = vdupq_n_u8(0xF);
        let mzero = vdupq_n_s32(0);
        let mone = vdupq_n_u8(1);
        let mtwo = vdupq_n_u8(2);

        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = y.d * x.d.to_f32();
            let dmin = y.d * x.dmin.to_f32();

            let q8sums = vpaddq_s16(
                vld1q_s16(y.bsums.as_ptr()),
                vld1q_s16(y.bsums.as_ptr().add(8)),
            );

            LittleEndian::read_u32_into(&x.scales, &mut utmp[0..3]);

            utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
            let uaux = utmp[1] & KMASK1;
            utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
            utmp[2] = uaux;
            utmp[0] &= KMASK1;

            let mins8 = vld1_u8((utmp.as_ptr() as *const u8).add(8));
            let mins = vreinterpretq_s16_u16(vmovl_u8(mins8));
            let prod = vaddq_s32(
                vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)),
                vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)),
            );
            let sumi_mins = vaddvq_s32(prod);

            let mut scales = utmp.as_ptr() as *const u8;

            let mut q5 = x.qs.as_ptr();
            let mut q8 = y.qs.as_ptr();

            let mut qhbits = vld1q_u8_x2(x.qh.as_ptr());

            let mut sumi = 0i32;

            for _j in 0..QK_K / 64 {
                let q5bits = vld1q_u8_x2(q5);
                q5 = q5.add(32);
                let q8bytes = vld1q_s8_x4(q8);
                q8 = q8.add(64);

                let q5h_0 = vshlq_n_u8(vandq_u8(mone, qhbits.0), 4);
                let q5h_1 = vshlq_n_u8(vandq_u8(mone, qhbits.1), 4);
                let q5h_2 = vshlq_n_u8(vandq_u8(mtwo, qhbits.0), 3);
                let q5h_3 = vshlq_n_u8(vandq_u8(mtwo, qhbits.1), 3);
                qhbits.0 = vshrq_n_u8(qhbits.0, 2);
                qhbits.1 = vshrq_n_u8(qhbits.1, 2);

                let q5bytes_0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits.0, m4b), q5h_0));
                let q5bytes_1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits.1, m4b), q5h_1));
                let q5bytes_2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits.0, 4), q5h_2));
                let q5bytes_3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits.1, 4), q5h_3));

                let p0 = vdotq_s32_local(mzero, q5bytes_0, q8bytes.0);
                let p1 = vdotq_s32_local(mzero, q5bytes_1, q8bytes.1);
                sumi += vaddvq_s32(vaddq_s32(p0, p1)) * *scales as i32;
                scales = scales.add(1);

                let p2 = vdotq_s32_local(mzero, q5bytes_2, q8bytes.2);
                let p3 = vdotq_s32_local(mzero, q5bytes_3, q8bytes.3);
                sumi += vaddvq_s32(vaddq_s32(p2, p3)) * *scales as i32;
                scales = scales.add(1);
            }
            sumf += d * sumi as f32 - dmin * sumi_mins as f32;
        }
    }
    Ok(sumf)
}
#[inline(always)]
#[cfg(feature = "arm-nightly-feat")]
pub(crate) fn i8mm_q5k_q8k(
    _n: usize,
    xs_0: &[BlockQ5K],
    xs_1: &[BlockQ5K],
    ys_0: &[BlockQ8K],
    ys_1: &[BlockQ8K],
) -> Result<[f32; 4]> {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;
    unsafe {
        let mut sumfv = vdupq_n_f32(0.0);
        let mut utmp_0 = [0u32; 4];
        let mut utmp_1 = [0u32; 4];
        let m4b = vdupq_n_u8(0xF);
        let mone = vdupq_n_u8(1);
        let mtwo = vdupq_n_u8(2);
        let mzero = vdupq_n_s32(0);
        for (x0, x1, y0, y1) in izip!(xs_0, xs_1, ys_0, ys_1) {
            let d_00: f32 = x0.d.to_f32() * y0.d;
            let d_01: f32 = x1.d.to_f32() * y0.d;
            let d_10: f32 = x0.d.to_f32() * y1.d;
            let d_11: f32 = x1.d.to_f32() * y1.d;

            let dmin_00 = -y0.d * x0.dmin.to_f32();
            let dmin_01 = -y0.d * x1.dmin.to_f32();
            let dmin_10 = -y1.d * x0.dmin.to_f32();
            let dmin_11 = -y1.d * x1.dmin.to_f32();

            let q8sums_0 = vpaddq_s16(
                vld1q_s16(y0.bsums.as_ptr()),
                vld1q_s16(y0.bsums.as_ptr().add(8)),
            );
            let q8sums_1 = vpaddq_s16(
                vld1q_s16(y1.bsums.as_ptr()),
                vld1q_s16(y1.bsums.as_ptr().add(8)),
            );

            LittleEndian::read_u32_into(&x0.scales, &mut utmp_0[0..3]);
            LittleEndian::read_u32_into(&x1.scales, &mut utmp_1[0..3]);

            utmp_0[3] = ((utmp_0[2] >> 4) & KMASK2) | (((utmp_0[1] >> 6) & KMASK3) << 4);
            let uaux = utmp_0[1] & KMASK1;
            utmp_0[1] = (utmp_0[2] & KMASK2) | (((utmp_0[0] >> 6) & KMASK3) << 4);
            utmp_0[2] = uaux;
            utmp_0[0] &= KMASK1;

            utmp_1[3] = ((utmp_1[2] >> 4) & KMASK2) | (((utmp_1[1] >> 6) & KMASK3) << 4);
            let uaux = utmp_1[1] & KMASK1;
            utmp_1[1] = (utmp_1[2] & KMASK2) | (((utmp_1[0] >> 6) & KMASK3) << 4);
            utmp_1[2] = uaux;
            utmp_1[0] &= KMASK1;

            let mins8_0 = vld1_u8((utmp_0.as_ptr() as *const u8).add(8));
            let mins8_1 = vld1_u8((utmp_1.as_ptr() as *const u8).add(8));
            let mins_0 = vreinterpretq_s16_u16(vmovl_u8(mins8_0));
            let mins_1 = vreinterpretq_s16_u16(vmovl_u8(mins8_1));

            // y0 x0
            let prod_00 = vaddq_s32(
                vmull_s16(vget_low_s16(q8sums_0), vget_low_s16(mins_0)),
                vmull_s16(vget_high_s16(q8sums_0), vget_high_s16(mins_0)),
            );
            // y0 x1
            let prod_01 = vaddq_s32(
                vmull_s16(vget_low_s16(q8sums_0), vget_low_s16(mins_1)),
                vmull_s16(vget_high_s16(q8sums_0), vget_high_s16(mins_1)),
            );
            // y1 x0
            let prod_10 = vaddq_s32(
                vmull_s16(vget_low_s16(q8sums_1), vget_low_s16(mins_0)),
                vmull_s16(vget_high_s16(q8sums_1), vget_high_s16(mins_0)),
            );
            // y1 x1
            let prod_11 = vaddq_s32(
                vmull_s16(vget_low_s16(q8sums_1), vget_low_s16(mins_1)),
                vmull_s16(vget_high_s16(q8sums_1), vget_high_s16(mins_1)),
            );
            let sumi_mins_00 = vaddvq_s32(prod_00);
            let sumi_mins_01 = vaddvq_s32(prod_01);
            let sumi_mins_10 = vaddvq_s32(prod_10);
            let sumi_mins_11 = vaddvq_s32(prod_11);

            let mut scales_0 = utmp_0.as_ptr() as *const u8;
            let mut scales_1 = utmp_1.as_ptr() as *const u8;

            let mut q5_0 = x0.qs.as_ptr();
            let mut q5_1 = x1.qs.as_ptr();
            let mut q8_0 = y0.qs.as_ptr();
            let mut q8_1 = y1.qs.as_ptr();

            let mut qhbits_0 = vld1q_u8_x2(x0.qh.as_ptr());
            let mut qhbits_1 = vld1q_u8_x2(x1.qh.as_ptr());

            let mut isum = vdupq_n_s32(0);
            for _j in 0..QK_K / 64 {
                let q5bits_0 = vld1q_u8_x2(q5_0);
                let q5bits_1 = vld1q_u8_x2(q5_1);
                q5_0 = q5_0.add(32);
                q5_1 = q5_1.add(32);
                let q8bytes_0 = vld1q_s8_x4(q8_0);
                let q8bytes_1 = vld1q_s8_x4(q8_1);
                q8_0 = q8_0.add(64);
                q8_1 = q8_1.add(64);

                let q5h0_0 = vshlq_n_u8(vandq_u8(mone, qhbits_0.0), 4);
                let q5h0_1 = vshlq_n_u8(vandq_u8(mone, qhbits_0.1), 4);
                let q5h0_2 = vshlq_n_u8(vandq_u8(mtwo, qhbits_0.0), 3);
                let q5h0_3 = vshlq_n_u8(vandq_u8(mtwo, qhbits_0.1), 3);

                let q5h1_0 = vshlq_n_u8(vandq_u8(mone, qhbits_1.0), 4);
                let q5h1_1 = vshlq_n_u8(vandq_u8(mone, qhbits_1.1), 4);
                let q5h1_2 = vshlq_n_u8(vandq_u8(mtwo, qhbits_1.0), 3);
                let q5h1_3 = vshlq_n_u8(vandq_u8(mtwo, qhbits_1.1), 3);

                qhbits_0.0 = vshrq_n_u8(qhbits_0.0, 2);
                qhbits_0.1 = vshrq_n_u8(qhbits_0.1, 2);
                qhbits_1.0 = vshrq_n_u8(qhbits_1.0, 2);
                qhbits_1.1 = vshrq_n_u8(qhbits_1.1, 2);

                let q5bytes0_0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits_0.0, m4b), q5h0_0));
                let q5bytes0_1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits_0.1, m4b), q5h0_1));
                let q5bytes0_2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits_0.0, 4), q5h0_2));
                let q5bytes0_3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits_0.1, 4), q5h0_3));

                let q5bytes1_0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits_1.0, m4b), q5h1_0));
                let q5bytes1_1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits_1.1, m4b), q5h1_1));
                let q5bytes1_2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits_1.0, 4), q5h1_2));
                let q5bytes1_3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits_1.1, 4), q5h1_3));

                let i8mm = i8mm_params::new(
                    q5bytes0_0,
                    q5bytes0_1,
                    q5bytes1_0,
                    q5bytes1_1,
                    q8bytes_0.0,
                    q8bytes_0.1,
                    q8bytes_1.0,
                    q8bytes_1.1,
                );
                let i8mmres = i8mm.calculate(mzero);

                let sc_arr = [
                    *scales_0 as i32,
                    *scales_1 as i32,
                    *scales_0 as i32,
                    *scales_1 as i32,
                ];
                let rawptr = &sc_arr as *const i32;
                let sc: int32x4_t = vld1q_s32(rawptr);
                isum = vmlaq_s32(isum, i8mmres, sc);

                scales_0 = scales_0.add(1);
                scales_1 = scales_1.add(1);

                let i8mm = i8mm_params::new(
                    q5bytes0_2,
                    q5bytes0_3,
                    q5bytes1_2,
                    q5bytes1_3,
                    q8bytes_0.2,
                    q8bytes_0.3,
                    q8bytes_1.2,
                    q8bytes_1.3,
                );
                let i8mmres = i8mm.calculate(mzero);
                let sc_arr = [
                    *scales_0 as i32,
                    *scales_1 as i32,
                    *scales_0 as i32,
                    *scales_1 as i32,
                ];
                let rawptr = &sc_arr as *const i32;
                let sc: int32x4_t = vld1q_s32(rawptr);
                isum = vmlaq_s32(isum, i8mmres, sc);

                scales_0 = scales_0.add(1);
                scales_1 = scales_1.add(1);
            }
            let factor_elems: [f32; 4] = [d_00, d_01, d_10, d_11];
            let rawptr = &factor_elems as *const f32;
            let factor: float32x4_t = vld1q_f32(rawptr);

            let dmin_arr: [f32; 4] = [dmin_00, dmin_01, dmin_10, dmin_11];
            let rawptr = &dmin_arr as *const f32;
            let dminv: float32x4_t = vld1q_f32(rawptr);

            let sumi_mins_arr: [i32; 4] = [sumi_mins_00, sumi_mins_01, sumi_mins_10, sumi_mins_11];
            let rawptr = &sumi_mins_arr as *const i32;
            let sumi_minsv: float32x4_t = vcvtq_f32_s32(vld1q_s32(rawptr));

            let fsum = vcvtq_f32_s32(isum);
            sumfv = vmlaq_f32(sumfv, fsum, factor);
            sumfv = vmlaq_f32(sumfv, dminv, sumi_minsv);
        }
        // extract elements of the vector register
        let f0 = vgetq_lane_f32(sumfv, 0);
        let f1 = vgetq_lane_f32(sumfv, 1);
        let f2 = vgetq_lane_f32(sumfv, 2);
        let f3 = vgetq_lane_f32(sumfv, 3);
        let res: [f32; 4] = [f0, f1, f2, f3];
        Ok(res)
    }
}

#[inline(always)]
pub(crate) fn vec_dot_q4k_q8k(n: usize, xs: &[BlockQ4K], ys: &[BlockQ8K]) -> Result<f32> {
    if n % QK_K != 0 {
        crate::bail!("vec_dot_q4k_q8k: {n} is not divisible by {QK_K}")
    }
    let mut sumf = 0f32;
    let mut utmp = [0u32; 4];
    let mut scales = [0u8; 16];
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    unsafe {
        let m4b = vdupq_n_u8(0xF);
        let mzero = vdupq_n_s32(0);

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
                    utmp[1] & KMASK1,
                    ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4),
                ]
                .as_ptr(),
            );
            utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
            utmp[0] &= KMASK1;

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
                let q8bytes = vld1q_s8_x2(q8);
                q8 = q8.add(32);
                let q4bytes = int8x16x2_t(
                    vreinterpretq_s8_u8(vandq_u8(q4bits.0, m4b)),
                    vreinterpretq_s8_u8(vandq_u8(q4bits.1, m4b)),
                );
                let p0 = vdotq_s32_local(mzero, q4bytes.0, q8bytes.0);
                let p1 = vdotq_s32_local(mzero, q4bytes.1, q8bytes.1);
                sumi1 += vaddvq_s32(vaddq_s32(p0, p1)) * scales[2 * j] as i32;

                let q8bytes = vld1q_s8_x2(q8);
                q8 = q8.add(32);
                let q4bytes = int8x16x2_t(
                    vreinterpretq_s8_u8(vshrq_n_u8(q4bits.0, 4)),
                    vreinterpretq_s8_u8(vshrq_n_u8(q4bits.1, 4)),
                );
                let p2 = vdotq_s32_local(mzero, q4bytes.0, q8bytes.0);
                let p3 = vdotq_s32_local(mzero, q4bytes.1, q8bytes.1);
                sumi2 += vaddvq_s32(vaddq_s32(p2, p3)) * scales[2 * j + 1] as i32;
            }
            sumf += d * (sumi1 + sumi2) as f32;
        }
    }
    Ok(sumf)
}

#[inline(always)]
#[cfg(feature = "arm-nightly-feat")]
pub(crate) fn i8mm_q4k_q8k(
    n: usize,
    xs_0: &[BlockQ4K],
    xs_1: &[BlockQ4K],
    ys_0: &[BlockQ8K],
    ys_1: &[BlockQ8K],
) -> Result<[f32; 4]> {
    if n % QK_K != 0 {
        crate::bail!("i8mm_q4k_q8k: {n} is not divisible by {QK_K}")
    }
    let mut utmp_0 = [0u32; 4];
    let mut utmp_1 = [0u32; 4];
    let mut scales_0 = [0u8; 16];
    let mut scales_1 = [0u8; 16];
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    unsafe {
        let mut sumfv = vdupq_n_f32(0.0);
        let m4b = vdupq_n_u8(0xF);

        for (x0, x1, y0, y1) in izip!(xs_0, xs_1, ys_0, ys_1) {
            let d_00: f32 = x0.d.to_f32() * y0.d;
            let d_01: f32 = x1.d.to_f32() * y0.d;
            let d_10: f32 = x0.d.to_f32() * y1.d;
            let d_11: f32 = x1.d.to_f32() * y1.d;

            let dmin_00 = x0.dmin.to_f32() * y0.d;
            let dmin_01 = x1.dmin.to_f32() * y0.d;
            let dmin_10 = x0.dmin.to_f32() * y1.d;
            let dmin_11 = x1.dmin.to_f32() * y1.d;

            let q8sums_0 = vpaddq_s16(
                vld1q_s16(y0.bsums.as_ptr()),
                vld1q_s16(y0.bsums.as_ptr().add(8)),
            );
            let q8sums_1 = vpaddq_s16(
                vld1q_s16(y1.bsums.as_ptr()),
                vld1q_s16(y1.bsums.as_ptr().add(8)),
            );
            LittleEndian::read_u32_into(&x0.scales, &mut utmp_0[0..3]);
            LittleEndian::read_u32_into(&x1.scales, &mut utmp_1[0..3]);

            let mins8_0 = vld1_u32(
                [
                    utmp_0[1] & KMASK1,
                    ((utmp_0[2] >> 4) & KMASK2) | (((utmp_0[1] >> 6) & KMASK3) << 4),
                ]
                .as_ptr(),
            );
            let mins8_1 = vld1_u32(
                [
                    utmp_1[1] & KMASK1,
                    ((utmp_1[2] >> 4) & KMASK2) | (((utmp_1[1] >> 6) & KMASK3) << 4),
                ]
                .as_ptr(),
            );
            utmp_0[1] = (utmp_0[2] & KMASK2) | (((utmp_0[0] >> 6) & KMASK3) << 4);
            utmp_0[0] &= KMASK1;

            utmp_1[1] = (utmp_1[2] & KMASK2) | (((utmp_1[0] >> 6) & KMASK3) << 4);
            utmp_1[0] &= KMASK1;

            let mins_0 = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8_0))); // from x0
            let mins_1 = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8_1))); // from x1

            // y0 x0
            let prod_00 = vaddq_s32(
                vmull_s16(vget_low_s16(q8sums_0), vget_low_s16(mins_0)),
                vmull_s16(vget_high_s16(q8sums_0), vget_high_s16(mins_0)),
            );
            // y0 x1
            let prod_01 = vaddq_s32(
                vmull_s16(vget_low_s16(q8sums_0), vget_low_s16(mins_1)),
                vmull_s16(vget_high_s16(q8sums_0), vget_high_s16(mins_1)),
            );
            // y1 x0
            let prod_10 = vaddq_s32(
                vmull_s16(vget_low_s16(q8sums_1), vget_low_s16(mins_0)),
                vmull_s16(vget_high_s16(q8sums_1), vget_high_s16(mins_0)),
            );
            // y1 x1
            let prod_11 = vaddq_s32(
                vmull_s16(vget_low_s16(q8sums_1), vget_low_s16(mins_1)),
                vmull_s16(vget_high_s16(q8sums_1), vget_high_s16(mins_1)),
            );

            let s = [
                -dmin_00 * vaddvq_s32(prod_00) as f32,
                -dmin_01 * vaddvq_s32(prod_01) as f32,
                -dmin_10 * vaddvq_s32(prod_10) as f32,
                -dmin_11 * vaddvq_s32(prod_11) as f32,
            ];
            let rawptr = &s as *const f32;
            let sumdiff: float32x4_t = vld1q_f32(rawptr);
            sumfv = vaddq_f32(sumfv, sumdiff);

            LittleEndian::write_u32_into(&utmp_0, &mut scales_0);
            LittleEndian::write_u32_into(&utmp_1, &mut scales_1);

            let mut q4_0 = x0.qs.as_ptr();
            let mut q4_1 = x1.qs.as_ptr();
            let mut q8_0 = y0.qs.as_ptr();
            let mut q8_1 = y1.qs.as_ptr();

            let mut sumi1 = vdupq_n_s32(0);
            let mut sumi2 = vdupq_n_s32(0);
            // 0..4
            for j in 0..QK_K / 64 {
                let xv0 = vld1q_u8_x2(q4_0);
                let xv0_0_original = xv0.0;
                let xv0_1_original = xv0.1;
                q4_0 = q4_0.add(32);

                let xv1 = vld1q_u8_x2(q4_1);
                let xv1_0_original = xv1.0;
                let xv1_1_original = xv1.1;
                q4_1 = q4_1.add(32);

                let yv0 = vld1q_s8_x2(q8_0);
                let yv0_0 = yv0.0;
                let yv0_1 = yv0.1;
                q8_0 = q8_0.add(32);

                let yv1 = vld1q_s8_x2(q8_1);
                let yv1_0 = yv1.0;
                let yv1_1 = yv1.1;
                q8_1 = q8_1.add(32);

                let xv0_0 = vreinterpretq_s8_u8(vandq_u8(xv0_0_original, m4b));
                let xv0_1 = vreinterpretq_s8_u8(vandq_u8(xv0_1_original, m4b));
                let xv1_0 = vreinterpretq_s8_u8(vandq_u8(xv1_0_original, m4b));
                let xv1_1 = vreinterpretq_s8_u8(vandq_u8(xv1_1_original, m4b));

                let i8mm = i8mm_params::new(xv0_0, xv0_1, xv1_0, xv1_1, yv0_0, yv0_1, yv1_0, yv1_1);
                let p1 = i8mm.calculate(vdupq_n_s32(0));

                //       x0 | x1
                // y0 | sc_0 sc_1
                // y1 | sc_0 sc_1
                let scarr = [
                    scales_0[2 * j] as i32,
                    scales_1[2 * j] as i32,
                    scales_0[2 * j] as i32,
                    scales_1[2 * j] as i32,
                ];
                let rawptr = &scarr as *const i32;
                let sc: int32x4_t = vld1q_s32(rawptr);
                sumi1 = vmlaq_s32(sumi1, p1, sc);

                let yv0 = vld1q_s8_x2(q8_0);
                let yv0_0 = yv0.0;
                let yv0_1 = yv0.1;
                q8_0 = q8_0.add(32);
                let yv1 = vld1q_s8_x2(q8_1);
                let yv1_0 = yv1.0;
                let yv1_1 = yv1.1;
                q8_1 = q8_1.add(32);

                let xv0_0 = vreinterpretq_s8_u8(vshrq_n_u8(xv0_0_original, 4));
                let xv0_1 = vreinterpretq_s8_u8(vshrq_n_u8(xv0_1_original, 4));
                let xv1_0 = vreinterpretq_s8_u8(vshrq_n_u8(xv1_0_original, 4));
                let xv1_1 = vreinterpretq_s8_u8(vshrq_n_u8(xv1_1_original, 4));

                let i8mm = i8mm_params::new(xv0_0, xv0_1, xv1_0, xv1_1, yv0_0, yv0_1, yv1_0, yv1_1);
                let p2 = i8mm.calculate(vdupq_n_s32(0));
                let sc_arr = [
                    scales_0[2 * j + 1] as i32,
                    scales_1[2 * j + 1] as i32,
                    scales_0[2 * j + 1] as i32,
                    scales_1[2 * j + 1] as i32,
                ];
                let rawptr = &sc_arr as *const i32;
                let sc: int32x4_t = vld1q_s32(rawptr);
                sumi2 = vmlaq_s32(sumi2, p2, sc);
            }
            let factor_elems: [f32; 4] = [d_00, d_01, d_10, d_11];
            let rawptr = &factor_elems as *const f32;
            let factor: float32x4_t = vld1q_f32(rawptr);

            let loop_sum_f32 = vcvtq_f32_s32(vaddq_s32(sumi1, sumi2));
            sumfv = vmlaq_f32(sumfv, loop_sum_f32, factor);
        }
        // extract elements of the vector register
        let f0 = vgetq_lane_f32(sumfv, 0);
        let f1 = vgetq_lane_f32(sumfv, 1);
        let f2 = vgetq_lane_f32(sumfv, 2);
        let f3 = vgetq_lane_f32(sumfv, 3);
        let res: [f32; 4] = [f0, f1, f2, f3];
        Ok(res)
    }
}

#[inline(always)]
pub(crate) fn vec_dot_q3k_q8k(n: usize, xs: &[BlockQ3K], ys: &[BlockQ8K]) -> Result<f32> {
    if n % QK_K != 0 {
        crate::bail!("vec_dot_q3k_q8k: {n} is not divisible by {QK_K}")
    }
    let mut sumf = 0f32;
    let mut utmp = [0u32; 4];
    let mut aux = [0u32; 3];
    const KMASK1: u32 = 0x03030303;
    const KMASK2: u32 = 0x0f0f0f0f;

    unsafe {
        let m3b = vdupq_n_u8(0x3);
        let mzero = vdupq_n_s32(0);
        let m0 = vdupq_n_u8(1);
        let m1 = vshlq_n_u8(m0, 1);
        let m2 = vshlq_n_u8(m0, 2);
        let m3 = vshlq_n_u8(m0, 3);
        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = y.d * x.d.to_f32();
            let mut q3 = x.qs.as_ptr();
            let qh = x.hmask.as_ptr();
            let mut q8 = y.qs.as_ptr();

            let mut qhbits = vld1q_u8_x2(qh);

            let mut isum = 0i32;

            // Set up scales
            LittleEndian::read_u32_into(&x.scales, &mut aux);

            utmp[3] = ((aux[1] >> 4) & KMASK2) | (((aux[2] >> 6) & KMASK1) << 4);
            utmp[2] = ((aux[0] >> 4) & KMASK2) | (((aux[2] >> 4) & KMASK1) << 4);
            utmp[1] = (aux[1] & KMASK2) | (((aux[2] >> 2) & KMASK1) << 4);
            utmp[0] = (aux[0] & KMASK2) | ((aux[2] & KMASK1) << 4);

            let mut scale = utmp.as_mut_ptr() as *mut i8;
            for j in 0..16 {
                *scale.add(j) -= 32i8
            }

            for j in 0..QK_K / 128 {
                let q3bits = vld1q_u8_x2(q3);
                q3 = q3.add(32);
                let q8bytes_1 = vld1q_s8_x4(q8);
                q8 = q8.add(64);
                let q8bytes_2 = vld1q_s8_x4(q8);
                q8 = q8.add(64);

                let q3h_0 = vshlq_n_u8(vbicq_u8(m0, qhbits.0), 2);
                let q3h_1 = vshlq_n_u8(vbicq_u8(m0, qhbits.1), 2);
                let q3h_2 = vshlq_n_u8(vbicq_u8(m1, qhbits.0), 1);
                let q3h_3 = vshlq_n_u8(vbicq_u8(m1, qhbits.1), 1);

                let q3bytes_0 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(q3bits.0, m3b)),
                    vreinterpretq_s8_u8(q3h_0),
                );
                let q3bytes_1 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(q3bits.1, m3b)),
                    vreinterpretq_s8_u8(q3h_1),
                );
                let q3bytes_2 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.0, 2), m3b)),
                    vreinterpretq_s8_u8(q3h_2),
                );
                let q3bytes_3 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.1, 2), m3b)),
                    vreinterpretq_s8_u8(q3h_3),
                );

                let p0 = vdotq_s32_local(mzero, q3bytes_0, q8bytes_1.0);
                let p1 = vdotq_s32_local(mzero, q3bytes_1, q8bytes_1.1);
                let p2 = vdotq_s32_local(mzero, q3bytes_2, q8bytes_1.2);
                let p3 = vdotq_s32_local(mzero, q3bytes_3, q8bytes_1.3);
                isum += vaddvq_s32(p0) * *scale as i32
                    + vaddvq_s32(p1) * *scale.add(1) as i32
                    + vaddvq_s32(p2) * *scale.add(2) as i32
                    + vaddvq_s32(p3) * *scale.add(3) as i32;
                scale = scale.add(4);

                let q3h_0 = vbicq_u8(m2, qhbits.0);
                let q3h_1 = vbicq_u8(m2, qhbits.1);
                let q3h_2 = vshrq_n_u8(vbicq_u8(m3, qhbits.0), 1);
                let q3h_3 = vshrq_n_u8(vbicq_u8(m3, qhbits.1), 1);

                let q3bytes_0 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.0, 4), m3b)),
                    vreinterpretq_s8_u8(q3h_0),
                );
                let q3bytes_1 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.1, 4), m3b)),
                    vreinterpretq_s8_u8(q3h_1),
                );
                let q3bytes_2 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.0, 6), m3b)),
                    vreinterpretq_s8_u8(q3h_2),
                );
                let q3bytes_3 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.1, 6), m3b)),
                    vreinterpretq_s8_u8(q3h_3),
                );

                let p0 = vdotq_s32_local(mzero, q3bytes_0, q8bytes_2.0);
                let p1 = vdotq_s32_local(mzero, q3bytes_1, q8bytes_2.1);
                let p2 = vdotq_s32_local(mzero, q3bytes_2, q8bytes_2.2);
                let p3 = vdotq_s32_local(mzero, q3bytes_3, q8bytes_2.3);
                isum += vaddvq_s32(p0) * *scale as i32
                    + vaddvq_s32(p1) * *scale.add(1) as i32
                    + vaddvq_s32(p2) * *scale.add(2) as i32
                    + vaddvq_s32(p3) * *scale.add(3) as i32;
                scale = scale.add(4);

                if j == 0 {
                    qhbits.0 = vshrq_n_u8(qhbits.0, 4);
                    qhbits.1 = vshrq_n_u8(qhbits.1, 4);
                }
            }
            sumf += d * isum as f32;
        }
    }
    Ok(sumf)
}

#[inline(always)]
#[cfg(feature = "arm-nightly-feat")]
pub(crate) fn i8mm_q3k_q8k(
    n: usize,
    xs_0: &[BlockQ3K],
    xs_1: &[BlockQ3K],
    ys_0: &[BlockQ8K],
    ys_1: &[BlockQ8K],
) -> Result<[f32; 4]> {
    if n % QK_K != 0 {
        crate::bail!("i8mm_q3k_q8k: {n} is not divisible by {QK_K}")
    }
    unsafe {
        let mut sumfv = vdupq_n_f32(0.0);
        let mut utmp_0 = [0u32; 4];
        let mut utmp_1 = [0u32; 4];
        let mut aux_0 = [0u32; 3];
        let mut aux_1 = [0u32; 3];
        const KMASK1: u32 = 0x03030303;
        const KMASK2: u32 = 0x0f0f0f0f;

        let m3b = vdupq_n_u8(0x3);
        let m0 = vdupq_n_u8(1);
        let m1 = vshlq_n_u8(m0, 1);
        let m2 = vshlq_n_u8(m0, 2);
        let m3 = vshlq_n_u8(m0, 3);

        for (x0, x1, y0, y1) in izip!(xs_0, xs_1, ys_0, ys_1) {
            let d_00: f32 = x0.d.to_f32() * y0.d;
            let d_01: f32 = x1.d.to_f32() * y0.d;
            let d_10: f32 = x0.d.to_f32() * y1.d;
            let d_11: f32 = x1.d.to_f32() * y1.d;

            let mut q3_0 = x0.qs.as_ptr();
            let mut q3_1 = x1.qs.as_ptr();

            let qh_0 = x0.hmask.as_ptr();
            let qh_1 = x1.hmask.as_ptr();

            let mut q8_0 = y0.qs.as_ptr();
            let mut q8_1 = y1.qs.as_ptr();

            let mut qhbits_0 = vld1q_u8_x2(qh_0);
            let mut qhbits_1 = vld1q_u8_x2(qh_1);

            let mut isum = vdupq_n_s32(0);

            // Set up scales
            LittleEndian::read_u32_into(&x0.scales, &mut aux_0);
            LittleEndian::read_u32_into(&x1.scales, &mut aux_1);

            utmp_0[3] = ((aux_0[1] >> 4) & KMASK2) | (((aux_0[2] >> 6) & KMASK1) << 4);
            utmp_0[2] = ((aux_0[0] >> 4) & KMASK2) | (((aux_0[2] >> 4) & KMASK1) << 4);
            utmp_0[1] = (aux_0[1] & KMASK2) | (((aux_0[2] >> 2) & KMASK1) << 4);
            utmp_0[0] = (aux_0[0] & KMASK2) | ((aux_0[2] & KMASK1) << 4);

            utmp_1[3] = ((aux_1[1] >> 4) & KMASK2) | (((aux_1[2] >> 6) & KMASK1) << 4);
            utmp_1[2] = ((aux_1[0] >> 4) & KMASK2) | (((aux_1[2] >> 4) & KMASK1) << 4);
            utmp_1[1] = (aux_1[1] & KMASK2) | (((aux_1[2] >> 2) & KMASK1) << 4);
            utmp_1[0] = (aux_1[0] & KMASK2) | ((aux_1[2] & KMASK1) << 4);

            let mut scale_0 = utmp_0.as_mut_ptr() as *mut i8;
            for j in 0..16 {
                *scale_0.add(j) -= 32i8
            }
            let mut scale_1 = utmp_1.as_mut_ptr() as *mut i8;
            for j in 0..16 {
                *scale_1.add(j) -= 32i8
            }
            for j in 0..QK_K / 128 {
                let q3bits_0 = vld1q_u8_x2(q3_0);
                let q3bits_1 = vld1q_u8_x2(q3_1);
                q3_0 = q3_0.add(32);
                q3_1 = q3_1.add(32);

                // "y0"
                let q8bytes0_1 = vld1q_s8_x4(q8_0);
                q8_0 = q8_0.add(64);
                let q8bytes0_2 = vld1q_s8_x4(q8_0);
                q8_0 = q8_0.add(64);

                // "y1"
                let q8bytes1_1 = vld1q_s8_x4(q8_1);
                q8_1 = q8_1.add(64);
                let q8bytes1_2 = vld1q_s8_x4(q8_1);
                q8_1 = q8_1.add(64);

                // "x0"
                let q3h_0_0 = vshlq_n_u8(vbicq_u8(m0, qhbits_0.0), 2);
                let q3h_0_1 = vshlq_n_u8(vbicq_u8(m0, qhbits_0.1), 2);
                let q3h_0_2 = vshlq_n_u8(vbicq_u8(m1, qhbits_0.0), 1);
                let q3h_0_3 = vshlq_n_u8(vbicq_u8(m1, qhbits_0.1), 1);

                // "x1"
                let q3h_1_0 = vshlq_n_u8(vbicq_u8(m0, qhbits_1.0), 2);
                let q3h_1_1 = vshlq_n_u8(vbicq_u8(m0, qhbits_1.1), 2);
                let q3h_1_2 = vshlq_n_u8(vbicq_u8(m1, qhbits_1.0), 1);
                let q3h_1_3 = vshlq_n_u8(vbicq_u8(m1, qhbits_1.1), 1);

                // "x0"
                let q3bytes_0_0 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(q3bits_0.0, m3b)),
                    vreinterpretq_s8_u8(q3h_0_0),
                );
                let q3bytes_0_1 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(q3bits_0.1, m3b)),
                    vreinterpretq_s8_u8(q3h_0_1),
                );
                let q3bytes_0_2 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits_0.0, 2), m3b)),
                    vreinterpretq_s8_u8(q3h_0_2),
                );
                let q3bytes_0_3 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits_0.1, 2), m3b)),
                    vreinterpretq_s8_u8(q3h_0_3),
                );
                // "x1"
                let q3bytes_1_0 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(q3bits_1.0, m3b)),
                    vreinterpretq_s8_u8(q3h_1_0),
                );
                let q3bytes_1_1 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(q3bits_1.1, m3b)),
                    vreinterpretq_s8_u8(q3h_1_1),
                );
                let q3bytes_1_2 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits_1.0, 2), m3b)),
                    vreinterpretq_s8_u8(q3h_1_2),
                );
                let q3bytes_1_3 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits_1.1, 2), m3b)),
                    vreinterpretq_s8_u8(q3h_1_3),
                );

                /* 4 x0s, 4 x1s
                 * 4 y0s, 4 y1s
                 * 1 step of i8mm needs 2 of each
                 *  -> 2 sets of i8mm calcs are needed
                 */
                let sc = i8mm_x_scales::new(&x_scales {
                    x00: *scale_0.add(0) as i32,
                    x01: *scale_0.add(1) as i32,
                    x10: *scale_1.add(0) as i32,
                    x11: *scale_1.add(1) as i32,
                });
                let i8mm = i8mm_params::new(
                    q3bytes_0_0,
                    q3bytes_0_1,
                    q3bytes_1_0,
                    q3bytes_1_1,
                    q8bytes0_1.0,
                    q8bytes0_1.1,
                    q8bytes1_1.0,
                    q8bytes1_1.1,
                );
                isum = i8mm.calculate_with_scales(isum, sc);

                let sc = i8mm_x_scales::new(&x_scales {
                    x00: *scale_0.add(2) as i32,
                    x01: *scale_0.add(3) as i32,
                    x10: *scale_1.add(2) as i32,
                    x11: *scale_1.add(3) as i32,
                });
                let i8mm = i8mm_params::new(
                    q3bytes_0_2,
                    q3bytes_0_3,
                    q3bytes_1_2,
                    q3bytes_1_3,
                    q8bytes0_1.2,
                    q8bytes0_1.3,
                    q8bytes1_1.2,
                    q8bytes1_1.3,
                );
                isum = i8mm.calculate_with_scales(isum, sc);

                scale_0 = scale_0.add(4);
                scale_1 = scale_1.add(4);

                let q3h_0_0 = vbicq_u8(m2, qhbits_0.0);
                let q3h_0_1 = vbicq_u8(m2, qhbits_0.1);

                let q3h_0_3 = vshrq_n_u8(vbicq_u8(m3, qhbits_0.1), 1);

                let q3h_1_0 = vbicq_u8(m2, qhbits_1.0);
                let q3h_1_1 = vbicq_u8(m2, qhbits_1.1);
                let q3h_1_2 = vshrq_n_u8(vbicq_u8(m3, qhbits_1.0), 1);
                let q3h_1_3 = vshrq_n_u8(vbicq_u8(m3, qhbits_1.1), 1);

                let q3bytes_0_0 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits_0.0, 4), m3b)),
                    vreinterpretq_s8_u8(q3h_0_0),
                );
                let q3bytes_0_1 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits_0.1, 4), m3b)),
                    vreinterpretq_s8_u8(q3h_0_1),
                );
                let q3bytes_0_2 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits_0.0, 6), m3b)),
                    vreinterpretq_s8_u8(q3h_0_2),
                );
                let q3bytes_0_3 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits_0.1, 6), m3b)),
                    vreinterpretq_s8_u8(q3h_0_3),
                );

                let q3bytes_1_0 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits_1.0, 4), m3b)),
                    vreinterpretq_s8_u8(q3h_1_0),
                );
                let q3bytes_1_1 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits_1.1, 4), m3b)),
                    vreinterpretq_s8_u8(q3h_1_1),
                );
                let q3bytes_1_2 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits_1.0, 6), m3b)),
                    vreinterpretq_s8_u8(q3h_1_2),
                );
                let q3bytes_1_3 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits_1.1, 6), m3b)),
                    vreinterpretq_s8_u8(q3h_1_3),
                );

                // Same as above
                let sc = i8mm_x_scales::new(&x_scales {
                    x00: *scale_0.add(0) as i32,
                    x01: *scale_0.add(1) as i32,
                    x10: *scale_1.add(0) as i32,
                    x11: *scale_1.add(1) as i32,
                });
                let i8mm = i8mm_params::new(
                    q3bytes_0_0,
                    q3bytes_0_1,
                    q3bytes_1_0,
                    q3bytes_1_1,
                    q8bytes0_2.0,
                    q8bytes0_2.1,
                    q8bytes1_2.0,
                    q8bytes1_2.1,
                );
                isum = i8mm.calculate_with_scales(isum, sc);

                let sc = i8mm_x_scales::new(&x_scales {
                    x00: *scale_0.add(2) as i32,
                    x01: *scale_0.add(3) as i32,
                    x10: *scale_1.add(2) as i32,
                    x11: *scale_1.add(3) as i32,
                });
                let i8mm = i8mm_params::new(
                    q3bytes_0_2,
                    q3bytes_0_3,
                    q3bytes_1_2,
                    q3bytes_1_3,
                    q8bytes0_2.2,
                    q8bytes0_2.3,
                    q8bytes1_2.2,
                    q8bytes1_2.3,
                );
                isum = i8mm.calculate_with_scales(isum, sc);

                scale_0 = scale_0.add(4);
                scale_1 = scale_1.add(4);

                if j == 0 {
                    qhbits_0.0 = vshrq_n_u8(qhbits_0.0, 4);
                    qhbits_0.1 = vshrq_n_u8(qhbits_0.1, 4);
                    qhbits_1.0 = vshrq_n_u8(qhbits_1.0, 4);
                    qhbits_1.1 = vshrq_n_u8(qhbits_1.1, 4);
                }
            }
            let factor_elems: [f32; 4] = [d_00, d_01, d_10, d_11];
            let rawptr = &factor_elems as *const f32;
            let factor: float32x4_t = vld1q_f32(rawptr);

            let fsum = vcvtq_f32_s32(isum);
            sumfv = vmlaq_f32(sumfv, fsum, factor);
        }
        // extract elements of the vector register
        let f0 = vgetq_lane_f32(sumfv, 0);
        let f1 = vgetq_lane_f32(sumfv, 1);
        let f2 = vgetq_lane_f32(sumfv, 2);
        let f3 = vgetq_lane_f32(sumfv, 3);
        let res: [f32; 4] = [f0, f1, f2, f3];
        Ok(res)
    }
}

#[inline(always)]
pub(crate) fn vec_dot_q2k_q8k(n: usize, xs: &[BlockQ2K], ys: &[BlockQ8K]) -> Result<f32> {
    if n % QK_K != 0 {
        crate::bail!("vec_dot_q2k_q8k: {n} is not divisible by {QK_K}")
    }
    let mut sumf = 0f32;
    let mut aux = [0u8; 16];

    unsafe {
        let m3 = vdupq_n_u8(0x3);
        let m4 = vdupq_n_u8(0xF);

        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = y.d * x.d.to_f32();
            let dmin = -y.d * x.dmin.to_f32();

            let mut q2 = x.qs.as_ptr();
            let mut q8 = y.qs.as_ptr();
            let sc = x.scales.as_ptr();

            let mins_and_scales = vld1q_u8(sc);
            let scales = vandq_u8(mins_and_scales, m4);
            vst1q_u8(aux.as_mut_ptr(), scales);

            let mins = vshrq_n_u8(mins_and_scales, 4);
            let q8sums = vld1q_s16_x2(y.bsums.as_ptr());
            let mins16 = int16x8x2_t(
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins))),
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins))),
            );
            let s0 = vaddq_s32(
                vmull_s16(vget_low_s16(mins16.0), vget_low_s16(q8sums.0)),
                vmull_s16(vget_high_s16(mins16.0), vget_high_s16(q8sums.0)),
            );
            let s1 = vaddq_s32(
                vmull_s16(vget_low_s16(mins16.1), vget_low_s16(q8sums.1)),
                vmull_s16(vget_high_s16(mins16.1), vget_high_s16(q8sums.1)),
            );
            sumf += dmin * vaddvq_s32(vaddq_s32(s0, s1)) as f32;

            let mut isum = 0i32;
            let mut is = 0usize;

            for _j in 0..QK_K / 128 {
                let q2bits = vld1q_u8_x2(q2);
                q2 = q2.add(32);

                let q8bytes = vld1q_s8_x2(q8);
                q8 = q8.add(32);
                let mut q2bytes = int8x16x2_t(
                    vreinterpretq_s8_u8(vandq_u8(q2bits.0, m3)),
                    vreinterpretq_s8_u8(vandq_u8(q2bits.1, m3)),
                );
                isum += multiply_accum_with_scale(&aux, is, 0, q2bytes, q8bytes);

                let q8bytes = vld1q_s8_x2(q8);
                q8 = q8.add(32);
                q2bytes.0 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.0, 2), m3));
                q2bytes.1 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.1, 2), m3));
                isum += multiply_accum_with_scale(&aux, is, 2, q2bytes, q8bytes);

                let q8bytes = vld1q_s8_x2(q8);
                q8 = q8.add(32);
                q2bytes.0 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.0, 4), m3));
                q2bytes.1 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.1, 4), m3));
                isum += multiply_accum_with_scale(&aux, is, 4, q2bytes, q8bytes);

                let q8bytes = vld1q_s8_x2(q8);
                q8 = q8.add(32);
                q2bytes.0 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.0, 6), m3));
                q2bytes.1 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.1, 6), m3));
                isum += multiply_accum_with_scale(&aux, is, 6, q2bytes, q8bytes);

                is += 8;
            }
            sumf += d * isum as f32;
        }
    }
    Ok(sumf)
}

#[inline(always)]
#[cfg_attr(
    target_arch = "aarch64-unknown-linux-gnu",
    target_feature(enable = "stdarch_neon_i8mm")
)]
#[cfg(feature = "arm-nightly-feat")]
pub(crate) fn i8mm_q2k_q8k(
    _n: usize,
    xs_0: &[BlockQ2K],
    xs_1: &[BlockQ2K],
    ys_0: &[BlockQ8K],
    ys_1: &[BlockQ8K],
) -> Result<[f32; 4]> {
    let mut aux_0 = [0u8; 16];
    let mut aux_1 = [0u8; 16];

    unsafe {
        let mut sumfv = vdupq_n_f32(0.0);
        let m3 = vdupq_n_u8(0x3);
        let m4 = vdupq_n_u8(0xF);
        for (x0, x1, y0, y1) in izip!(xs_0, xs_1, ys_0, ys_1) {
            let d_00: f32 = x0.d.to_f32() * y0.d;
            let d_01: f32 = x1.d.to_f32() * y0.d;
            let d_10: f32 = x0.d.to_f32() * y1.d;
            let d_11: f32 = x1.d.to_f32() * y1.d;

            let dmin_00 = -y0.d * x0.dmin.to_f32();
            let dmin_01 = -y0.d * x1.dmin.to_f32();
            let dmin_10 = -y1.d * x0.dmin.to_f32();
            let dmin_11 = -y1.d * x1.dmin.to_f32();

            let mut q2_0 = x0.qs.as_ptr();
            let mut q2_1 = x1.qs.as_ptr();
            let mut q8_0 = y0.qs.as_ptr();
            let mut q8_1 = y1.qs.as_ptr();

            let sc_0 = x0.scales.as_ptr();
            let sc_1 = x1.scales.as_ptr();

            let mins_and_scales_0 = vld1q_u8(sc_0);
            let mins_and_scales_1 = vld1q_u8(sc_1);

            let scales_0 = vandq_u8(mins_and_scales_0, m4);
            let scales_1 = vandq_u8(mins_and_scales_1, m4);

            vst1q_u8(aux_0.as_mut_ptr(), scales_0);
            vst1q_u8(aux_1.as_mut_ptr(), scales_1);

            let mins_0 = vshrq_n_u8(mins_and_scales_0, 4);
            let mins_1 = vshrq_n_u8(mins_and_scales_1, 4);

            let q8sums_0 = vld1q_s16_x2(y0.bsums.as_ptr());
            let q8sums_1 = vld1q_s16_x2(y1.bsums.as_ptr());

            let mins16_0 = int16x8x2_t(
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins_0))),
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins_0))),
            );
            let mins16_1 = int16x8x2_t(
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins_1))),
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins_1))),
            );
            // x --> mins16
            // y --> q8sums
            let s00l = vaddq_s32(
                vmull_s16(vget_low_s16(mins16_0.0), vget_low_s16(q8sums_0.0)),
                vmull_s16(vget_high_s16(mins16_0.0), vget_high_s16(q8sums_0.0)),
            );
            let s00h = vaddq_s32(
                vmull_s16(vget_low_s16(mins16_0.1), vget_low_s16(q8sums_0.1)),
                vmull_s16(vget_high_s16(mins16_0.1), vget_high_s16(q8sums_0.1)),
            );

            // 01 -> y0 * x1
            let s01l = vaddq_s32(
                vmull_s16(vget_low_s16(mins16_1.0), vget_low_s16(q8sums_0.0)),
                vmull_s16(vget_high_s16(mins16_1.0), vget_high_s16(q8sums_0.0)),
            );
            let s01h = vaddq_s32(
                vmull_s16(vget_low_s16(mins16_1.1), vget_low_s16(q8sums_0.1)),
                vmull_s16(vget_high_s16(mins16_1.1), vget_high_s16(q8sums_0.1)),
            );

            // 10 -> y1 * x0
            let s10l = vaddq_s32(
                vmull_s16(vget_low_s16(mins16_0.0), vget_low_s16(q8sums_1.0)),
                vmull_s16(vget_high_s16(mins16_0.0), vget_high_s16(q8sums_1.0)),
            );
            let s10h = vaddq_s32(
                vmull_s16(vget_low_s16(mins16_0.1), vget_low_s16(q8sums_1.1)),
                vmull_s16(vget_high_s16(mins16_0.1), vget_high_s16(q8sums_1.1)),
            );

            // 11 -> y1 * x1
            let s11l = vaddq_s32(
                vmull_s16(vget_low_s16(mins16_1.0), vget_low_s16(q8sums_1.0)),
                vmull_s16(vget_high_s16(mins16_1.0), vget_high_s16(q8sums_1.0)),
            );
            let s11h = vaddq_s32(
                vmull_s16(vget_low_s16(mins16_1.1), vget_low_s16(q8sums_1.1)),
                vmull_s16(vget_high_s16(mins16_1.1), vget_high_s16(q8sums_1.1)),
            );

            let sumf_elems: [f32; 4] = [
                dmin_00 * vaddvq_s32(vaddq_s32(s00l, s00h)) as f32,
                dmin_01 * vaddvq_s32(vaddq_s32(s01l, s01h)) as f32,
                dmin_10 * vaddvq_s32(vaddq_s32(s10l, s10h)) as f32,
                dmin_11 * vaddvq_s32(vaddq_s32(s11l, s11h)) as f32,
            ];
            let rawptr = &sumf_elems as *const f32;
            sumfv = vaddq_f32(sumfv, vld1q_f32(rawptr));

            let mut isum = vdupq_n_s32(0i32);
            let mut is = 0usize;

            for _j in 0..QK_K / 128 {
                let q2bits_0 = vld1q_u8_x2(q2_0);
                q2_0 = q2_0.add(32);
                let mut q2bytes_0 = int8x16x2_t(
                    vreinterpretq_s8_u8(vandq_u8(q2bits_0.0, m3)),
                    vreinterpretq_s8_u8(vandq_u8(q2bits_0.1, m3)),
                );
                let q2bits_1 = vld1q_u8_x2(q2_1);
                q2_1 = q2_1.add(32);
                let mut q2bytes_1 = int8x16x2_t(
                    vreinterpretq_s8_u8(vandq_u8(q2bits_1.0, m3)),
                    vreinterpretq_s8_u8(vandq_u8(q2bits_1.1, m3)),
                );

                let q8bytes_0 = vld1q_s8_x2(q8_0);
                q8_0 = q8_0.add(32);
                let q8bytes_1 = vld1q_s8_x2(q8_1);
                q8_1 = q8_1.add(32);
                isum = vaddq_s32(
                    isum,
                    i8mm_accum_with_scale(
                        &aux_0, &aux_1, is, 0, q2bytes_0, q2bytes_1, q8bytes_0, q8bytes_1,
                    ),
                );

                q2bytes_0.0 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits_0.0, 2), m3));
                q2bytes_0.1 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits_0.1, 2), m3));
                q2bytes_1.0 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits_1.0, 2), m3));
                q2bytes_1.1 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits_1.1, 2), m3));
                let q8bytes_0 = vld1q_s8_x2(q8_0);
                q8_0 = q8_0.add(32);
                let q8bytes_1 = vld1q_s8_x2(q8_1);
                q8_1 = q8_1.add(32);
                isum = vaddq_s32(
                    isum,
                    i8mm_accum_with_scale(
                        &aux_0, &aux_1, is, 2, q2bytes_0, q2bytes_1, q8bytes_0, q8bytes_1,
                    ),
                );

                q2bytes_0.0 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits_0.0, 4), m3));
                q2bytes_0.1 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits_0.1, 4), m3));
                q2bytes_1.0 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits_1.0, 4), m3));
                q2bytes_1.1 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits_1.1, 4), m3));
                let q8bytes_0 = vld1q_s8_x2(q8_0);
                q8_0 = q8_0.add(32);
                let q8bytes_1 = vld1q_s8_x2(q8_1);
                q8_1 = q8_1.add(32);
                isum = vaddq_s32(
                    isum,
                    i8mm_accum_with_scale(
                        &aux_0, &aux_1, is, 4, q2bytes_0, q2bytes_1, q8bytes_0, q8bytes_1,
                    ),
                );

                q2bytes_0.0 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits_0.0, 6), m3));
                q2bytes_0.1 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits_0.1, 6), m3));
                q2bytes_1.0 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits_1.0, 6), m3));
                q2bytes_1.1 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits_1.1, 6), m3));
                let q8bytes_0 = vld1q_s8_x2(q8_0);
                q8_0 = q8_0.add(32);
                let q8bytes_1 = vld1q_s8_x2(q8_1);
                q8_1 = q8_1.add(32);
                isum = vaddq_s32(
                    isum,
                    i8mm_accum_with_scale(
                        &aux_0, &aux_1, is, 6, q2bytes_0, q2bytes_1, q8bytes_0, q8bytes_1,
                    ),
                );
                is += 8;
            }

            let factor_elems: [f32; 4] = [d_00, d_01, d_10, d_11];
            let rawptr = &factor_elems as *const f32;
            let factor: float32x4_t = vld1q_f32(rawptr);

            let fsum = vcvtq_f32_s32(isum);
            sumfv = vmlaq_f32(sumfv, fsum, factor);
        }
        // extract elements of the vector register
        let f0 = vgetq_lane_f32(sumfv, 0);
        let f1 = vgetq_lane_f32(sumfv, 1);
        let f2 = vgetq_lane_f32(sumfv, 2);
        let f3 = vgetq_lane_f32(sumfv, 3);
        let res: [f32; 4] = [f0, f1, f2, f3];
        Ok(res)
    }
}

#[inline(always)]
unsafe fn multiply_accum_with_scale(
    aux: &[u8; 16],
    is: usize,
    index: usize,
    q2bytes: int8x16x2_t,
    q8bytes: int8x16x2_t,
) -> i32 {
    let mzero = vdupq_n_s32(0);
    let p1 = vdotq_s32_local(mzero, q2bytes.0, q8bytes.0);
    let p2 = vdotq_s32_local(mzero, q2bytes.1, q8bytes.1);
    vaddvq_s32(p1) * aux[is + index] as i32 + vaddvq_s32(p2) * aux[is + 1 + index] as i32
}

#[inline(always)]
#[cfg_attr(
    target_arch = "aarch64-unknown-linux-gnu",
    target_feature(enable = "stdarch_neon_i8mm")
)]
#[cfg(feature = "arm-nightly-feat")]
unsafe fn i8mm_accum_with_scale(
    aux_0: &[u8; 16],
    aux_1: &[u8; 16],
    is: usize,
    index: usize,
    q2bytes_0: int8x16x2_t,
    q2bytes_1: int8x16x2_t,
    q8bytes_0: int8x16x2_t,
    q8bytes_1: int8x16x2_t,
) -> int32x4_t {
    let mzero = vdupq_n_s32(0);

    let c00 = aux_0[is + index] as i32;
    let c01 = aux_0[is + index + 1] as i32;
    let c10 = aux_1[is + index] as i32;
    let c11 = aux_1[is + index + 1] as i32;

    let x00 = q2bytes_0.0;
    let x01 = q2bytes_0.1;
    let x10 = q2bytes_1.0;
    let x11 = q2bytes_1.1;

    let y00 = q8bytes_0.0;
    let y01 = q8bytes_0.1;
    let y10 = q8bytes_1.0;
    let y11 = q8bytes_1.1;

    let x_sc = x_scales {
        x00: c00,
        x01: c01,
        x10: c10,
        x11: c11,
    };
    let i8mm_sc = i8mm_x_scales::new(&x_sc);
    let mm = i8mm_params::new(x00, x01, x10, x11, y00, y01, y10, y11);
    mm.calculate_with_scales(mzero, i8mm_sc)
}
#[allow(non_camel_case_types)]
#[cfg(feature = "arm-nightly-feat")]
struct i8mm_params {
    x0: int8x16_t,
    x1: int8x16_t,
    x2: int8x16_t,
    x3: int8x16_t,
    y0: int8x16_t,
    y1: int8x16_t,
    y2: int8x16_t,
    y3: int8x16_t,
}

#[allow(non_camel_case_types)]
#[cfg(feature = "arm-nightly-feat")]
/// scales from scalar version
struct x_scales {
    x00: i32,
    x01: i32,
    x10: i32,
    x11: i32,
}
#[allow(non_camel_case_types)]
#[cfg(feature = "arm-nightly-feat")]
/// scales reorganized to fit i8mm calculations
struct i8mm_x_scales {
    sc0: int32x4_t,
    sc1: int32x4_t,
}

#[cfg(feature = "arm-nightly-feat")]
impl i8mm_x_scales {
    #[inline(always)]
    unsafe fn new(sc: &x_scales) -> Self {
        let v00 = vdupq_n_s32(sc.x00);
        let v01 = vdupq_n_s32(sc.x01);
        let v10 = vdupq_n_s32(sc.x10);
        let v11 = vdupq_n_s32(sc.x11);

        let sc0 = vzip1q_s32(v00, v10);
        let sc1 = vzip1q_s32(v01, v11);

        i8mm_x_scales { sc0, sc1 }
    }
}

#[cfg(feature = "arm-nightly-feat")]
impl i8mm_params {
    #[inline(always)]
    unsafe fn new(
        xv0_0: int8x16_t,
        xv0_1: int8x16_t,
        xv1_0: int8x16_t,
        xv1_1: int8x16_t,
        yv0_0: int8x16_t,
        yv0_1: int8x16_t,
        yv1_0: int8x16_t,
        yv1_1: int8x16_t,
    ) -> Self {
        // 1. 16xi8 -> 2xi64
        let xv0_0 = vreinterpretq_s64_s8(xv0_0);
        let xv0_1 = vreinterpretq_s64_s8(xv0_1);
        let xv1_0 = vreinterpretq_s64_s8(xv1_0);
        let xv1_1 = vreinterpretq_s64_s8(xv1_1);

        let yv0_0 = vreinterpretq_s64_s8(yv0_0);
        let yv0_1 = vreinterpretq_s64_s8(yv0_1);
        let yv1_0 = vreinterpretq_s64_s8(yv1_0);
        let yv1_1 = vreinterpretq_s64_s8(yv1_1);

        // 2. ZIP
        let x0_0 = vzip1q_s64(xv0_0, xv1_0);
        let x0_1 = vzip2q_s64(xv0_0, xv1_0);
        let x1_0 = vzip1q_s64(xv0_1, xv1_1);
        let x1_1 = vzip2q_s64(xv0_1, xv1_1);

        let y0_0 = vzip1q_s64(yv0_0, yv1_0);
        let y0_1 = vzip2q_s64(yv0_0, yv1_0);
        let y1_0 = vzip1q_s64(yv0_1, yv1_1);
        let y1_1 = vzip2q_s64(yv0_1, yv1_1);

        // 3. interpret back
        let x0_0 = vreinterpretq_s8_s64(x0_0);
        let x0_1 = vreinterpretq_s8_s64(x0_1);
        let x1_0 = vreinterpretq_s8_s64(x1_0);
        let x1_1 = vreinterpretq_s8_s64(x1_1);

        let y0_0 = vreinterpretq_s8_s64(y0_0);
        let y0_1 = vreinterpretq_s8_s64(y0_1);
        let y1_0 = vreinterpretq_s8_s64(y1_0);
        let y1_1 = vreinterpretq_s8_s64(y1_1);

        i8mm_params {
            x0: x0_0,
            x1: x0_1,
            x2: x1_0,
            x3: x1_1,
            y0: y0_0,
            y1: y0_1,
            y2: y1_0,
            y3: y1_1,
        }
    }

    #[inline(always)]
    unsafe fn calculate(&self, acc: int32x4_t) -> int32x4_t {
        if is_aarch64_feature_detected!("i8mm") {
            self.impl_calc(acc)
        } else {
            // never takes this branch, but the check is needed
            // for inlining the vmmlaq intrinsics
            // see:
            // https://community.arm.com/arm-community-blogs/b/architectures-and-processors-blog/posts/rust-neon-intrinsics
            unreachable!();
        }
    }
    unsafe fn impl_calc(&self, acc: int32x4_t) -> int32x4_t {
        let mut a = acc;
        a = vmmlaq_s32(a, self.y0, self.x0);
        a = vmmlaq_s32(a, self.y1, self.x1);
        a = vmmlaq_s32(a, self.y2, self.x2);
        vmmlaq_s32(a, self.y3, self.x3)
    }

    unsafe fn calculate_with_scales(&self, acc: int32x4_t, scales: i8mm_x_scales) -> int32x4_t {
        if is_aarch64_feature_detected!("i8mm") {
            self.impl_calc_scales(acc, scales)
        } else {
            // never takes this branch, but the check is needed
            // for inlining the vmmlaq intrinsics
            // see:
            // https://community.arm.com/arm-community-blogs/b/architectures-and-processors-blog/posts/rust-neon-intrinsics
            unreachable!();
        }
    }
    #[inline(always)]
    unsafe fn impl_calc_scales(&self, acc: int32x4_t, scales: i8mm_x_scales) -> int32x4_t {
        let mzero = vdupq_n_s32(0);
        let a = vmulq_s32(vmmlaq_s32(mzero, self.y0, self.x0), scales.sc0);
        let b = vmulq_s32(vmmlaq_s32(mzero, self.y1, self.x1), scales.sc0);
        let c = vmulq_s32(vmmlaq_s32(mzero, self.y2, self.x2), scales.sc1);
        let d = vmulq_s32(vmmlaq_s32(mzero, self.y3, self.x3), scales.sc1);

        let mut sum;
        sum = vaddq_s32(acc, a);
        sum = vaddq_s32(sum, b);
        sum = vaddq_s32(sum, c);
        sum = vaddq_s32(sum, d);
        sum
    }
}

#[inline(always)]
#[cfg(feature = "arm-nightly-feat")]
unsafe fn vdotq_s32_local(vz: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    if is_aarch64_feature_detected!("dotprod") {
        vdotq_s32(vz, a, b)
    } else {
        unreachable!();
    }
}
#[inline(always)]
#[cfg(not(feature = "arm-nightly-feat"))]
unsafe fn vdotq_s32_local(_vz: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let p0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    let p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1))
}
