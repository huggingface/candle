#[cfg(target_feature = "dotprod")]
use super::k_quants::BlockQ4Kx8;
use super::k_quants::{
    BlockQ2K, BlockQ3K, BlockQ4K, BlockQ4_0, BlockQ5K, BlockQ6K, BlockQ8K, BlockQ8_0, QK8_0, QK_K,
};
use byteorder::{ByteOrder, LittleEndian};

#[allow(unused_imports)]
#[cfg(target_arch = "arm")]
use core::arch::arm::*;

#[allow(unused_imports)]
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[inline(always)]
unsafe fn vdotq_s32(a: int8x16_t, b: int8x16_t) -> int32x4_t {
    #[cfg(target_feature = "dotprod")]
    {
        let mut acc: int32x4_t = vdupq_n_s32(0);
        core::arch::asm!(
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) acc,
            a = in(vreg) a,
            b = in(vreg) b,
            options(nostack, nomem),
        );
        acc
    }
    #[cfg(not(target_feature = "dotprod"))]
    {
        let p0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
        let p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
        vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1))
    }
}

/// Two SDOT ops into one accumulator
#[inline(always)]
unsafe fn vdotq_s32_pair(a0: int8x16_t, b0: int8x16_t, a1: int8x16_t, b1: int8x16_t) -> int32x4_t {
    #[cfg(target_feature = "dotprod")]
    {
        let mut acc: int32x4_t = vdupq_n_s32(0);
        core::arch::asm!(
            "sdot {acc:v}.4s, {a0:v}.16b, {b0:v}.16b",
            "sdot {acc:v}.4s, {a1:v}.16b, {b1:v}.16b",
            acc = inout(vreg) acc,
            a0 = in(vreg) a0,
            b0 = in(vreg) b0,
            a1 = in(vreg) a1,
            b1 = in(vreg) b1,
            options(nostack, nomem),
        );
        acc
    }
    #[cfg(not(target_feature = "dotprod"))]
    {
        let p0 = vmull_s8(vget_low_s8(a0), vget_low_s8(b0));
        let p1 = vmull_s8(vget_high_s8(a0), vget_high_s8(b0));
        let p2 = vmull_s8(vget_low_s8(a1), vget_low_s8(b1));
        let p3 = vmull_s8(vget_high_s8(a1), vget_high_s8(b1));
        vaddq_s32(
            vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)),
            vaddq_s32(vpaddlq_s16(p2), vpaddlq_s16(p3)),
        )
    }
}

/// Accumulating SDOT: acc += dot4(a, b) for each lane.
#[cfg(target_feature = "dotprod")]
#[inline(always)]
unsafe fn sdot_acc(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let mut out = acc;
    core::arch::asm!(
        "sdot {out:v}.4s, {a:v}.16b, {b:v}.16b",
        out = inout(vreg) out,
        a   = in(vreg) a,
        b   = in(vreg) b,
        options(nostack, nomem),
    );
    out
}

/// Decode one `BlockQ4Kx8` sub-block entry into two `int16x8_t` vectors: `(mins, scales)`, each holding values 0-63.
/// See [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/f8cc15f163e784c58fe13aee58ebc03055bb0c40/ggml/src/ggml-cpu/arch/arm/repack.cpp#L29)
#[cfg(target_feature = "dotprod")]
#[inline(always)]
unsafe fn decode_q4kx8_scales(scales_in: *const u8) -> (int16x8_t, int16x8_t) {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;
    // Direct unaligned reads avoid the copy_nonoverlapping stack round-trip.
    let sm0 = (scales_in as *const u32).read_unaligned();
    let sm1 = (scales_in.add(4) as *const u32).read_unaligned();
    let sm2 = (scales_in.add(8) as *const u32).read_unaligned();
    let mins_0_3 = sm1 & KMASK1;
    let mins_4_7 = ((sm2 >> 4) & KMASK2) | (((sm1 >> 6) & KMASK3) << 4);
    let out_mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(vcreate_u32(
        (mins_0_3 as u64) | ((mins_4_7 as u64) << 32),
    ))));
    let sc_0 = sm0 & KMASK1;
    let sc_1 = (sm2 & KMASK2) | (((sm0 >> 6) & KMASK3) << 4);
    let out_scales = vmovl_s8(vreinterpret_s8_u8(vreinterpret_u8_u32(vcreate_u32(
        (sc_0 as u64) | ((sc_1 as u64) << 32),
    ))));
    (out_mins, out_scales)
}

/// Merge two per-lane (abs_max, signed_val) accumulator pairs.
/// Each output lane holds the signed value with the larger absolute value.
#[inline(always)]
unsafe fn merge_signed_max(
    abs_a: float32x4_t,
    smax_a: float32x4_t,
    abs_b: float32x4_t,
    smax_b: float32x4_t,
) -> (float32x4_t, float32x4_t) {
    (
        vmaxq_f32(abs_a, abs_b),
        vbslq_f32(vcgtq_f32(abs_b, abs_a), smax_b, smax_a),
    )
}

#[inline(always)]
pub(crate) fn vec_dot_q4_0_q8_0(n: usize, xs: &[BlockQ4_0], ys: &[BlockQ8_0]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK8_0),
        "vec_dot_q4_0_q8_0: {n} is not divisible by {QK8_0}"
    );
    let nb = n / QK8_0;
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

            let pl0 = vdotq_s32(v0_0ls, v1_0l);
            let ph0 = vdotq_s32(v0_0hs, v1_0h);
            sumv0 = vmlaq_n_f32(
                sumv0,
                vcvtq_f32_s32(vaddq_s32(pl0, ph0)),
                x0.d.to_f32() * y0.d.to_f32(),
            );
        }
        vaddvq_f32(sumv0)
    }
}

#[inline(always)]
pub(crate) fn vec_dot_q8_0_q8_0(n: usize, xs: &[BlockQ8_0], ys: &[BlockQ8_0]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK8_0),
        "vec_dot_q8_0_q8_0: {n} is not divisible by {QK8_0}"
    );
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

            let p0 = vdotq_s32(x0_0, y0_0);
            let p1 = vdotq_s32(x0_1, y0_1);

            sumv0 = vmlaq_n_f32(
                sumv0,
                vcvtq_f32_s32(vaddq_s32(p0, p1)),
                x0.d.to_f32() * y0.d.to_f32(),
            );
        }
        vaddvq_f32(sumv0)
    }
}

#[inline(always)]
pub(crate) fn vec_dot_4_q8_0_q8_0(
    n: usize,
    xs0: &[BlockQ8_0],
    xs1: &[BlockQ8_0],
    xs2: &[BlockQ8_0],
    xs3: &[BlockQ8_0],
    ys: &[BlockQ8_0],
) -> (f32, f32, f32, f32) {
    debug_assert!(
        n.is_multiple_of(QK8_0),
        "vec_dot_4_q8_0_q8_0: {n} is not divisible by {QK8_0}"
    );
    let nb = n / QK8_0;
    unsafe {
        let mut sum0 = vdupq_n_f32(0.0f32);
        let mut sum1 = vdupq_n_f32(0.0f32);
        let mut sum2 = vdupq_n_f32(0.0f32);
        let mut sum3 = vdupq_n_f32(0.0f32);
        for i in 0..nb {
            let y = &ys[i];
            let y0 = vld1q_s8(y.qs.as_ptr());
            let y1 = vld1q_s8(y.qs.as_ptr().add(16));
            let yd = y.d.to_f32();
            let x0 = &xs0[i];
            let x1 = &xs1[i];
            let x2 = &xs2[i];
            let x3 = &xs3[i];
            // Each column uses 2 SDOTs fused into one accumulator via vdotq_s32_pair.
            // All 4 columns share y0/y1 and have independent accumulators.
            let p0 = vdotq_s32_pair(
                vld1q_s8(x0.qs.as_ptr()),
                y0,
                vld1q_s8(x0.qs.as_ptr().add(16)),
                y1,
            );
            let p1 = vdotq_s32_pair(
                vld1q_s8(x1.qs.as_ptr()),
                y0,
                vld1q_s8(x1.qs.as_ptr().add(16)),
                y1,
            );
            let p2 = vdotq_s32_pair(
                vld1q_s8(x2.qs.as_ptr()),
                y0,
                vld1q_s8(x2.qs.as_ptr().add(16)),
                y1,
            );
            let p3 = vdotq_s32_pair(
                vld1q_s8(x3.qs.as_ptr()),
                y0,
                vld1q_s8(x3.qs.as_ptr().add(16)),
                y1,
            );
            sum0 = vmlaq_n_f32(sum0, vcvtq_f32_s32(p0), x0.d.to_f32() * yd);
            sum1 = vmlaq_n_f32(sum1, vcvtq_f32_s32(p1), x1.d.to_f32() * yd);
            sum2 = vmlaq_n_f32(sum2, vcvtq_f32_s32(p2), x2.d.to_f32() * yd);
            sum3 = vmlaq_n_f32(sum3, vcvtq_f32_s32(p3), x3.d.to_f32() * yd);
        }
        (
            vaddvq_f32(sum0),
            vaddvq_f32(sum1),
            vaddvq_f32(sum2),
            vaddvq_f32(sum3),
        )
    }
}

#[inline(always)]
pub(crate) fn vec_dot_q8k_q8k(n: usize, xs: &[BlockQ8K], ys: &[BlockQ8K]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_q8k_q8k: {n} is not divisible by {QK_K}"
    );
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
    sumf
}

#[inline(always)]
pub(crate) fn vec_dot_q6k_q8k(n: usize, xs: &[BlockQ6K], ys: &[BlockQ8K]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_q6k_q8k: {n} is not divisible by {QK_K}"
    );
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

                let p0 = vdotq_s32(q6bytes_0, q8bytes.0);
                let p1 = vdotq_s32(q6bytes_1, q8bytes.1);
                let (scale0, scale1) = (*scale as i32, *scale.add(1) as i32);
                isum += vaddvq_s32(p0) * scale0 + vaddvq_s32(p1) * scale1;
                scale = scale.add(2);

                let p2 = vdotq_s32(q6bytes_2, q8bytes.2);
                let p3 = vdotq_s32(q6bytes_3, q8bytes.3);
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

                let p0 = vdotq_s32(q6bytes_0, q8bytes.0);
                let p1 = vdotq_s32(q6bytes_1, q8bytes.1);
                let (scale0, scale1) = (*scale as i32, *scale.add(1) as i32);
                isum += vaddvq_s32(p0) * scale0 + vaddvq_s32(p1) * scale1;
                scale = scale.add(2);

                let p2 = vdotq_s32(q6bytes_2, q8bytes.2);
                let p3 = vdotq_s32(q6bytes_3, q8bytes.3);
                let (scale0, scale1) = (*scale as i32, *scale.add(1) as i32);
                isum += vaddvq_s32(p2) * scale0 + vaddvq_s32(p3) * scale1;
                scale = scale.add(2);
            }
            sum += d_all * y.d * ((isum - 32 * isum_mins) as f32);
        }
    }
    sum
}

/// Four Q6K dot products sharing one Q8K load.
#[inline(always)]
pub(crate) fn vec_dot_4_q6k_q8k(
    n: usize,
    xs0: &[BlockQ6K],
    xs1: &[BlockQ6K],
    xs2: &[BlockQ6K],
    xs3: &[BlockQ6K],
    ys: &[BlockQ8K],
) -> (f32, f32, f32, f32) {
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_4_q6k_q8k: {n} is not divisible by {QK_K}"
    );

    let mut sum0 = 0f32;
    let mut sum1 = 0f32;
    let mut sum2 = 0f32;
    let mut sum3 = 0f32;

    unsafe {
        let m4b = vdupq_n_u8(0xF);
        let mone = vdupq_n_u8(3);

        for ((((x0, x1), x2), x3), y) in xs0
            .iter()
            .zip(xs1.iter())
            .zip(xs2.iter())
            .zip(xs3.iter())
            .zip(ys.iter())
        {
            let yd = y.d;

            // Q8K bsums - loaded once for all four columns.
            let q8sums = vld1q_s16_x2(y.bsums.as_ptr());

            // Compute isum_mins for each column: dot(q8sums, column_scales).
            macro_rules! col_isum_mins {
                ($x:ident) => {{
                    let scales_v = vld1q_s8($x.scales.as_ptr());
                    let q6sc = int16x8x2_t(
                        vmovl_s8(vget_low_s8(scales_v)),
                        vmovl_s8(vget_high_s8(scales_v)),
                    );
                    let prod = vaddq_s32(
                        vaddq_s32(
                            vmull_s16(vget_low_s16(q8sums.0), vget_low_s16(q6sc.0)),
                            vmull_s16(vget_high_s16(q8sums.0), vget_high_s16(q6sc.0)),
                        ),
                        vaddq_s32(
                            vmull_s16(vget_low_s16(q8sums.1), vget_low_s16(q6sc.1)),
                            vmull_s16(vget_high_s16(q8sums.1), vget_high_s16(q6sc.1)),
                        ),
                    );
                    vaddvq_s32(prod)
                }};
            }

            let isum_mins0 = col_isum_mins!(x0);
            let isum_mins1 = col_isum_mins!(x1);
            let isum_mins2 = col_isum_mins!(x2);
            let isum_mins3 = col_isum_mins!(x3);

            let mut q6_0 = x0.ql.as_ptr();
            let mut qh_0 = x0.qh.as_ptr();
            let mut sc_0 = x0.scales.as_ptr();
            let mut q6_1 = x1.ql.as_ptr();
            let mut qh_1 = x1.qh.as_ptr();
            let mut sc_1 = x1.scales.as_ptr();
            let mut q6_2 = x2.ql.as_ptr();
            let mut qh_2 = x2.qh.as_ptr();
            let mut sc_2 = x2.scales.as_ptr();
            let mut q6_3 = x3.ql.as_ptr();
            let mut qh_3 = x3.qh.as_ptr();
            let mut sc_3 = x3.scales.as_ptr();
            let mut q8 = y.qs.as_ptr();

            let mut isum0 = 0i32;
            let mut isum1 = 0i32;
            let mut isum2 = 0i32;
            let mut isum3 = 0i32;

            for _j in 0..QK_K / 128 {
                // Load Q8K bytes once - shared across all four columns.
                let q8lo = vld1q_s8_x4(q8);
                q8 = q8.add(64);
                let q8hi = vld1q_s8_x4(q8);
                q8 = q8.add(64);

                // Decode one column's Q6 bits and accumulate into its isum.
                // q8lo/q8hi are shared from the outer scope.
                macro_rules! process_col {
                    ($q6:ident, $qh:ident, $sc:ident, $isum:ident) => {
                        let qhb = vld1q_u8_x2($qh);
                        $qh = $qh.add(32);
                        let q6b = vld1q_u8_x4($q6);
                        $q6 = $q6.add(64);

                        // First half: low nibbles of ql + bits[1:0] and bits[3:2] of qh.
                        let qh00 = vshlq_n_u8(vandq_u8(mone, qhb.0), 4);
                        let qh01 = vshlq_n_u8(vandq_u8(mone, qhb.1), 4);
                        let qh10 = vshlq_n_u8(vandq_u8(mone, vshrq_n_u8(qhb.0, 2)), 4);
                        let qh11 = vshlq_n_u8(vandq_u8(mone, vshrq_n_u8(qhb.1, 2)), 4);

                        let q6b0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6b.0, m4b), qh00));
                        let q6b1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6b.1, m4b), qh01));
                        let q6b2 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6b.2, m4b), qh10));
                        let q6b3 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6b.3, m4b), qh11));

                        let p0 = vdotq_s32(q6b0, q8lo.0);
                        let p1 = vdotq_s32(q6b1, q8lo.1);
                        $isum +=
                            vaddvq_s32(p0) * (*$sc as i32) + vaddvq_s32(p1) * (*$sc.add(1) as i32);
                        $sc = $sc.add(2);

                        let p2 = vdotq_s32(q6b2, q8lo.2);
                        let p3 = vdotq_s32(q6b3, q8lo.3);
                        $isum +=
                            vaddvq_s32(p2) * (*$sc as i32) + vaddvq_s32(p3) * (*$sc.add(1) as i32);
                        $sc = $sc.add(2);

                        // Second half: high nibbles of ql + bits[5:4] and bits[7:6] of qh.
                        let qh20 = vshlq_n_u8(vandq_u8(mone, vshrq_n_u8(qhb.0, 4)), 4);
                        let qh21 = vshlq_n_u8(vandq_u8(mone, vshrq_n_u8(qhb.1, 4)), 4);
                        let qh30 = vshlq_n_u8(vandq_u8(mone, vshrq_n_u8(qhb.0, 6)), 4);
                        let qh31 = vshlq_n_u8(vandq_u8(mone, vshrq_n_u8(qhb.1, 6)), 4);

                        let q6b0 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6b.0, 4), qh20));
                        let q6b1 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6b.1, 4), qh21));
                        let q6b2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6b.2, 4), qh30));
                        let q6b3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6b.3, 4), qh31));

                        let p0 = vdotq_s32(q6b0, q8hi.0);
                        let p1 = vdotq_s32(q6b1, q8hi.1);
                        $isum +=
                            vaddvq_s32(p0) * (*$sc as i32) + vaddvq_s32(p1) * (*$sc.add(1) as i32);
                        $sc = $sc.add(2);

                        let p2 = vdotq_s32(q6b2, q8hi.2);
                        let p3 = vdotq_s32(q6b3, q8hi.3);
                        $isum +=
                            vaddvq_s32(p2) * (*$sc as i32) + vaddvq_s32(p3) * (*$sc.add(1) as i32);
                        $sc = $sc.add(2);
                    };
                }

                process_col!(q6_0, qh_0, sc_0, isum0);
                process_col!(q6_1, qh_1, sc_1, isum1);
                process_col!(q6_2, qh_2, sc_2, isum2);
                process_col!(q6_3, qh_3, sc_3, isum3);
            }

            sum0 += x0.d.to_f32() * yd * ((isum0 - 32 * isum_mins0) as f32);
            sum1 += x1.d.to_f32() * yd * ((isum1 - 32 * isum_mins1) as f32);
            sum2 += x2.d.to_f32() * yd * ((isum2 - 32 * isum_mins2) as f32);
            sum3 += x3.d.to_f32() * yd * ((isum3 - 32 * isum_mins3) as f32);
        }
    }

    (sum0, sum1, sum2, sum3)
}

#[inline(always)]
pub(crate) fn vec_dot_q5k_q8k(n: usize, xs: &[BlockQ5K], ys: &[BlockQ8K]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_q5k_q8k: {n} is not divisible by {QK_K}"
    );
    let mut sumf = 0f32;
    let mut utmp = [0u32; 4];
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    unsafe {
        let m4b = vdupq_n_u8(0xF);
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

                let p0 = vdotq_s32(q5bytes_0, q8bytes.0);
                let p1 = vdotq_s32(q5bytes_1, q8bytes.1);
                sumi += vaddvq_s32(vaddq_s32(p0, p1)) * *scales as i32;
                scales = scales.add(1);

                let p2 = vdotq_s32(q5bytes_2, q8bytes.2);
                let p3 = vdotq_s32(q5bytes_3, q8bytes.3);
                sumi += vaddvq_s32(vaddq_s32(p2, p3)) * *scales as i32;
                scales = scales.add(1);
            }
            sumf += d * sumi as f32 - dmin * sumi_mins as f32;
        }
    }
    sumf
}

#[inline(always)]
pub(crate) fn vec_dot_q4k_q8k(n: usize, xs: &[BlockQ4K], ys: &[BlockQ8K]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_q4k_q8k: {n} is not divisible by {QK_K}"
    );
    let mut sumf = 0f32;
    let mut utmp = [0u32; 4];
    let mut scales = [0u8; 16];
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

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
                let p0 = vdotq_s32(q4bytes.0, q8bytes.0);
                let p1 = vdotq_s32(q4bytes.1, q8bytes.1);
                sumi1 += vaddvq_s32(vaddq_s32(p0, p1)) * scales[2 * j] as i32;

                let q8bytes = vld1q_s8_x2(q8);
                q8 = q8.add(32);
                let q4bytes = int8x16x2_t(
                    vreinterpretq_s8_u8(vshrq_n_u8(q4bits.0, 4)),
                    vreinterpretq_s8_u8(vshrq_n_u8(q4bits.1, 4)),
                );
                let p2 = vdotq_s32(q4bytes.0, q8bytes.0);
                let p3 = vdotq_s32(q4bytes.1, q8bytes.1);
                sumi2 += vaddvq_s32(vaddq_s32(p2, p3)) * scales[2 * j + 1] as i32;
            }
            sumf += d * (sumi1 + sumi2) as f32;
        }
    }
    sumf
}

/// Four Q4K dot products sharing one Q8K load.
#[inline(always)]
pub(crate) fn vec_dot_4_q4k_q8k(
    n: usize,
    xs0: &[BlockQ4K],
    xs1: &[BlockQ4K],
    xs2: &[BlockQ4K],
    xs3: &[BlockQ4K],
    ys: &[BlockQ8K],
) -> (f32, f32, f32, f32) {
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_4_q4k_q8k: {n} is not divisible by {QK_K}"
    );

    let mut sum0 = 0f32;
    let mut sum1 = 0f32;
    let mut sum2 = 0f32;
    let mut sum3 = 0f32;

    let mut utmp = [0u32; 4];
    let mut sc0 = [0u8; 16];
    let mut sc1 = [0u8; 16];
    let mut sc2 = [0u8; 16];
    let mut sc3 = [0u8; 16];

    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    // Decode Q4K scale+min bytes into an 8-entry scales array and a float32x4x2
    // mins vector. Returns the mins vector so the caller can compute min correction.
    macro_rules! decode_q4k_scales {
        ($x:ident, $sc:ident) => {{
            LittleEndian::read_u32_into(&$x.scales, &mut utmp[0..3]);
            let mins8 = vld1_u32(
                [
                    utmp[1] & KMASK1,
                    ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4),
                ]
                .as_ptr(),
            );
            utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
            utmp[0] &= KMASK1;
            LittleEndian::write_u32_into(&utmp, &mut $sc);
            vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)))
        }};
    }

    // Accumulate one column's nibble-group contribution into a scalar sumi.
    macro_rules! dot_col {
        ($q4:ident, $sc:ident, $vsum1:ident, $vsum2:ident, $q8lo:ident, $q8hi:ident, $j:ident, $m4b:ident) => {
            let bits = vld1q_u8_x2($q4);
            $q4 = $q4.add(32);
            let q4lo = int8x16x2_t(
                vreinterpretq_s8_u8(vandq_u8(bits.0, $m4b)),
                vreinterpretq_s8_u8(vandq_u8(bits.1, $m4b)),
            );
            $vsum1 = vmlaq_n_s32(
                $vsum1,
                vdotq_s32_pair(q4lo.0, $q8lo.0, q4lo.1, $q8lo.1),
                $sc[2 * $j] as i32,
            );
            let q4hi = int8x16x2_t(
                vreinterpretq_s8_u8(vshrq_n_u8(bits.0, 4)),
                vreinterpretq_s8_u8(vshrq_n_u8(bits.1, 4)),
            );
            $vsum2 = vmlaq_n_s32(
                $vsum2,
                vdotq_s32_pair(q4hi.0, $q8hi.0, q4hi.1, $q8hi.1),
                $sc[2 * $j + 1] as i32,
            );
        };
    }

    unsafe {
        let m4b = vdupq_n_u8(0xF);

        for ((((x0, x1), x2), x3), y) in xs0
            .iter()
            .zip(xs1.iter())
            .zip(xs2.iter())
            .zip(xs3.iter())
            .zip(ys.iter())
        {
            let yd = y.d;

            // Q8K bsums - loaded once for all four columns.
            let q8sums = vpaddq_s16(
                vld1q_s16(y.bsums.as_ptr()),
                vld1q_s16(y.bsums.as_ptr().add(8)),
            );

            // Decode scales and apply min correction for each column.
            let mins0 = decode_q4k_scales!(x0, sc0);
            let mins1 = decode_q4k_scales!(x1, sc1);
            let mins2 = decode_q4k_scales!(x2, sc2);
            let mins3 = decode_q4k_scales!(x3, sc3);

            let d0 = yd * x0.d.to_f32();
            let d1 = yd * x1.d.to_f32();
            let d2 = yd * x2.d.to_f32();
            let d3 = yd * x3.d.to_f32();

            // min correction: sum -= dmin * dot(q8sums, mins)
            macro_rules! min_correct {
                ($mins:ident, $dmin:expr, $sum:ident) => {
                    let prod = vaddq_s32(
                        vmull_s16(vget_low_s16(q8sums), vget_low_s16($mins)),
                        vmull_s16(vget_high_s16(q8sums), vget_high_s16($mins)),
                    );
                    $sum -= $dmin * vaddvq_s32(prod) as f32;
                };
            }
            min_correct!(mins0, yd * x0.dmin.to_f32(), sum0);
            min_correct!(mins1, yd * x1.dmin.to_f32(), sum1);
            min_correct!(mins2, yd * x2.dmin.to_f32(), sum2);
            min_correct!(mins3, yd * x3.dmin.to_f32(), sum3);

            let mut q4_0 = x0.qs.as_ptr();
            let mut q4_1 = x1.qs.as_ptr();
            let mut q4_2 = x2.qs.as_ptr();
            let mut q4_3 = x3.qs.as_ptr();
            let mut q8 = y.qs.as_ptr();

            let mut s0a = vdupq_n_s32(0);
            let mut s0b = vdupq_n_s32(0);
            let mut s1a = vdupq_n_s32(0);
            let mut s1b = vdupq_n_s32(0);
            let mut s2a = vdupq_n_s32(0);
            let mut s2b = vdupq_n_s32(0);
            let mut s3a = vdupq_n_s32(0);
            let mut s3b = vdupq_n_s32(0);

            for j in 0..QK_K / 64 {
                // Load Q8K once and reuse across all four columns.
                let q8lo = vld1q_s8_x2(q8);
                q8 = q8.add(32);
                let q8hi = vld1q_s8_x2(q8);
                q8 = q8.add(32);

                dot_col!(q4_0, sc0, s0a, s0b, q8lo, q8hi, j, m4b);
                dot_col!(q4_1, sc1, s1a, s1b, q8lo, q8hi, j, m4b);
                dot_col!(q4_2, sc2, s2a, s2b, q8lo, q8hi, j, m4b);
                dot_col!(q4_3, sc3, s3a, s3b, q8lo, q8hi, j, m4b);
            }

            sum0 += d0 * vaddvq_s32(vaddq_s32(s0a, s0b)) as f32;
            sum1 += d1 * vaddvq_s32(vaddq_s32(s1a, s1b)) as f32;
            sum2 += d2 * vaddvq_s32(vaddq_s32(s2a, s2b)) as f32;
            sum3 += d3 * vaddvq_s32(vaddq_s32(s3a, s3b)) as f32;
        }
    }

    (sum0, sum1, sum2, sum3)
}

#[inline(always)]
pub(crate) fn vec_dot_q3k_q8k(n: usize, xs: &[BlockQ3K], ys: &[BlockQ8K]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_q3k_q8k: {n} is not divisible by {QK_K}"
    );
    let mut sumf = 0f32;
    let mut utmp = [0u32; 4];
    let mut aux = [0u32; 3];
    const KMASK1: u32 = 0x03030303;
    const KMASK2: u32 = 0x0f0f0f0f;

    unsafe {
        let m3b = vdupq_n_u8(0x3);
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

                let p0 = vdotq_s32(q3bytes_0, q8bytes_1.0);
                let p1 = vdotq_s32(q3bytes_1, q8bytes_1.1);
                let p2 = vdotq_s32(q3bytes_2, q8bytes_1.2);
                let p3 = vdotq_s32(q3bytes_3, q8bytes_1.3);
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

                let p0 = vdotq_s32(q3bytes_0, q8bytes_2.0);
                let p1 = vdotq_s32(q3bytes_1, q8bytes_2.1);
                let p2 = vdotq_s32(q3bytes_2, q8bytes_2.2);
                let p3 = vdotq_s32(q3bytes_3, q8bytes_2.3);
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
    sumf
}

#[inline(always)]
pub(crate) fn vec_dot_q2k_q8k(n: usize, xs: &[BlockQ2K], ys: &[BlockQ8K]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_q2k_q8k: {n} is not divisible by {QK_K}"
    );
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

            // TODO: dotprod
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
    sumf
}

/// Quantize a row of f32 activations into Q8K format
#[inline(always)]
pub(crate) fn quantize_row_q8k(xs: &[f32], ys: &mut [BlockQ8K]) {
    debug_assert!(
        xs.len().is_multiple_of(QK_K),
        "quantize_row_q8k: {} is not a multiple of {QK_K}",
        xs.len()
    );
    unsafe {
        for (chunk, y) in xs.chunks_exact(QK_K).zip(ys.iter_mut()) {
            // Find the element with the maximum absolute value, preserving its sign.
            let (mut vabs_max0, mut vsmax0) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
            let (mut vabs_max1, mut vsmax1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
            let (mut vabs_max2, mut vsmax2) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
            let (mut vabs_max3, mut vsmax3) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
            let mut p = chunk.as_ptr();
            for _ in 0..QK_K / 16 {
                let (v0, v1) = (vld1q_f32(p), vld1q_f32(p.add(4)));
                let (v2, v3) = (vld1q_f32(p.add(8)), vld1q_f32(p.add(12)));
                p = p.add(16);
                (vabs_max0, vsmax0) = merge_signed_max(vabs_max0, vsmax0, vabsq_f32(v0), v0);
                (vabs_max1, vsmax1) = merge_signed_max(vabs_max1, vsmax1, vabsq_f32(v1), v1);
                (vabs_max2, vsmax2) = merge_signed_max(vabs_max2, vsmax2, vabsq_f32(v2), v2);
                (vabs_max3, vsmax3) = merge_signed_max(vabs_max3, vsmax3, vabsq_f32(v3), v3);
            }
            // Tree-reduce 4 accumulators to 1.
            let (abs01, smax01) = merge_signed_max(vabs_max0, vsmax0, vabs_max1, vsmax1);
            let (abs23, smax23) = merge_signed_max(vabs_max2, vsmax2, vabs_max3, vsmax3);
            let (abs_v, smax_v) = merge_signed_max(abs01, smax01, abs23, smax23);
            // Cross lane reduce to scalar
            let mask_lohi = vcgt_f32(vget_high_f32(abs_v), vget_low_f32(abs_v));
            let abs_pair = vmax_f32(vget_low_f32(abs_v), vget_high_f32(abs_v));
            let smax_pair = vbsl_f32(mask_lohi, vget_high_f32(smax_v), vget_low_f32(smax_v));
            let max_signed = if vget_lane_f32(abs_pair, 1) > vget_lane_f32(abs_pair, 0) {
                vget_lane_f32(smax_pair, 1)
            } else {
                vget_lane_f32(smax_pair, 0)
            };

            if max_signed == 0.0f32 {
                y.d = 0.0f32;
                y.qs.fill(0);
                y.bsums.fill(0);
                continue;
            }

            let iscale = -127.0f32 / max_signed;
            let vscale = vdupq_n_f32(iscale);

            // Quantize f32 -> i8. Multiply, round-to-nearest, saturating narrow.
            let mut out = y.qs.as_mut_ptr();
            let mut p = chunk.as_ptr();
            for j in 0..QK_K / 16 {
                let f0 = vmulq_f32(vld1q_f32(p), vscale);
                let f1 = vmulq_f32(vld1q_f32(p.add(4)), vscale);
                let f2 = vmulq_f32(vld1q_f32(p.add(8)), vscale);
                let f3 = vmulq_f32(vld1q_f32(p.add(12)), vscale);
                p = p.add(16);
                let s01 = vcombine_s16(
                    vqmovn_s32(vcvtaq_s32_f32(f0)),
                    vqmovn_s32(vcvtaq_s32_f32(f1)),
                );
                let s23 = vcombine_s16(
                    vqmovn_s32(vcvtaq_s32_f32(f2)),
                    vqmovn_s32(vcvtaq_s32_f32(f3)),
                );
                let q = vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23));
                vst1q_s8(out, q);
                out = out.add(16);
                y.bsums[j] = vaddvq_s32(vpaddlq_s16(vpaddlq_s8(q))) as i16;
            }

            y.d = 1.0f32 / iscale;
        }
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
    let p1 = vdotq_s32(q2bytes.0, q8bytes.0);
    let p2 = vdotq_s32(q2bytes.1, q8bytes.1);
    vaddvq_s32(p1) * aux[is + index] as i32 + vaddvq_s32(p2) * aux[is + 1 + index] as i32
}

/// Eight Q4K dot products sharing one Q8K load, using the `BlockQ4Kx8` layout.
#[cfg(target_feature = "dotprod")]
#[inline(always)]
pub(crate) fn vec_dot_8_q4k_q8k(n: usize, xs: &[BlockQ4Kx8], ys: &[BlockQ8K]) -> [f32; 8] {
    debug_assert!(n.is_multiple_of(QK_K));
    let mut out = [0f32; 8];

    #[inline(always)]
    unsafe fn load_f16x4(ptr: *const half::f16) -> float32x4_t {
        let raw = vld1_u64(ptr as *const u64);
        let mut result: float32x4_t;
        core::arch::asm!(
            "fcvtl {out:v}.4s, {inp:v}.4h",
            inp = in(vreg) raw,
            out = out(vreg) result,
            options(nostack, nomem),
        );
        result
    }

    unsafe {
        let mut vacc_0 = vdupq_n_f32(0.0f32);
        let mut vacc_1 = vdupq_n_f32(0.0f32);
        let m4b = vdupq_n_u8(0x0f);
        for (q4, q8) in xs.iter().zip(ys.iter()) {
            let q8d = q8.d;
            let q8d_v = vdupq_n_f32(q8d);
            let sb_scale_0 = vmulq_f32(load_f16x4(q4.d.as_ptr()), q8d_v);
            let sb_scale_1 = vmulq_f32(load_f16x4(q4.d.as_ptr().add(4)), q8d_v);
            let sb_min_0 = vmulq_f32(load_f16x4(q4.dmin.as_ptr()), q8d_v);
            let sb_min_1 = vmulq_f32(load_f16x4(q4.dmin.as_ptr().add(4)), q8d_v);
            // bsums: 16 sums of 16 elements -> pairwise add -> 8 sums of 32 elements.
            // Kept as a NEON register. Const generic lane extraction below avoids
            // the stack spill that runtime indexed bsums_arr[2*sb] causes.
            let bsums = vpaddq_s16(
                vld1q_s16(q8.bsums.as_ptr()),
                vld1q_s16(q8.bsums.as_ptr().add(8)),
            );
            let mut bias_0 = vdupq_n_s32(0);
            let mut bias_1 = vdupq_n_s32(0);
            // Manually unroll the 4-sub-block loop (QK_K/64 = 4).
            macro_rules! process_sb {
                ($sb:literal) => {{
                    let (mins0, sc0) = decode_q4kx8_scales(q4.scales.as_ptr().add($sb * 24));
                    let (mins1, sc1) = decode_q4kx8_scales(q4.scales.as_ptr().add($sb * 24 + 12));
                    let q8p = q8.qs.as_ptr().add($sb * 64);
                    let q8_0 = vreinterpretq_s8_s64(vld1q_dup_s64(q8p as *const i64));
                    let q8_1 = vreinterpretq_s8_s64(vld1q_dup_s64(q8p.add(8) as *const i64));
                    let q8_2 = vreinterpretq_s8_s64(vld1q_dup_s64(q8p.add(16) as *const i64));
                    let q8_3 = vreinterpretq_s8_s64(vld1q_dup_s64(q8p.add(24) as *const i64));
                    let q8_4 = vreinterpretq_s8_s64(vld1q_dup_s64(q8p.add(32) as *const i64));
                    let q8_5 = vreinterpretq_s8_s64(vld1q_dup_s64(q8p.add(40) as *const i64));
                    let q8_6 = vreinterpretq_s8_s64(vld1q_dup_s64(q8p.add(48) as *const i64));
                    let q8_7 = vreinterpretq_s8_s64(vld1q_dup_s64(q8p.add(56) as *const i64));
                    let q4p = q4.qs.as_ptr().add($sb * QK_K);
                    let s0 = vld1q_u8_x2(q4p);
                    let s0h = vld1q_u8_x2(q4p.add(32));
                    let s1 = vld1q_u8_x2(q4p.add(64));
                    let s1h = vld1q_u8_x2(q4p.add(96));
                    let s2 = vld1q_u8_x2(q4p.add(128));
                    let s2h = vld1q_u8_x2(q4p.add(160));
                    let s3 = vld1q_u8_x2(q4p.add(192));
                    let s3h = vld1q_u8_x2(q4p.add(224));
                    let (b00, b10) = (s0.0, s0.1);
                    let (b20, b30) = (s0h.0, s0h.1);
                    let (b01, b11) = (s1.0, s1.1);
                    let (b21, b31) = (s1h.0, s1h.1);
                    let (b02, b12) = (s2.0, s2.1);
                    let (b22, b32) = (s2h.0, s2h.1);
                    let (b03, b13) = (s3.0, s3.1);
                    let (b23, b33) = (s3h.0, s3h.1);
                    let mut a0 = vdupq_n_s32(0);
                    a0 = sdot_acc(a0, vreinterpretq_s8_u8(vandq_u8(b00, m4b)), q8_0);
                    a0 = sdot_acc(a0, vreinterpretq_s8_u8(vandq_u8(b01, m4b)), q8_1);
                    a0 = sdot_acc(a0, vreinterpretq_s8_u8(vandq_u8(b02, m4b)), q8_2);
                    a0 = sdot_acc(a0, vreinterpretq_s8_u8(vandq_u8(b03, m4b)), q8_3);
                    let mut h0 = vdupq_n_s32(0);
                    h0 = sdot_acc(h0, vreinterpretq_s8_u8(vshrq_n_u8(b00, 4)), q8_4);
                    h0 = sdot_acc(h0, vreinterpretq_s8_u8(vshrq_n_u8(b01, 4)), q8_5);
                    h0 = sdot_acc(h0, vreinterpretq_s8_u8(vshrq_n_u8(b02, 4)), q8_6);
                    h0 = sdot_acc(h0, vreinterpretq_s8_u8(vshrq_n_u8(b03, 4)), q8_7);
                    let mut a1 = vdupq_n_s32(0);
                    a1 = sdot_acc(a1, vreinterpretq_s8_u8(vandq_u8(b10, m4b)), q8_0);
                    a1 = sdot_acc(a1, vreinterpretq_s8_u8(vandq_u8(b11, m4b)), q8_1);
                    a1 = sdot_acc(a1, vreinterpretq_s8_u8(vandq_u8(b12, m4b)), q8_2);
                    a1 = sdot_acc(a1, vreinterpretq_s8_u8(vandq_u8(b13, m4b)), q8_3);
                    let mut h1 = vdupq_n_s32(0);
                    h1 = sdot_acc(h1, vreinterpretq_s8_u8(vshrq_n_u8(b10, 4)), q8_4);
                    h1 = sdot_acc(h1, vreinterpretq_s8_u8(vshrq_n_u8(b11, 4)), q8_5);
                    h1 = sdot_acc(h1, vreinterpretq_s8_u8(vshrq_n_u8(b12, 4)), q8_6);
                    h1 = sdot_acc(h1, vreinterpretq_s8_u8(vshrq_n_u8(b13, 4)), q8_7);
                    let sumf_lo_03 =
                        vcvtq_f32_s32(vmulq_s32(vmovl_s16(vget_low_s16(sc0)), vpaddq_s32(a0, a1)));
                    vacc_0 = vfmaq_f32(vacc_0, sb_scale_0, sumf_lo_03);
                    let sumf_hi_03 =
                        vcvtq_f32_s32(vmulq_s32(vmovl_s16(vget_low_s16(sc1)), vpaddq_s32(h0, h1)));
                    vacc_0 = vfmaq_f32(vacc_0, sb_scale_0, sumf_hi_03);
                    let mut a2 = vdupq_n_s32(0);
                    a2 = sdot_acc(a2, vreinterpretq_s8_u8(vandq_u8(b20, m4b)), q8_0);
                    a2 = sdot_acc(a2, vreinterpretq_s8_u8(vandq_u8(b21, m4b)), q8_1);
                    a2 = sdot_acc(a2, vreinterpretq_s8_u8(vandq_u8(b22, m4b)), q8_2);
                    a2 = sdot_acc(a2, vreinterpretq_s8_u8(vandq_u8(b23, m4b)), q8_3);
                    let mut h2 = vdupq_n_s32(0);
                    h2 = sdot_acc(h2, vreinterpretq_s8_u8(vshrq_n_u8(b20, 4)), q8_4);
                    h2 = sdot_acc(h2, vreinterpretq_s8_u8(vshrq_n_u8(b21, 4)), q8_5);
                    h2 = sdot_acc(h2, vreinterpretq_s8_u8(vshrq_n_u8(b22, 4)), q8_6);
                    h2 = sdot_acc(h2, vreinterpretq_s8_u8(vshrq_n_u8(b23, 4)), q8_7);
                    let mut a3 = vdupq_n_s32(0);
                    a3 = sdot_acc(a3, vreinterpretq_s8_u8(vandq_u8(b30, m4b)), q8_0);
                    a3 = sdot_acc(a3, vreinterpretq_s8_u8(vandq_u8(b31, m4b)), q8_1);
                    a3 = sdot_acc(a3, vreinterpretq_s8_u8(vandq_u8(b32, m4b)), q8_2);
                    a3 = sdot_acc(a3, vreinterpretq_s8_u8(vandq_u8(b33, m4b)), q8_3);
                    let mut h3 = vdupq_n_s32(0);
                    h3 = sdot_acc(h3, vreinterpretq_s8_u8(vshrq_n_u8(b30, 4)), q8_4);
                    h3 = sdot_acc(h3, vreinterpretq_s8_u8(vshrq_n_u8(b31, 4)), q8_5);
                    h3 = sdot_acc(h3, vreinterpretq_s8_u8(vshrq_n_u8(b32, 4)), q8_6);
                    h3 = sdot_acc(h3, vreinterpretq_s8_u8(vshrq_n_u8(b33, 4)), q8_7);
                    let sumf_lo_47 =
                        vcvtq_f32_s32(vmulq_s32(vmovl_s16(vget_high_s16(sc0)), vpaddq_s32(a2, a3)));
                    vacc_1 = vfmaq_f32(vacc_1, sb_scale_1, sumf_lo_47);
                    let sumf_hi_47 =
                        vcvtq_f32_s32(vmulq_s32(vmovl_s16(vget_high_s16(sc1)), vpaddq_s32(h2, h3)));
                    vacc_1 = vfmaq_f32(vacc_1, sb_scale_1, sumf_hi_47);
                    // Const-generic lane extraction avoids any stack spill of bsums
                    let bl = vdup_n_s16(vgetq_lane_s16::<{ $sb * 2 }>(bsums));
                    let bh = vdup_n_s16(vgetq_lane_s16::<{ $sb * 2 + 1 }>(bsums));
                    bias_0 = vmlal_s16(bias_0, bl, vget_low_s16(mins0));
                    bias_0 = vmlal_s16(bias_0, bh, vget_low_s16(mins1));
                    bias_1 = vmlal_s16(bias_1, bl, vget_high_s16(mins0));
                    bias_1 = vmlal_s16(bias_1, bh, vget_high_s16(mins1));
                }};
            }
            process_sb!(0);
            process_sb!(1);
            process_sb!(2);
            process_sb!(3);
            // Apply dmin correction per block: acc -= bias * (y.d * x.dmin)
            vacc_0 = vmlsq_f32(vacc_0, vcvtq_f32_s32(bias_0), sb_min_0);
            vacc_1 = vmlsq_f32(vacc_1, vcvtq_f32_s32(bias_1), sb_min_1);
        }
        vst1q_f32(out.as_mut_ptr(), vacc_0);
        vst1q_f32(out.as_mut_ptr().add(4), vacc_1);
    }
    out
}
