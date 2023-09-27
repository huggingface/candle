use super::k_quants::{BlockQ4K, BlockQ4_0, BlockQ8K, BlockQ8_0, QK8_0, QK_K};
use crate::Result;
use byteorder::{ByteOrder, LittleEndian};
use half::f16;

use core::arch::wasm32::*;

#[inline(always)]
pub(crate) fn vec_dot_q4_0_q8_0(n: usize, xs: &[BlockQ4_0], ys: &[BlockQ8_0]) -> Result<f32> {
    let qk = QK8_0;
    if n % QK8_0 != 0 {
        crate::bail!("vec_dot_q4_0_q8_0: {n} is not divisible by {qk}")
    }
    let nb = n / QK8_0;
    if nb % 2 != 0 {
        crate::bail!("vec_dot_q4_0_q8_0: {nb} is not even")
    }
    unsafe {
        let mut acc = f32x4_splat(0.0f32);
        for (x, y) in xs.iter().zip(ys.iter()) {
            let x1234 = v128_load(x.qs.as_ptr() as *const v128);
            let x12 = v128_and(x1234, u8x16_splat(0x0F));
            let x12 = i8x16_sub(x12, i8x16_splat(8));
            let x34 = u8x16_shr(x1234, 4);
            let x34 = i8x16_sub(x34, i8x16_splat(8));

            let x1 = i16x8_extend_low_i8x16(x12);
            let y1 = i16x8_load_extend_i8x8(y.qs.as_ptr());
            let sum_xy = i32x4_dot_i16x8(x1, y1);

            let x2 = i16x8_extend_high_i8x16(x12);
            let y2 = i16x8_load_extend_i8x8(y.qs.as_ptr().add(8));
            let sum_xy = i32x4_add(sum_xy, i32x4_dot_i16x8(x2, y2));

            let x3 = i16x8_extend_low_i8x16(x34);
            let y3 = i16x8_load_extend_i8x8(y.qs.as_ptr().add(16));
            let sum_xy = i32x4_add(sum_xy, i32x4_dot_i16x8(x3, y3));

            let x4 = i16x8_extend_high_i8x16(x34);
            let y4 = i16x8_load_extend_i8x8(y.qs.as_ptr().add(24));
            let sum_xy = i32x4_add(sum_xy, i32x4_dot_i16x8(x4, y4));

            let sum_xy = f32x4_convert_i32x4(sum_xy);

            // f32x4_relaxed_madd is nightly only.
            let d = f32x4_splat(f16::to_f32(x.d) * f16::to_f32(y.d));
            let scaled = f32x4_mul(sum_xy, d);
            acc = f32x4_add(acc, scaled)
        }
        let res = f32x4_extract_lane::<0>(acc)
            + f32x4_extract_lane::<1>(acc)
            + f32x4_extract_lane::<2>(acc)
            + f32x4_extract_lane::<3>(acc);
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
    if nb % 2 != 0 {
        crate::bail!("vec_dot_q8_0_q8_0: {nb} is not even")
    }
    unsafe {
        let mut acc = f32x4_splat(0.0f32);
        for (x, y) in xs.iter().zip(ys.iter()) {
            let x1 = i16x8_load_extend_i8x8(x.qs.as_ptr());
            let y1 = i16x8_load_extend_i8x8(y.qs.as_ptr());
            let sum_xy = i32x4_dot_i16x8(x1, y1);

            let x2 = i16x8_load_extend_i8x8(x.qs.as_ptr().add(8));
            let y2 = i16x8_load_extend_i8x8(y.qs.as_ptr().add(8));
            let sum_xy = i32x4_add(sum_xy, i32x4_dot_i16x8(x2, y2));

            let x3 = i16x8_load_extend_i8x8(x.qs.as_ptr().add(16));
            let y3 = i16x8_load_extend_i8x8(y.qs.as_ptr().add(16));
            let sum_xy = i32x4_add(sum_xy, i32x4_dot_i16x8(x3, y3));

            let x4 = i16x8_load_extend_i8x8(x.qs.as_ptr().add(24));
            let y4 = i16x8_load_extend_i8x8(y.qs.as_ptr().add(24));
            let sum_xy = i32x4_add(sum_xy, i32x4_dot_i16x8(x4, y4));

            let sum_xy = f32x4_convert_i32x4(sum_xy);

            // f32x4_relaxed_madd is nightly only.
            let d = f32x4_splat(f16::to_f32(x.d) * f16::to_f32(y.d));
            let scaled = f32x4_mul(sum_xy, d);
            acc = f32x4_add(acc, scaled)
        }
        let res = f32x4_extract_lane::<0>(acc)
            + f32x4_extract_lane::<1>(acc)
            + f32x4_extract_lane::<2>(acc)
            + f32x4_extract_lane::<3>(acc);
        Ok(res)
    }
}

#[inline(always)]
pub(crate) fn vec_dot_q4k_q8k(n: usize, xs: &[BlockQ4K], ys: &[BlockQ8K]) -> Result<f32> {
    if n % QK_K != 0 {
        crate::bail!("vec_dot_q4k_q8k: {n} is not divisible by {QK_K}")
    }

    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let mut utmp: [u32; 4] = [0; 4];
    let mut scales: [u8; 8] = [0; 8];
    let mut mins: [u8; 8] = [0; 8];

    let mut aux8: [i8; QK_K] = [0; QK_K];
    let mut sums = f32x4_splat(0f32);

    let mut sumf = 0.0;
    let sums = unsafe {
        for (y, x) in ys.iter().zip(xs.iter()) {
            let q4 = &x.qs;
            let q8 = &y.qs;

            let mut a = &mut aux8[..];
            let mut q4 = &q4[..];
            for _ in 0..QK_K / 64 {
                for l in 0..32 {
                    a[l] = (q4[l] & 0xF) as i8;
                }
                a = &mut a[32..];
                for l in 0..32 {
                    a[l] = (q4[l] >> 4) as i8;
                }
                a = &mut a[32..];
                q4 = &q4[32..];
            }

            LittleEndian::read_u32_into(&x.scales, &mut utmp[0..3]);

            utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
            let uaux = utmp[1] & KMASK1;
            utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
            utmp[2] = uaux;
            utmp[0] &= KMASK1;

            //extract scales and mins
            LittleEndian::write_u32_into(&utmp[0..2], &mut scales);
            LittleEndian::write_u32_into(&utmp[2..4], &mut mins);

            let mut sumi = 0;
            for j in 0..QK_K / 16 {
                sumi += y.bsums[j] as i32 * mins[j / 2] as i32;
            }

            let mut a = &mut aux8[..];
            let mut q8 = &q8[..];

            let mut aux32 = i32x4_splat(0i32);
            for scale in scales {
                let scale = i32x4_splat(scale as i32);
                for _ in 0..4 {
                    let v_q8 = i16x8_load_extend_i8x8(q8.as_ptr());
                    let v_a = i16x8_load_extend_i8x8(a.as_ptr());
                    let aux16 = i16x8_mul(v_q8, v_a);
                    aux32 = i32x4_add(aux32, i32x4_mul(scale, i32x4_extend_low_i16x8(aux16)));
                    aux32 = i32x4_add(aux32, i32x4_mul(scale, i32x4_extend_high_i16x8(aux16)));
                    q8 = &q8[8..];
                    a = &mut a[8..];
                }
            }
            let aux32 = f32x4_convert_i32x4(aux32);
            let d = f32x4_splat(x.d.to_f32() * y.d);
            sums = f32x4_add(sums, f32x4_mul(aux32, d));
            let dmin = x.dmin.to_f32() * y.d;
            sumf -= dmin * sumi as f32;
        }
        f32x4_extract_lane::<0>(sums)
            + f32x4_extract_lane::<1>(sums)
            + f32x4_extract_lane::<2>(sums)
            + f32x4_extract_lane::<3>(sums)
    };
    Ok(sumf + sums)
}
