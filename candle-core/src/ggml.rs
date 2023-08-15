//! Support for the GGML file format.

use crate::{DType, Device, Result, Tensor};
use byteorder::{LittleEndian, ReadBytesExt};
use half::f16;

// Default to QK_K 256 rather than 64.
pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = 12;

pub const QK4_0: usize = 32;
pub const QK4_1: usize = 32;
pub const QK5_0: usize = 32;
pub const QK5_1: usize = 32;
pub const QK8_0: usize = 32;
pub const QK8_1: usize = 32;

#[repr(C)]
pub struct BlockQ4_0 {
    d: f16,
    qs: [u8; QK4_0 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ4_0>() == 18);

#[repr(C)]
pub struct BlockQ4_1 {
    d: f16,
    m: f16,
    qs: [u8; QK4_1 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ4_1>() == 20);

#[repr(C)]
pub struct BlockQ5_0 {
    d: f16,
    qh: [u8; 4],
    qs: [u8; QK5_0 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ5_0>() == 22);

#[repr(C)]
pub struct BlockQ5_1 {
    d: f16,
    m: f16,
    qh: [u8; 4],
    qs: [u8; QK5_1 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ5_1>() == 24);

#[repr(C)]
pub struct BlockQ8_0 {
    d: f16,
    qs: [u8; QK8_0],
}
const _: () = assert!(std::mem::size_of::<BlockQ8_0>() == 34);

#[repr(C)]
pub struct BlockQ8_1 {
    d: f16,
    s: f16,
    qs: [u8; QK8_1],
}
const _: () = assert!(std::mem::size_of::<BlockQ8_1>() == 36);

#[repr(C)]
pub struct BlockQ2K {
    scales: [u8; QK_K / 16],
    qs: [u8; QK_K / 4],
    d: f16,
    dmin: f16,
}
const _: () = assert!(QK_K / 16 + QK_K / 4 + 2 * 2 == std::mem::size_of::<BlockQ2K>());

#[repr(C)]
pub struct BlockQ3K {
    hmask: [u8; QK_K / 8],
    qs: [u8; QK_K / 4],
    scales: [u8; 12],
    d: f16,
}
const _: () = assert!(QK_K / 8 + QK_K / 4 + 12 + 2 == std::mem::size_of::<BlockQ3K>());

// https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/k_quants.h#L82
#[repr(C)]
pub struct BlockQ4K {
    d: f16,
    dmin: f16,
    scales: [u8; K_SCALE_SIZE],
    qs: [u8; QK_K / 2],
}
const _: () = assert!(QK_K / 2 + K_SCALE_SIZE + 2 * 2 == std::mem::size_of::<BlockQ4K>());

#[repr(C)]
pub struct BlockQ5K {
    d: f16,
    dmin: f16,
    scales: [u8; K_SCALE_SIZE],
    qh: [u8; QK_K / 8],
    qs: [u8; QK_K / 2],
}
const _: () =
    assert!(QK_K / 8 + QK_K / 2 + 2 * 2 + K_SCALE_SIZE == std::mem::size_of::<BlockQ5K>());

#[repr(C)]
pub struct BlockQ6K {
    ql: [u8; QK_K / 2],
    qh: [u8; QK_K / 4],
    scales: [i8; QK_K / 16],
    d: f16,
}
const _: () = assert!(3 * QK_K / 4 + QK_K / 16 + 2 == std::mem::size_of::<BlockQ6K>());

// https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1525
fn dequantize_row_q4_0(xs: &[BlockQ4_0], ys: &mut [f32]) -> Result<()> {
    let k = ys.len();
    if k % QK4_0 != 0 {
        crate::bail!("dequantize_row_q4_0: {k} is not divisible by {QK4_0}")
    }

    let nb = k / QK4_0;
    for i in 0..nb {
        let d = xs[i].d.to_f32();

        for j in 0..(QK4_0 / 2) {
            let x0 = (xs[i].qs[j] & 0x0F) as i16 - 8;
            let x1 = (xs[i].qs[j] >> 4) as i16 - 8;

            ys[i * QK4_0 + j] = (x0 as f32) * d;
            ys[i * QK4_0 + j + QK4_0 / 2] = (x1 as f32) * d;
        }
    }
    Ok(())
}

// https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1545
fn dequantize_row_q4_1(xs: &[BlockQ4_1], ys: &mut [f32]) -> Result<()> {
    let k = ys.len();
    if k % QK4_1 != 0 {
        crate::bail!("dequantize_row_q4_1: {k} is not divisible by {QK4_1}");
    }

    let nb = k / QK4_1;
    for i in 0..nb {
        let d = xs[i].d.to_f32();
        let m = xs[i].m.to_f32();

        for j in 0..(QK4_1 / 2) {
            let x0 = xs[i].qs[j] & 0x0F;
            let x1 = xs[i].qs[j] >> 4;

            ys[i * QK4_1 + j] = (x0 as f32) * d + m;
            ys[i * QK4_1 + j + QK4_1 / 2] = (x1 as f32) * d + m;
        }
    }
    Ok(())
}

// https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1566
fn dequantize_row_q5_0(xs: &[BlockQ5_0], ys: &mut [f32]) -> Result<()> {
    let k = ys.len();
    if k % QK5_0 != 0 {
        crate::bail!("dequantize_row_q5_0: {k} is not divisible by {QK5_0}");
    }

    let nb = k / QK5_0;
    for i in 0..nb {
        let d = xs[i].d.to_f32();
        let qh: u32 = unsafe { std::mem::transmute_copy(&xs[i].qh) };

        for j in 0..(QK5_0 / 2) {
            let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
            let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;

            let x0 = ((xs[i].qs[j] & 0x0F) | xh_0) as i32 - 16;
            let x1 = ((xs[i].qs[j] >> 4) | xh_1) as i32 - 16;

            ys[i * QK5_0 + j] = (x0 as f32) * d;
            ys[i * QK5_0 + j + QK5_0 / 2] = (x1 as f32) * d;
        }
    }
    Ok(())
}

// https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1592
fn dequantize_row_q5_1(xs: &[BlockQ5_1], ys: &mut [f32]) -> Result<()> {
    let k = ys.len();
    if k % QK5_1 != 0 {
        crate::bail!("dequantize_row_q5_1: {k} is not divisible by {QK5_1}");
    }

    let nb = k / QK5_1;
    for i in 0..nb {
        let d = xs[i].d.to_f32();
        let m = xs[i].m.to_f32();
        let qh: u32 = unsafe { std::mem::transmute_copy(&xs[i].qh) };

        for j in 0..(QK5_1 / 2) {
            let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
            let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;

            let x0 = (xs[i].qs[j] & 0x0F) | xh_0;
            let x1 = (xs[i].qs[j] >> 4) | xh_1;

            ys[i * QK5_1 + j] = (x0 as f32) * d + m;
            ys[i * QK5_1 + j + QK5_1 / 2] = (x1 as f32) * d + m;
        }
    }
    Ok(())
}

// https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1619
fn dequantize_row_q8_0(vx: &[BlockQ8_0], ys: &mut [f32]) -> Result<()> {
    let k = ys.len();
    if k % QK8_0 != 0 {
        crate::bail!("dequantize_row_q8_0: {k} is not divisible by {QK8_0}");
    }

    let nb = k / QK8_0;
    let xs: &[BlockQ8_0] = unsafe { std::mem::transmute(vx) };

    for i in 0..nb {
        let d = xs[i].d.to_f32();

        for j in 0..QK8_0 {
            ys[i * QK8_0 + j] = xs[i].qs[j] as f32 * d;
        }
    }
    Ok(())
}

// https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L354
fn dequantize_row_q2k(xs: &[BlockQ2K], ys: &mut [f32]) -> Result<()> {
    let k = ys.len();
    if k % QK_K != 0 {
        crate::bail!("dequantize_row_q2k: {k} is not divisible by {QK_K}")
    }
    let mut ys_index = 0;
    for x in xs {
        let d = x.d.to_f32();
        let min = x.dmin.to_f32();
        let q = &x.qs;

        let mut is = 0;
        for n in (0..QK_K).step_by(128) {
            // Step by 32 over q.
            let q = &q[n / 4..];
            let mut shift = 0;
            for _j in 0..4 {
                let sc = x.scales[is];
                is += 1;
                let dl = d * (sc & 0xF) as f32;
                let ml = min * (sc >> 4) as f32;
                for q in &q[..16] {
                    let y = dl * ((q >> shift) & 3) as i8 as f32 - ml;
                    ys[ys_index] = y;
                    ys_index += 1;
                }

                let sc = x.scales[is];
                is += 1;
                let dl = d * (sc & 0xF) as f32;
                let ml = min * (sc >> 4) as f32;
                for q in &q[16..32] {
                    let y = dl * ((q >> shift) & 3) as i8 as f32 - ml;
                    ys[ys_index] = y;
                    ys_index += 1;
                }

                shift += 2;
            }
        }
    }
    Ok(())
}

fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        let d = q[j] & 63;
        let m = q[j + 4] & 63;
        (d, m)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}
// https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L735
fn dequantize_row_q4k(xs: &[BlockQ4K], ys: &mut [f32]) -> Result<()> {
    let k = ys.len();
    if k % QK_K != 0 {
        crate::bail!("dequantize_row_q4k: {k} is not divisible by {QK_K}")
    }
    let mut ys_index = 0;
    for x in xs.iter() {
        let d = x.d.to_f32();
        let min = x.dmin.to_f32();
        let q = &x.qs;
        let mut is = 0;
        for j in (0..QK_K).step_by(64) {
            let q = &q[j / 2..j / 2 + 32];
            let (sc, m) = get_scale_min_k4(is, &x.scales);
            let d1 = d * sc as f32;
            let m1 = min * m as f32;
            let (sc, m) = get_scale_min_k4(is + 1, &x.scales);
            let d2 = d * sc as f32;
            let m2 = min * m as f32;
            for q in q {
                let y = d1 * (q & 0xF) as f32 - m1;
                ys[ys_index] = y;
                ys_index += 1;
            }
            for q in q {
                let y = d2 * (q >> 4) as f32 - m2;
                ys[ys_index] = y;
                ys_index += 1;
            }
            is += 2;
        }
    }
    Ok(())
}

// https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L533
fn dequantize_row_q3k(_xs: &[BlockQ3K], _ys: &mut [f32]) -> Result<()> {
    todo!()
}

// https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L928
fn dequantize_row_q5k(xs: &[BlockQ5K], ys: &mut [f32]) -> Result<()> {
    let k = ys.len();
    if k % QK_K != 0 {
        crate::bail!("dequantize_row_q5k: {k} is not divisible by {QK_K}")
    }
    let mut ys_index = 0;
    for x in xs.iter() {
        let d = x.d.to_f32();
        let min = x.dmin.to_f32();
        let ql = &x.qs;
        let qh = &x.qh;
        let mut is = 0;
        let mut u1 = 1;
        let mut u2 = 2;
        for j in (0..QK_K).step_by(64) {
            let ql = &ql[j / 2..j / 2 + 32];
            let (sc, m) = get_scale_min_k4(is, &x.scales);
            let d1 = d * sc as f32;
            let m1 = min * m as f32;
            let (sc, m) = get_scale_min_k4(is + 1, &x.scales);
            let d2 = d * sc as f32;
            let m2 = min * m as f32;
            for (ql, qh) in ql.iter().zip(qh) {
                let to_add = if qh & u1 != 0 { 16 } else { 1 };
                let y = d1 * ((ql & 0xF) + to_add) as f32 - m1;
                ys[ys_index] = y;
                ys_index += 1;
            }
            for (ql, qh) in ql.iter().zip(qh) {
                let to_add = if qh & u2 != 0 { 16 } else { 1 };
                let y = d2 * ((ql >> 4) + to_add) as f32 - m2;
                ys[ys_index] = y;
                ys_index += 1;
            }
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    Ok(())
}

// https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L1067
fn dequantize_row_q6k(xs: &[BlockQ6K], ys: &mut [f32]) -> Result<()> {
    let k = ys.len();
    if k % QK_K != 0 {
        crate::bail!("dequantize_row_q6k: {k} is not divisible by {QK_K}")
    }
    for x in xs.iter() {
        let d = x.d.to_f32();
        let ql = &x.ql;
        let qh = &x.qh;
        let sc = &x.scales;
        for n in (0..QK_K).step_by(128) {
            let idx = n / 128;
            let ys = &mut ys[n..];
            let sc = &sc[8 * idx..];
            let ql = &ql[64 * idx..];
            let qh = &qh[32 * idx..];
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32;
                let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
                ys[l] = d * sc[is] as f32 * q1 as f32;
                ys[l + 32] = d * sc[is + 2] as f32 * q2 as f32;
                ys[l + 64] = d * sc[is + 4] as f32 * q3 as f32;
                ys[l + 96] = d * sc[is + 6] as f32 * q4 as f32;
            }
        }
    }
    Ok(())
}

pub trait GgmlType: Sized {
    const DTYPE: GgmlDType;
    const BLCK_SIZE: usize;

    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()>;
    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()>;

    type VecDotType: GgmlType;
    // Dot product used as a building block for quantized mat-mul.
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32>;
}

impl GgmlType for BlockQ4_0 {
    const DTYPE: GgmlDType = GgmlDType::Q4_0;
    const BLCK_SIZE: usize = QK8_0;

    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        dequantize_row_q4_0(xs, ys)
    }

    fn from_float(_: &[f32], _: &mut [Self]) -> Result<()> {
        todo!()
    }

    type VecDotType = BlockQ8_0;

    // https://github.com/ggerganov/llama.cpp/blob/b5ffb2849d23afe73647f68eec7b68187af09be6/ggml.c#L2361C10-L2361C122
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
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
        for i in 0..nb {
            let mut sum_i = 0;
            for j in 0..qk / 2 {
                let v0 = (xs[i].qs[j] & 0x0F) - 8;
                let v1 = (xs[i].qs[j] >> 4) - 8;
                sum_i += v0 * ys[i].qs[j] + v1 * ys[i].qs[j + qk / 2]
            }
            sumf += sum_i as f32 * f16::to_f32(xs[i].d) * f16::to_f32(ys[i].d)
        }
        Ok(sumf)
    }
}

impl GgmlType for BlockQ8_0 {
    const DTYPE: GgmlDType = GgmlDType::Q8_0;
    const BLCK_SIZE: usize = QK8_0;

    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        dequantize_row_q8_0(xs, ys)
    }

    fn from_float(_: &[f32], _: &mut [Self]) -> Result<()> {
        todo!()
    }

    type VecDotType = BlockQ8_0;

    fn vec_dot(_: usize, _: &[Self], _: &[Self::VecDotType]) -> Result<f32> {
        todo!()
    }
}

const BLCK0: usize = 16;
const BLCK1: usize = 16;

// This implementation is in-line with the ggml one and keeps the same variable names.
// https://github.com/ggerganov/llama.cpp/blob/b5ffb2849d23afe73647f68eec7b68187af09be6/ggml.c#L10605
pub fn forward_mul_mat<T: GgmlType>(src0: &[T], src1: &[f32], dst: &mut [f32]) -> Result<()> {
    let (ne00, ne01, ne02, ne03) = (1, 1, 1, 1);
    let (ne10, ne11, ne12, ne13) = (1, 1, 1, 1);
    // The strides are in bytes in ggml, however we use the number of elements in candle.
    let (_, nb1, nb2, nb3) = (1, 1, 1, 1);
    let (_, nb01, nb02, nb03) = (1, 1, 1, 1);
    let (_, nb11, nb12, nb13) = (1, 1, 1, 1);

    let nr0 = ne01; // src0 rows
    let nr1 = ne11 * ne12 * ne13;

    // TODO: Either add multi-threading or remove these bits.
    let ir010 = 0;
    let ir011 = nr0;
    let ir110 = 0;
    let ir111 = nr1;
    let r2 = ne12 / ne02;
    let r3 = ne13 / ne03;

    // TODO: Pre-allocate this.
    let wdata = &mut [];
    if ne10 % T::BLCK_SIZE != 0 {
        crate::bail!(
            "forward_mul_mat: ne10 {ne10} is not divisible by block size {}",
            T::BLCK_SIZE
        )
    }
    let row_size = ne10 / T::BLCK_SIZE;
    for i13 in 0..ne13 {
        for i12 in 0..ne12 {
            for i11 in 0..ne11 {
                let wdata_idx = i11 + i12 * ne11 + i13 * ne11 * ne12;
                let wdata = &mut wdata[wdata_idx..wdata_idx + row_size];
                let src1 = &src1[i13 * nb13 + i12 * nb12 + i11 * nb11..];
                T::VecDotType::from_float(src1, wdata)?
            }
        }
    }
    for iir1 in (ir110..ir111).step_by(BLCK1) {
        for iir0 in (ir010..ir011).step_by(BLCK0) {
            for ir1 in iir1..usize::min(iir1 + BLCK1, ir111) {
                let i13 = ir1 / (ne12 * ne11);
                let i12 = (ir1 - i13 * ne12 * ne11) / ne11;
                let i11 = ir1 - i13 * ne12 * ne11 - i12 * ne11;

                let i03 = i13 / r3;
                let i02 = i12 / r2;

                let i1 = i11;
                let i2 = i12;
                let i3 = i13;

                let src0_row = &src0[i02 * nb02 + i03 * nb03..];
                let src1_col = &wdata[(i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size..];
                let dst_col = &mut dst[i1 * nb1 + i2 * nb2 + i3 * nb3..];
                for ir0 in iir0..usize::min(iir0 + BLCK0, ir011) {
                    let src0_row = &src0_row[ir0 * nb01..];
                    let v = T::vec_dot(ne00, src0_row, src1_col)?;
                    dst_col[ir0 - iir0] += v
                }
            }
        }
    }
    Ok(())
}

// https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/llama.h#L37
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Magic {
    Ggjt,
    Ggla,
    Ggmf,
    Ggml,
    Ggsn,
}

impl TryFrom<u32> for Magic {
    type Error = crate::Error;
    fn try_from(value: u32) -> Result<Self> {
        let magic = match value {
            0x67676a74 => Self::Ggjt,
            0x67676c61 => Self::Ggla,
            0x67676d66 => Self::Ggmf,
            0x67676d6c => Self::Ggml,
            0x6767736e => Self::Ggsn,
            _ => crate::bail!("unknown magic {value:08x}"),
        };
        Ok(magic)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersionedMagic {
    GgmlUnversioned,
    GgmfV1,
    GgjtV1,
    GgjtV2,
    GgjtV3,
}

impl VersionedMagic {
    fn read<R: std::io::Read>(reader: &mut R) -> Result<Self> {
        let magic = reader.read_u32::<LittleEndian>()?;
        let magic = Magic::try_from(magic)?;
        if magic == Magic::Ggml {
            return Ok(Self::GgmlUnversioned);
        }
        let version = reader.read_u32::<LittleEndian>()?;
        let versioned_magic = match (magic, version) {
            (Magic::Ggmf, 1) => Self::GgmfV1,
            (Magic::Ggjt, 1) => Self::GgjtV1,
            (Magic::Ggjt, 2) => Self::GgjtV2,
            (Magic::Ggjt, 3) => Self::GgjtV3,
            _ => crate::bail!("ggml: unsupported magic/version {magic:?}/{version}"),
        };
        Ok(versioned_magic)
    }

    fn align32(&self) -> bool {
        match self {
            Self::GgmlUnversioned | Self::GgmfV1 => false,
            Self::GgjtV1 | Self::GgjtV2 | Self::GgjtV3 => true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HParams {
    pub n_vocab: u32,
    pub n_embd: u32,
    pub n_mult: u32,
    pub n_head: u32,
    pub n_layer: u32,
    pub n_rot: u32,
    pub ftype: u32,
}

impl HParams {
    fn read<R: std::io::Read>(reader: &mut R) -> Result<Self> {
        let n_vocab = reader.read_u32::<LittleEndian>()?;
        let n_embd = reader.read_u32::<LittleEndian>()?;
        let n_mult = reader.read_u32::<LittleEndian>()?;
        let n_head = reader.read_u32::<LittleEndian>()?;
        let n_layer = reader.read_u32::<LittleEndian>()?;
        let n_rot = reader.read_u32::<LittleEndian>()?;
        let ftype = reader.read_u32::<LittleEndian>()?;
        Ok(Self {
            n_vocab,
            n_embd,
            n_mult,
            n_head,
            n_layer,
            n_rot,
            ftype,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Vocab {
    pub token_score_pairs: Vec<(Vec<u8>, f32)>,
}

impl Vocab {
    fn read<R: std::io::Read>(reader: &mut R, n_vocab: usize) -> Result<Self> {
        // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/llama.cpp#L556
        let mut token_score_pairs = Vec::with_capacity(n_vocab);
        for _index in 0..n_vocab {
            let len = reader.read_u32::<LittleEndian>()? as usize;
            let mut word = vec![0u8; len];
            reader.read_exact(&mut word)?;
            let score = reader.read_f32::<LittleEndian>()?;
            token_score_pairs.push((word, score))
        }
        Ok(Self { token_score_pairs })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlDType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
}

impl GgmlDType {
    fn from_u32(u: u32) -> Result<Self> {
        let dtype = match u {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            _ => crate::bail!("unknown dtype for tensor {u}"),
        };
        Ok(dtype)
    }

    fn type_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_0 => std::mem::size_of::<BlockQ4_0>(),
            Self::Q4_1 => std::mem::size_of::<BlockQ4_1>(),
            Self::Q5_0 => std::mem::size_of::<BlockQ5_0>(),
            Self::Q5_1 => std::mem::size_of::<BlockQ5_1>(),
            // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L932
            Self::Q8_0 => std::mem::size_of::<BlockQ8_0>(),
            Self::Q8_1 => std::mem::size_of::<BlockQ8_1>(),
            Self::Q2K => std::mem::size_of::<BlockQ2K>(),
            Self::Q3K => std::mem::size_of::<BlockQ3K>(),
            Self::Q4K => std::mem::size_of::<BlockQ4K>(),
            Self::Q5K => std::mem::size_of::<BlockQ5K>(),
            Self::Q6K => std::mem::size_of::<BlockQ6K>(),
        }
    }

    fn blck_size(&self) -> usize {
        match self {
            Self::F32 => 1,
            Self::F16 => 1,
            Self::Q4_0 => QK4_0,
            Self::Q4_1 => QK4_1,
            Self::Q5_0 => QK5_0,
            Self::Q5_1 => QK5_1,
            Self::Q8_0 => QK8_0,
            Self::Q8_1 => QK8_1,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K => QK_K,
        }
    }
}

fn dequantize_and_create_tensor<T, F>(
    raw_data: &[u8],
    tensor_elems: usize,
    size_in_bytes: usize,
    dims: Vec<usize>,
    device: &Device,
    dequantize_row: F,
) -> Result<Tensor>
where
    F: Fn(&[T], &mut [f32]) -> Result<()>,
{
    let mut f32_data = vec![0f32; tensor_elems];
    let raw_data_ptr = raw_data.as_ptr();
    let n_blocks = size_in_bytes / std::mem::size_of::<T>();
    let raw_data = unsafe { std::slice::from_raw_parts(raw_data_ptr as *const T, n_blocks) };
    dequantize_row(raw_data, &mut f32_data)?;
    Tensor::from_vec(f32_data, dims, device)
}

/// Creates a [Tensor] from a raw GGML tensor.
pub fn tensor_from_ggml(
    ggml_dtype: GgmlDType,
    raw_data: &[u8],
    dims: Vec<usize>,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let tensor_elems = dims.iter().product::<usize>();
    let size_in_bytes = tensor_elems * ggml_dtype.type_size() / ggml_dtype.blck_size();

    let tensor = match ggml_dtype {
        GgmlDType::F32 => Tensor::from_raw_buffer(raw_data, DType::F32, &dims, device),
        GgmlDType::F16 => Tensor::from_raw_buffer(raw_data, DType::F16, &dims, device),
        GgmlDType::Q4_0 => dequantize_and_create_tensor(
            raw_data,
            tensor_elems,
            size_in_bytes,
            dims,
            device,
            dequantize_row_q4_0,
        ),
        GgmlDType::Q4_1 => dequantize_and_create_tensor(
            raw_data,
            tensor_elems,
            size_in_bytes,
            dims,
            device,
            dequantize_row_q4_1,
        ),
        GgmlDType::Q5_0 => dequantize_and_create_tensor(
            raw_data,
            tensor_elems,
            size_in_bytes,
            dims,
            device,
            dequantize_row_q5_0,
        ),
        GgmlDType::Q5_1 => dequantize_and_create_tensor(
            raw_data,
            tensor_elems,
            size_in_bytes,
            dims,
            device,
            dequantize_row_q5_1,
        ),
        GgmlDType::Q8_0 => dequantize_and_create_tensor(
            raw_data,
            tensor_elems,
            size_in_bytes,
            dims,
            device,
            dequantize_row_q8_0,
        ),
        GgmlDType::Q2K => dequantize_and_create_tensor(
            raw_data,
            tensor_elems,
            size_in_bytes,
            dims,
            device,
            dequantize_row_q2k,
        ),
        GgmlDType::Q3K => dequantize_and_create_tensor(
            raw_data,
            tensor_elems,
            size_in_bytes,
            dims,
            device,
            dequantize_row_q3k,
        ),
        GgmlDType::Q4K => dequantize_and_create_tensor(
            raw_data,
            tensor_elems,
            size_in_bytes,
            dims,
            device,
            dequantize_row_q4k,
        ),
        GgmlDType::Q5K => dequantize_and_create_tensor(
            raw_data,
            tensor_elems,
            size_in_bytes,
            dims,
            device,
            dequantize_row_q5k,
        ),
        GgmlDType::Q6K => dequantize_and_create_tensor(
            raw_data,
            tensor_elems,
            size_in_bytes,
            dims,
            device,
            dequantize_row_q6k,
        ),

        _ => crate::bail!("quantized type {dtype:?} is not supported yet"),
    }?;
    //We only have ggml-quant to f32 conversions, meaning we have to convert to the desired type
    if tensor.dtype() != dtype {
        tensor.to_dtype(dtype)
    } else {
        Ok(tensor)
    }
}

fn read_one_tensor<R: std::io::Seek + std::io::Read>(
    reader: &mut R,
    magic: VersionedMagic,
    dtype: DType,
    device: &Device,
) -> Result<(String, Tensor)> {
    let n_dims = reader.read_u32::<LittleEndian>()?;
    let name_len = reader.read_u32::<LittleEndian>()?;
    let ggml_dtype = reader.read_u32::<LittleEndian>()?;
    let ggml_dtype = GgmlDType::from_u32(ggml_dtype)?;
    let mut dims = vec![0u32; n_dims as usize];
    reader.read_u32_into::<LittleEndian>(&mut dims)?;
    let mut name = vec![0u8; name_len as usize];
    reader.read_exact(&mut name)?;
    let name = String::from_utf8_lossy(&name).into_owned();

    if magic.align32() {
        let pos = reader.stream_position()?;
        reader.seek(std::io::SeekFrom::Current(((32 - pos % 32) % 32) as i64))?;
    }
    let dims = dims.iter().map(|&u| u as usize).collect::<Vec<_>>();
    let tensor_elems = dims.iter().product::<usize>();
    let size_in_bytes = tensor_elems * ggml_dtype.type_size() / ggml_dtype.blck_size();
    println!("{name} {ggml_dtype:?} {dims:?}");
    // TODO: Mmap version to avoid copying the data around?
    let mut raw_data = vec![0u8; size_in_bytes];
    reader.read_exact(&mut raw_data)?;
    match tensor_from_ggml(ggml_dtype, &raw_data, dims, dtype, device) {
        Ok(tensor) => Ok((name, tensor)),
        Err(e) => crate::bail!("Error creating tensor {name}: {e}"),
    }
}

#[derive(Debug)]
pub struct Content {
    pub magic: VersionedMagic,
    pub hparams: HParams,
    pub vocab: Vocab,
    pub tensors: Vec<(String, Tensor)>,
}

impl Content {
    pub fn read<R: std::io::Seek + std::io::Read>(
        reader: &mut R,
        dtype: DType,
        device: &Device,
    ) -> Result<Content> {
        // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/llama.cpp#L505
        let last_position = reader.seek(std::io::SeekFrom::End(0))?;
        reader.seek(std::io::SeekFrom::Start(0))?;
        let magic = VersionedMagic::read(reader)?;
        let hparams = HParams::read(reader)?;
        let vocab = Vocab::read(reader, hparams.n_vocab as usize)?;
        let mut tensors = vec![];

        while reader.stream_position()? != last_position {
            let (name, tensor) = read_one_tensor(reader, magic, dtype, device)?;
            tensors.push((name, tensor))
        }
        Ok(Self {
            magic,
            hparams,
            vocab,
            tensors,
        })
    }
}
