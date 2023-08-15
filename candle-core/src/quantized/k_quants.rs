use super::GgmlDType;
use crate::Result;
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

pub trait GgmlType: Sized + Clone {
    const DTYPE: GgmlDType;
    const BLCK_SIZE: usize;
    type VecDotType: GgmlType;

    // This is only safe for types that include immediate values such as float/int/...
    fn zeros() -> Self {
        unsafe { std::mem::MaybeUninit::zeroed().assume_init() }
    }
    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()>;
    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()>;

    /// Dot product used as a building block for quantized mat-mul.
    /// n is the number of elements to be considered.
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32>;
}

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ4_0 {
    d: f16,
    qs: [u8; QK4_0 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ4_0>() == 18);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ4_1 {
    d: f16,
    m: f16,
    qs: [u8; QK4_1 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ4_1>() == 20);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ5_0 {
    d: f16,
    qh: [u8; 4],
    qs: [u8; QK5_0 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ5_0>() == 22);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ5_1 {
    d: f16,
    m: f16,
    qh: [u8; 4],
    qs: [u8; QK5_1 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ5_1>() == 24);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ8_0 {
    d: f16,
    qs: [u8; QK8_0],
}
const _: () = assert!(std::mem::size_of::<BlockQ8_0>() == 34);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ8_1 {
    d: f16,
    s: f16,
    qs: [u8; QK8_1],
}
const _: () = assert!(std::mem::size_of::<BlockQ8_1>() == 36);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ2K {
    scales: [u8; QK_K / 16],
    qs: [u8; QK_K / 4],
    d: f16,
    dmin: f16,
}
const _: () = assert!(QK_K / 16 + QK_K / 4 + 2 * 2 == std::mem::size_of::<BlockQ2K>());

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ3K {
    hmask: [u8; QK_K / 8],
    qs: [u8; QK_K / 4],
    scales: [u8; 12],
    d: f16,
}
const _: () = assert!(QK_K / 8 + QK_K / 4 + 12 + 2 == std::mem::size_of::<BlockQ3K>());

#[derive(Debug, Clone, PartialEq)]
// https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/k_quants.h#L82
#[repr(C)]
pub struct BlockQ4K {
    d: f16,
    dmin: f16,
    scales: [u8; K_SCALE_SIZE],
    qs: [u8; QK_K / 2],
}
const _: () = assert!(QK_K / 2 + K_SCALE_SIZE + 2 * 2 == std::mem::size_of::<BlockQ4K>());

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ6K {
    ql: [u8; QK_K / 2],
    qh: [u8; QK_K / 4],
    scales: [i8; QK_K / 16],
    d: f16,
}
const _: () = assert!(3 * QK_K / 4 + QK_K / 16 + 2 == std::mem::size_of::<BlockQ6K>());

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ8K {
    d: f32,
    qs: [i8; QK_K],
    bsums: [i16; QK_K / 16],
}
const _: () = assert!(4 + QK_K + QK_K / 16 * 2 == std::mem::size_of::<BlockQ8K>());

impl GgmlType for BlockQ4_1 {
    const DTYPE: GgmlDType = GgmlDType::Q4_1;
    const BLCK_SIZE: usize = QK4_1;
    type VecDotType = BlockQ8_1;

    fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> Result<f32> {
        todo!()
    }

    fn from_float(_xs: &[f32], _ys: &mut [Self]) -> Result<()> {
        todo!()
    }

    // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1545
    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
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
}

impl GgmlType for BlockQ5_0 {
    const DTYPE: GgmlDType = GgmlDType::Q5_0;
    const BLCK_SIZE: usize = QK5_0;
    type VecDotType = BlockQ8_0;

    fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> Result<f32> {
        todo!()
    }

    fn from_float(_xs: &[f32], _ys: &mut [Self]) -> Result<()> {
        todo!()
    }

    // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1566
    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
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
}

impl GgmlType for BlockQ5_1 {
    const DTYPE: GgmlDType = GgmlDType::Q5_1;
    const BLCK_SIZE: usize = QK5_1;
    type VecDotType = BlockQ8_1;

    fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> Result<f32> {
        todo!()
    }

    fn from_float(_xs: &[f32], _ys: &mut [Self]) -> Result<()> {
        todo!()
    }

    // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1592
    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
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
}

impl GgmlType for BlockQ2K {
    const DTYPE: GgmlDType = GgmlDType::Q2K;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;

    fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> Result<f32> {
        todo!()
    }

    fn from_float(_xs: &[f32], _ys: &mut [Self]) -> Result<()> {
        todo!()
    }
    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L354
    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
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

impl GgmlType for BlockQ4K {
    const DTYPE: GgmlDType = GgmlDType::Q4K;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;

    fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> Result<f32> {
        todo!()
    }

    fn from_float(_xs: &[f32], _ys: &mut [Self]) -> Result<()> {
        todo!()
    }
    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L735
    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
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
}

impl GgmlType for BlockQ3K {
    const DTYPE: GgmlDType = GgmlDType::Q3K;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;

    fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> Result<f32> {
        todo!()
    }

    fn from_float(_xs: &[f32], _ys: &mut [Self]) -> Result<()> {
        todo!()
    }

    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L533
    fn to_float(_xs: &[Self], _ys: &mut [f32]) -> Result<()> {
        todo!()
    }
}

// https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L928
impl GgmlType for BlockQ5K {
    const DTYPE: GgmlDType = GgmlDType::Q5K;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;

    fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> Result<f32> {
        todo!()
    }

    fn from_float(_xs: &[f32], _ys: &mut [Self]) -> Result<()> {
        todo!()
    }
    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
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
}

impl GgmlType for BlockQ6K {
    const DTYPE: GgmlDType = GgmlDType::Q6K;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;

    fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> Result<f32> {
        todo!()
    }

    fn from_float(_xs: &[f32], _ys: &mut [Self]) -> Result<()> {
        todo!()
    }

    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L1067
    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
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
}

impl GgmlType for BlockQ8K {
    const DTYPE: GgmlDType = GgmlDType::Q8K;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;

    fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> Result<f32> {
        todo!()
    }

    fn from_float(_xs: &[f32], _ys: &mut [Self]) -> Result<()> {
        todo!()
    }

    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L533
    fn to_float(_xs: &[Self], _ys: &mut [f32]) -> Result<()> {
        todo!()
    }
}

impl GgmlType for BlockQ4_0 {
    const DTYPE: GgmlDType = GgmlDType::Q4_0;
    const BLCK_SIZE: usize = QK4_0;
    type VecDotType = BlockQ8_0;

    // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1525
    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
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

    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        // quantize_row_q4_0
        let qk = Self::BLCK_SIZE;
        let k = xs.len();
        if k % qk != 0 {
            crate::bail!("{k} is not divisible by {}", qk);
        };
        let nb = k / qk;
        if ys.len() != nb {
            crate::bail!("size mismatch {} {} {}", xs.len(), ys.len(), qk,)
        }
        for (i, ys) in ys.iter_mut().enumerate() {
            let mut amax = 0f32;
            let mut max = 0f32;

            let xs = &xs[i * qk..(i + 1) * qk];
            for &x in xs.iter() {
                if amax < x.abs() {
                    amax = x.abs();
                    max = x;
                }
            }
            let d = max / -8.0;
            let id = if d != 0f32 { 1. / d } else { 0. };
            ys.d = f16::from_f32(d);

            for (j, q) in ys.qs.iter_mut().enumerate() {
                let x0 = xs[j] * id;
                let x1 = xs[qk / 2 + j] * id;
                let xi0 = u8::min(15, (x0 + 8.5) as u8);
                let xi1 = u8::min(15, (x1 + 8.5) as u8);
                *q = xi0 | (xi1 << 4)
            }
        }
        Ok(())
    }

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
                let v0 = (xs[i].qs[j] & 0x0F) as i32 - 8;
                let v1 = (xs[i].qs[j] >> 4) as i32 - 8;
                sum_i += v0 * ys[i].qs[j] as i32 + v1 * ys[i].qs[j + qk / 2] as i32
            }
            sumf += sum_i as f32 * f16::to_f32(xs[i].d) * f16::to_f32(ys[i].d)
        }
        Ok(sumf)
    }
}

impl GgmlType for BlockQ8_0 {
    const DTYPE: GgmlDType = GgmlDType::Q8_0;
    const BLCK_SIZE: usize = QK8_0;
    type VecDotType = BlockQ8_0;

    // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1619
    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        let k = ys.len();
        if k % QK8_0 != 0 {
            crate::bail!("dequantize_row_q8_0: {k} is not divisible by {QK8_0}");
        }

        let nb = k / QK8_0;

        for i in 0..nb {
            let d = xs[i].d.to_f32();

            for j in 0..QK8_0 {
                ys[i * QK8_0 + j] = xs[i].qs[j] as f32 * d;
            }
        }
        Ok(())
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        // quantize_row_q8_0
        let k = xs.len();
        if k % Self::BLCK_SIZE != 0 {
            crate::bail!("{k} is not divisible by {}", Self::BLCK_SIZE);
        };
        let nb = k / Self::BLCK_SIZE;
        if ys.len() != nb {
            crate::bail!(
                "size mismatch {} {} {}",
                xs.len(),
                ys.len(),
                Self::BLCK_SIZE
            )
        }
        for (i, ys) in ys.iter_mut().enumerate() {
            let mut amax = 0f32;
            let xs = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];
            for &x in xs.iter() {
                amax = amax.max(x.abs())
            }
            let d = amax / ((1 << 7) - 1) as f32;
            let id = if d != 0f32 { 1. / d } else { 0. };
            ys.d = f16::from_f32(d);
            for (y, &x) in ys.qs.iter_mut().zip(xs.iter()) {
                *y = f32::round(x * id) as u8
            }
        }
        Ok(())
    }

    fn vec_dot(_: usize, _: &[Self], _: &[Self::VecDotType]) -> Result<f32> {
        todo!()
    }
}

impl GgmlType for BlockQ8_1 {
    const DTYPE: GgmlDType = GgmlDType::Q3K;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8_1;

    fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> Result<f32> {
        todo!()
    }

    fn from_float(_xs: &[f32], _ys: &mut [Self]) -> Result<()> {
        todo!()
    }

    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L533
    fn to_float(_xs: &[Self], _ys: &mut [f32]) -> Result<()> {
        todo!()
    }
}

// https://github.com/ggerganov/llama.cpp/blob/b5ffb2849d23afe73647f68eec7b68187af09be6/ggml.c#L10605
pub fn matmul<T: GgmlType>(
    mkn: (usize, usize, usize),
    lhs: &[f32],
    rhs_t: &[T],
    dst: &mut [f32],
) -> Result<()> {
    let (m, k, n) = mkn;
    if m * k != lhs.len() {
        crate::bail!("unexpected lhs length {} {mkn:?}", lhs.len());
    }

    let k_in_lhs_blocks = (k + T::BLCK_SIZE - 1) / T::BLCK_SIZE;
    let k_in_rhs_blocks = (k + T::VecDotType::BLCK_SIZE - 1) / T::VecDotType::BLCK_SIZE;
    // TODO: Do not make this copy if the DotType is f32.
    // TODO: Pre-allocate this.
    let mut lhs_b = vec![T::VecDotType::zeros(); m * k_in_lhs_blocks];
    for row_idx in 0..m {
        let lhs_b = &mut lhs_b[row_idx * k_in_lhs_blocks..(row_idx + 1) * k_in_lhs_blocks];
        let lhs = &lhs[row_idx * k..(row_idx + 1) * k];
        T::VecDotType::from_float(lhs, lhs_b)?
    }
    let lhs_b = lhs_b.as_slice();

    for row_idx in 0..m {
        let lhs_row = &lhs_b[row_idx * k_in_lhs_blocks..(row_idx + 1) * k_in_lhs_blocks];
        let dst_row = &mut dst[row_idx * n..(row_idx + 1) * n];
        for (col_idx, dst) in dst_row.iter_mut().enumerate() {
            let rhs_col = &rhs_t[col_idx * k_in_rhs_blocks..(col_idx + 1) * k_in_rhs_blocks];
            *dst = T::vec_dot(k, rhs_col, lhs_row)?;
        }
    }
    Ok(())
}

impl GgmlType for f32 {
    const DTYPE: GgmlDType = GgmlDType::F32;
    const BLCK_SIZE: usize = 1;
    type VecDotType = f32;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        if xs.len() < n {
            crate::bail!("size mismatch {} < {n}", xs.len())
        }
        if ys.len() < n {
            crate::bail!("size mismatch {} < {n}", ys.len())
        }
        let mut res = 0f32;
        unsafe { crate::cpu::vec_dot_f32(xs.as_ptr(), ys.as_ptr(), &mut res, n) };
        Ok(res)
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        if xs.len() != ys.len() {
            crate::bail!("size mismatch {} {}", xs.len(), ys.len());
        }
        ys.copy_from_slice(xs);
        Ok(())
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        if xs.len() != ys.len() {
            crate::bail!("size mismatch {} {}", xs.len(), ys.len());
        }
        ys.copy_from_slice(xs);
        Ok(())
    }
}

impl GgmlType for f16 {
    const DTYPE: GgmlDType = GgmlDType::F16;
    const BLCK_SIZE: usize = 1;
    type VecDotType = f16;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        if xs.len() < n {
            crate::bail!("size mismatch {} < {n}", xs.len())
        }
        if ys.len() < n {
            crate::bail!("size mismatch {} < {n}", ys.len())
        }
        let mut res = 0f32;
        unsafe { crate::cpu::vec_dot_f16(xs.as_ptr(), ys.as_ptr(), &mut res, n) };
        Ok(res)
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        if xs.len() != ys.len() {
            crate::bail!("size mismatch {} {}", xs.len(), ys.len());
        }
        // TODO: vectorize
        for (x, y) in xs.iter().zip(ys.iter_mut()) {
            *y = f16::from_f32(*x)
        }
        Ok(())
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        if xs.len() != ys.len() {
            crate::bail!("size mismatch {} {}", xs.len(), ys.len());
        }
        // TODO: vectorize
        for (x, y) in xs.iter().zip(ys.iter_mut()) {
            *y = x.to_f32()
        }
        Ok(())
    }
}
