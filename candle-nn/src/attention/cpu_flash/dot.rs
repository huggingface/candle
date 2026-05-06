//! Shared dot-product primitives for CPU flash-attention kernels.

use half::{bf16, f16};

/// Dot product of f32 q and `Self` k, accumulating in f32.
pub trait DotF32: Sized {
    fn dot_f32(q: &[f32], k: &[Self]) -> f32;
}

impl DotF32 for f32 {
    #[inline(always)]
    fn dot_f32(q: &[f32], k: &[f32]) -> f32 {
        let n = q.len();
        let mut s = 0.0f32;
        let mut i = 0;
        while i + 4 <= n {
            s += q[i] * k[i]
                + q[i + 1] * k[i + 1]
                + q[i + 2] * k[i + 2]
                + q[i + 3] * k[i + 3];
            i += 4;
        }
        while i < n {
            s += q[i] * k[i];
            i += 1;
        }
        s
    }
}

impl DotF32 for f16 {
    #[inline(always)]
    fn dot_f32(q: &[f32], k: &[f16]) -> f32 {
        let n = q.len();
        let mut s = 0.0f32;
        let mut i = 0;
        while i + 8 <= n {
            s += q[i] * k[i].to_f32()
                + q[i + 1] * k[i + 1].to_f32()
                + q[i + 2] * k[i + 2].to_f32()
                + q[i + 3] * k[i + 3].to_f32()
                + q[i + 4] * k[i + 4].to_f32()
                + q[i + 5] * k[i + 5].to_f32()
                + q[i + 6] * k[i + 6].to_f32()
                + q[i + 7] * k[i + 7].to_f32();
            i += 8;
        }
        while i < n {
            s += q[i] * k[i].to_f32();
            i += 1;
        }
        s
    }
}

impl DotF32 for bf16 {
    #[inline(always)]
    fn dot_f32(q: &[f32], k: &[bf16]) -> f32 {
        let n = q.len();
        let mut s = 0.0f32;
        let mut i = 0;
        while i + 8 <= n {
            s += q[i] * k[i].to_f32()
                + q[i + 1] * k[i + 1].to_f32()
                + q[i + 2] * k[i + 2].to_f32()
                + q[i + 3] * k[i + 3].to_f32()
                + q[i + 4] * k[i + 4].to_f32()
                + q[i + 5] * k[i + 5].to_f32()
                + q[i + 6] * k[i + 6].to_f32()
                + q[i + 7] * k[i + 7].to_f32();
            i += 8;
        }
        while i < n {
            s += q[i] * k[i].to_f32();
            i += 1;
        }
        s
    }
}

impl DotF32 for f64 {
    #[inline(always)]
    fn dot_f32(q: &[f32], k: &[f64]) -> f32 {
        let n = q.len();
        let mut s = 0.0f64;
        let mut i = 0;
        while i + 4 <= n {
            s += (q[i] as f64) * k[i]
                + (q[i + 1] as f64) * k[i + 1]
                + (q[i + 2] as f64) * k[i + 2]
                + (q[i + 3] as f64) * k[i + 3];
            i += 4;
        }
        while i < n {
            s += (q[i] as f64) * k[i];
            i += 1;
        }
        s as f32
    }
}
