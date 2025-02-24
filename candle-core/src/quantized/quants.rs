use super::GgmlDType;
use crate::Result;
use half::f16;
use rayon::prelude::*;

pub trait GgmlType: Sized + Clone + Send + Sync {
    const DTYPE: GgmlDType;
    const BLCK_SIZE: usize;
    type VecDotType: GgmlType;
    const SUPPORTS_I8MM: bool;

    // This is only safe for types that include immediate values such as float/int/...
    fn zeros() -> Self {
        unsafe { std::mem::MaybeUninit::zeroed().assume_init() }
    }
    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()>;
    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()>;

    /// Dot product used as a building block for quantized mat-mul.
    /// n is the number of elements to be considered.
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32>;

    /// Generic implementation of the dot product without simd optimizations.
    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32>;
}

impl GgmlType for f32 {
    const DTYPE: GgmlDType = GgmlDType::F32;
    const BLCK_SIZE: usize = 1;
    type VecDotType = f32;
    const SUPPORTS_I8MM: bool = false;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
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
    const SUPPORTS_I8MM: bool = false;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
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

    let k_in_lhs_blocks = k.div_ceil(T::BLCK_SIZE);
    let k_in_rhs_blocks = k.div_ceil(T::VecDotType::BLCK_SIZE);
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

        let result: Result<Vec<_>> = dst_row
            .into_par_iter()
            .enumerate()
            .with_min_len(128)
            .with_max_len(512)
            .map(|(col_idx, dst)| {
                let rhs_col = &rhs_t[col_idx * k_in_rhs_blocks..(col_idx + 1) * k_in_rhs_blocks];
                T::vec_dot(k, rhs_col, lhs_row).map(|value| *dst = value)
            })
            .collect();

        result?;
    }
    Ok(())
}
