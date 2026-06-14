/// Helper functions to write CPU kernels.
use crate::backend::BackendStorage;
use crate::nditer::NdIter;
use crate::{Error, Layout, Result, WithDType};

type C = super::CpuStorage;
pub trait Map1 {
    fn f<T: WithDType>(&self, vs: &[T], layout: &Layout) -> Result<Vec<T>>;

    fn map(&self, vs: &C, layout: &Layout) -> Result<C> {
        match vs {
            C::U8(vs) => Ok(C::U8(self.f(vs, layout)?)),
            C::U32(vs) => Ok(C::U32(self.f(vs, layout)?)),
            C::I16(vs) => Ok(C::I16(self.f(vs, layout)?)),
            C::I32(vs) => Ok(C::I32(self.f(vs, layout)?)),
            C::I64(vs) => Ok(C::I64(self.f(vs, layout)?)),
            C::BF16(vs) => Ok(C::BF16(self.f(vs, layout)?)),
            C::F16(vs) => Ok(C::F16(self.f(vs, layout)?)),
            C::F32(vs) => Ok(C::F32(self.f(vs, layout)?)),
            C::F64(vs) => Ok(C::F64(self.f(vs, layout)?)),
            C::F8E4M3(vs) => Ok(C::F8E4M3(self.f(vs, layout)?)),
            // Dummy types don't support Map1 operations
            C::F6E2M3(_) => Err(Error::UnsupportedDTypeForOp(vs.dtype(), "map1").bt()),
            C::F6E3M2(_) => Err(Error::UnsupportedDTypeForOp(vs.dtype(), "map1").bt()),
            C::F4(_) => Err(Error::UnsupportedDTypeForOp(vs.dtype(), "map1").bt()),
            C::F8E8M0(_) => Err(Error::UnsupportedDTypeForOp(vs.dtype(), "map1").bt()),
        }
    }
}

pub trait Map1Any {
    fn f<T: WithDType, W: Fn(Vec<T>) -> C>(&self, vs: &[T], layout: &Layout, wrap: W) -> Result<C>;

    fn map(&self, vs: &C, layout: &Layout) -> Result<C> {
        match vs {
            C::U8(vs) => Ok(self.f(vs, layout, C::U8)?),
            C::U32(vs) => Ok(self.f(vs, layout, C::U32)?),
            C::I16(vs) => Ok(self.f(vs, layout, C::I16)?),
            C::I32(vs) => Ok(self.f(vs, layout, C::I32)?),
            C::I64(vs) => Ok(self.f(vs, layout, C::I64)?),
            C::BF16(vs) => Ok(self.f(vs, layout, C::BF16)?),
            C::F16(vs) => Ok(self.f(vs, layout, C::F16)?),
            C::F32(vs) => Ok(self.f(vs, layout, C::F32)?),
            C::F64(vs) => Ok(self.f(vs, layout, C::F64)?),
            C::F8E4M3(vs) => Ok(self.f(vs, layout, C::F8E4M3)?),
            // Dummy types don't support Map1Any operations
            C::F6E2M3(_) => Err(Error::UnsupportedDTypeForOp(vs.dtype(), "map1any").bt()),
            C::F6E3M2(_) => Err(Error::UnsupportedDTypeForOp(vs.dtype(), "map1any").bt()),
            C::F4(_) => Err(Error::UnsupportedDTypeForOp(vs.dtype(), "map1any").bt()),
            C::F8E8M0(_) => Err(Error::UnsupportedDTypeForOp(vs.dtype(), "map1any").bt()),
        }
    }
}

pub trait Map2 {
    const OP: &'static str;
    fn f<T: WithDType>(&self, v1: &[T], l1: &Layout, v2: &[T], l2: &Layout) -> Result<Vec<T>>;

    fn map(&self, v1: &C, l1: &Layout, v2: &C, l2: &Layout) -> Result<C> {
        match (v1, v2) {
            (C::U8(v1), C::U8(v2)) => Ok(C::U8(self.f(v1, l1, v2, l2)?)),
            (C::U32(v1), C::U32(v2)) => Ok(C::U32(self.f(v1, l1, v2, l2)?)),
            (C::I16(v1), C::I16(v2)) => Ok(C::I16(self.f(v1, l1, v2, l2)?)),
            (C::I32(v1), C::I32(v2)) => Ok(C::I32(self.f(v1, l1, v2, l2)?)),
            (C::I64(v1), C::I64(v2)) => Ok(C::I64(self.f(v1, l1, v2, l2)?)),
            (C::BF16(v1), C::BF16(v2)) => Ok(C::BF16(self.f(v1, l1, v2, l2)?)),
            (C::F16(v1), C::F16(v2)) => Ok(C::F16(self.f(v1, l1, v2, l2)?)),
            (C::F32(v1), C::F32(v2)) => Ok(C::F32(self.f(v1, l1, v2, l2)?)),
            (C::F64(v1), C::F64(v2)) => Ok(C::F64(self.f(v1, l1, v2, l2)?)),
            (C::F8E4M3(v1), C::F8E4M3(v2)) => Ok(C::F8E4M3(self.f(v1, l1, v2, l2)?)),
            _ => Err(Error::DTypeMismatchBinaryOp {
                lhs: v1.dtype(),
                rhs: v2.dtype(),
                op: Self::OP,
            }
            .bt()),
        }
    }
}

pub trait Map2InPlace {
    const OP: &'static str;
    fn f<T: WithDType>(&self, v1: &mut [T], l1: &Layout, v2: &[T], l2: &Layout) -> Result<()>;

    fn map(&self, v1: &mut C, l1: &Layout, v2: &C, l2: &Layout) -> Result<()> {
        match (v1, v2) {
            (C::U8(v1), C::U8(v2)) => self.f(v1, l1, v2, l2)?,
            (C::U32(v1), C::U32(v2)) => self.f(v1, l1, v2, l2)?,
            (C::I16(v1), C::I16(v2)) => self.f(v1, l1, v2, l2)?,
            (C::I32(v1), C::I32(v2)) => self.f(v1, l1, v2, l2)?,
            (C::I64(v1), C::I64(v2)) => self.f(v1, l1, v2, l2)?,
            (C::BF16(v1), C::BF16(v2)) => self.f(v1, l1, v2, l2)?,
            (C::F16(v1), C::F16(v2)) => self.f(v1, l1, v2, l2)?,
            (C::F32(v1), C::F32(v2)) => self.f(v1, l1, v2, l2)?,
            (C::F64(v1), C::F64(v2)) => self.f(v1, l1, v2, l2)?,
            (C::F8E4M3(v1), C::F8E4M3(v2)) => self.f(v1, l1, v2, l2)?,
            (v1, v2) => Err(Error::DTypeMismatchBinaryOp {
                lhs: v1.dtype(),
                rhs: v2.dtype(),
                op: Self::OP,
            }
            .bt())?,
        };
        Ok(())
    }
}

pub trait Map2U8 {
    const OP: &'static str;
    fn f<T: WithDType>(&self, v1: &[T], l1: &Layout, v2: &[T], l2: &Layout) -> Result<Vec<u8>>;

    fn map(&self, v1: &C, l1: &Layout, v2: &C, l2: &Layout) -> Result<C> {
        match (v1, v2) {
            (C::U8(v1), C::U8(v2)) => Ok(C::U8(self.f(v1, l1, v2, l2)?)),
            (C::U32(v1), C::U32(v2)) => Ok(C::U8(self.f(v1, l1, v2, l2)?)),
            (C::I16(v1), C::I16(v2)) => Ok(C::U8(self.f(v1, l1, v2, l2)?)),
            (C::I32(v1), C::I32(v2)) => Ok(C::U8(self.f(v1, l1, v2, l2)?)),
            (C::I64(v1), C::I64(v2)) => Ok(C::U8(self.f(v1, l1, v2, l2)?)),
            (C::BF16(v1), C::BF16(v2)) => Ok(C::U8(self.f(v1, l1, v2, l2)?)),
            (C::F16(v1), C::F16(v2)) => Ok(C::U8(self.f(v1, l1, v2, l2)?)),
            (C::F32(v1), C::F32(v2)) => Ok(C::U8(self.f(v1, l1, v2, l2)?)),
            (C::F64(v1), C::F64(v2)) => Ok(C::U8(self.f(v1, l1, v2, l2)?)),
            (C::F8E4M3(v1), C::F8E4M3(v2)) => Ok(C::U8(self.f(v1, l1, v2, l2)?)),
            _ => Err(Error::DTypeMismatchBinaryOp {
                lhs: v1.dtype(),
                rhs: v2.dtype(),
                op: Self::OP,
            }
            .bt()),
        }
    }
}

pub fn binary_map<T: Copy, U: Copy, F: FnMut(T, T) -> U>(
    lhs_l: &Layout,
    rhs_l: &Layout,
    lhs: &[T],
    rhs: &[T],
    mut f: F,
) -> Vec<U> {
    let el_count = lhs_l.shape().elem_count();
    let mut result: Vec<U> = Vec::with_capacity(el_count);

    let nd_iter = NdIter::new([lhs_l, rhs_l]);
    let inner_size = nd_iter.inner_size;
    let [inner_ls, inner_rs] = nd_iter.inner_strides;

    for [lhs_off, rhs_off] in nd_iter {
        for i in 0..inner_size {
            result.push(f(lhs[lhs_off + i * inner_ls], rhs[rhs_off + i * inner_rs]));
        }
    }

    result
}

// Similar to binary_map but with vectorized variants.
pub fn binary_map_vec<
    T: Copy,
    F: FnMut(T, T) -> T,
    FV: FnMut(&[T], &[T], &mut [T]),
    FSV: FnMut(T, &[T], &mut [T]),
>(
    lhs_l: &Layout,
    rhs_l: &Layout,
    lhs: &[T],
    rhs: &[T],
    mut f: F,
    mut f_vec: FV,
    mut f_scalar_vec: FSV,
) -> Vec<T> {
    let el_count = lhs_l.shape().elem_count();
    let mut ys: Vec<T> = Vec::with_capacity(el_count);
    let ys_to_set = unsafe {
        let s = ys.spare_capacity_mut();
        std::mem::transmute::<&mut [std::mem::MaybeUninit<T>], &mut [T]>(s)
    };

    let nd_iter = NdIter::new([lhs_l, rhs_l]);
    let inner_size = nd_iter.inner_size;
    let [inner_ls, inner_rs] = nd_iter.inner_strides;

    let mut dst_off = 0usize;

    for [lhs_off, rhs_off] in nd_iter {
        match (inner_ls, inner_rs) {
            (1, 1) => f_vec(
                &lhs[lhs_off..lhs_off + inner_size],
                &rhs[rhs_off..rhs_off + inner_size],
                &mut ys_to_set[dst_off..dst_off + inner_size],
            ),
            (1, 0) => {
                let r = rhs[rhs_off];
                f_scalar_vec(
                    r,
                    &lhs[lhs_off..lhs_off + inner_size],
                    &mut ys_to_set[dst_off..dst_off + inner_size],
                );
            }
            (0, 1) => {
                let l = lhs[lhs_off];
                for i in 0..inner_size {
                    ys_to_set[dst_off + i] = f(l, rhs[rhs_off + i]);
                }
            }
            _ => {
                for i in 0..inner_size {
                    ys_to_set[dst_off + i] =
                        f(lhs[lhs_off + i * inner_ls], rhs[rhs_off + i * inner_rs]);
                }
            }
        }
        dst_off += inner_size;
    }

    // SAFETY: all el_count elements have been written in the dispatch loop above.
    unsafe { ys.set_len(el_count) };
    ys
}

pub fn unary_map<T: Copy, U: Copy, F: FnMut(T) -> U>(
    vs: &[T],
    layout: &Layout,
    mut f: F,
) -> Vec<U> {
    match layout.strided_blocks() {
        crate::StridedBlocks::SingleBlock { start_offset, len } => vs
            [start_offset..start_offset + len]
            .iter()
            .map(|&v| f(v))
            .collect(),
        crate::StridedBlocks::UniformBlocks {
            start_offset,
            block_len,
            count,
            src_stride,
        } => {
            let mut result = Vec::with_capacity(count * block_len);
            if block_len == 1 {
                for i in 0..count {
                    let v = unsafe { vs.get_unchecked(start_offset + i * src_stride) };
                    result.push(f(*v))
                }
            } else {
                for i in 0..count {
                    let src_start = start_offset + i * src_stride;
                    for offset in 0..block_len {
                        let v = unsafe { vs.get_unchecked(src_start + offset) };
                        result.push(f(*v))
                    }
                }
            }
            result
        }
        crate::StridedBlocks::MultipleBlocks {
            block_start_index,
            block_len,
        } => {
            let mut result = Vec::with_capacity(layout.shape().elem_count());
            // Specialize the case where block_len is one to avoid the second loop.
            if block_len == 1 {
                for index in block_start_index {
                    let v = unsafe { vs.get_unchecked(index) };
                    result.push(f(*v))
                }
            } else {
                for index in block_start_index {
                    for offset in 0..block_len {
                        let v = unsafe { vs.get_unchecked(index + offset) };
                        result.push(f(*v))
                    }
                }
            }
            result
        }
    }
}

pub fn unary_map_vec<T: Copy, U: Copy, F: FnMut(T) -> U, FV: FnMut(&[T], &mut [U])>(
    vs: &[T],
    layout: &Layout,
    mut f: F,
    mut f_vec: FV,
) -> Vec<U> {
    match layout.strided_blocks() {
        crate::StridedBlocks::SingleBlock { start_offset, len } => {
            let mut ys: Vec<U> = Vec::with_capacity(len);
            let ys_to_set = ys.spare_capacity_mut();
            let ys_to_set = unsafe {
                std::mem::transmute::<&mut [std::mem::MaybeUninit<U>], &mut [U]>(ys_to_set)
            };
            f_vec(&vs[start_offset..start_offset + len], ys_to_set);
            // SAFETY: values are all set by f_vec.
            unsafe { ys.set_len(len) };
            ys
        }
        crate::StridedBlocks::UniformBlocks {
            start_offset,
            block_len,
            count,
            src_stride,
        } => {
            let el_count = count * block_len;
            if block_len == 1 {
                let mut result = Vec::with_capacity(count);
                for i in 0..count {
                    let v = unsafe { vs.get_unchecked(start_offset + i * src_stride) };
                    result.push(f(*v))
                }
                result
            } else {
                let mut ys: Vec<U> = Vec::with_capacity(el_count);
                let ys_to_set = ys.spare_capacity_mut();
                let ys_to_set = unsafe {
                    std::mem::transmute::<&mut [std::mem::MaybeUninit<U>], &mut [U]>(ys_to_set)
                };
                let mut dst_index = 0;
                for i in 0..count {
                    let src_start = start_offset + i * src_stride;
                    f_vec(
                        &vs[src_start..src_start + block_len],
                        &mut ys_to_set[dst_index..dst_index + block_len],
                    );
                    dst_index += block_len;
                }
                // SAFETY: values are all set by f_vec.
                unsafe { ys.set_len(el_count) };
                ys
            }
        }
        crate::StridedBlocks::MultipleBlocks {
            block_start_index,
            block_len,
        } => {
            let el_count = layout.shape().elem_count();
            // Specialize the case where block_len is one to avoid the second loop.
            if block_len == 1 {
                let mut result = Vec::with_capacity(el_count);
                for index in block_start_index {
                    let v = unsafe { vs.get_unchecked(index) };
                    result.push(f(*v))
                }
                result
            } else {
                let mut ys: Vec<U> = Vec::with_capacity(el_count);
                let ys_to_set = ys.spare_capacity_mut();
                let ys_to_set = unsafe {
                    std::mem::transmute::<&mut [std::mem::MaybeUninit<U>], &mut [U]>(ys_to_set)
                };
                let mut dst_index = 0;
                for src_index in block_start_index {
                    let vs = &vs[src_index..src_index + block_len];
                    let ys = &mut ys_to_set[dst_index..dst_index + block_len];
                    f_vec(vs, ys);
                    dst_index += block_len;
                }
                // SAFETY: values are all set by f_vec.
                unsafe { ys.set_len(el_count) };
                ys
            }
        }
    }
}
