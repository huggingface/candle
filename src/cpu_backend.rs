use crate::op::{BinaryOp, UnaryOp};
use crate::{DType, Error, Result, Shape, StridedIndex};
use gemm::{gemm, Parallelism};

// TODO: Think about whether we would be better off with a dtype and
// a buffer as an owned slice of bytes.
// TODO: Maybe we should not implement [Clone] here and instead have an explicit allocator +
// intercept the oom errors to avoid panicking and provide a proper error.
#[derive(Debug, Clone)]
pub enum CpuStorage {
    U32(Vec<u32>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

fn unary_map<T: Copy, U: Copy, F: FnMut(T) -> U>(
    shape: &Shape,
    stride: &[usize],
    vs: &[T],
    mut f: F,
) -> Vec<U> {
    if shape.is_contiguous(stride) {
        vs[..shape.elem_count()].iter().map(|&v| f(v)).collect()
    } else {
        StridedIndex::new(shape.dims(), stride)
            .map(|i| f(vs[i]))
            .collect()
    }
}

// This function maps over two strided index sequences. It supports broadcasting in case
// `lhs_stride` or `rhs_stride` has a length shorter than `shape`.
fn binary_map<T: Copy, F: FnMut(T, T) -> T>(
    shape: &Shape,
    lhs_stride: &[usize],
    rhs_stride: &[usize],
    lhs: &[T],
    rhs: &[T],
    mut f: F,
) -> Vec<T> {
    let dims = shape.dims();
    let broadcast_ldims = dims.len() - lhs_stride.len();
    let broadcast_rdims = dims.len() - rhs_stride.len();
    let elem_count = shape.elem_count();
    if broadcast_ldims == 0 && broadcast_rdims == 0 {
        if shape.is_contiguous(lhs_stride) && shape.is_contiguous(rhs_stride) {
            (0..shape.elem_count()).map(|i| f(lhs[i], rhs[i])).collect()
        } else {
            let lhs_index = StridedIndex::new(dims, lhs_stride);
            let rhs_index = StridedIndex::new(dims, rhs_stride);
            lhs_index
                .zip(rhs_index)
                .map(|(lhs_i, rhs_i)| f(lhs[lhs_i], rhs[rhs_i]))
                .collect()
        }
    } else if broadcast_rdims == 0 {
        let mut res = Vec::new();
        res.reserve(elem_count);
        let lhs_v: Vec<T> = StridedIndex::new(dims, lhs_stride)
            .map(|i| lhs[i])
            .collect();
        let mut i = 0;
        for rhs_i in StridedIndex::new(dims, rhs_stride) {
            res.push(f(lhs_v[i], rhs[rhs_i]));
            i += 1;
            if i >= lhs_v.len() {
                i = 0
            }
        }
        res
    } else if broadcast_ldims == 0 {
        let mut res = Vec::new();
        res.reserve(elem_count);
        let rhs_v: Vec<T> = StridedIndex::new(dims, rhs_stride)
            .map(|i| rhs[i])
            .collect();
        let mut i = 0;
        for lhs_i in StridedIndex::new(dims, lhs_stride) {
            res.push(f(lhs[lhs_i], rhs_v[i]));
            i += 1;
            if i >= rhs_v.len() {
                i = 0
            }
        }
        res
    } else {
        panic!("unexpected broadcasting dims: {shape:?} {lhs_stride:?} {rhs_stride:?}")
    }
}

impl CpuStorage {
    pub fn dtype(&self) -> DType {
        match self {
            Self::U32(_) => DType::U32,
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
        }
    }

    pub fn as_slice<D: crate::WithDType>(&self) -> Result<&[D]> {
        D::cpu_storage_as_slice(self)
    }

    pub fn as_mut_slice<D: crate::WithDType>(&mut self) -> Result<&mut [D]> {
        D::cpu_storage_as_mut_slice(self)
    }

    pub(crate) fn to_dtype(&self, shape: &Shape, stride: &[usize], dtype: DType) -> Result<Self> {
        // TODO: find a way around the quadratic number of cases below.
        match (self, dtype) {
            (Self::U32(storage), DType::F32) => {
                let data = unary_map(shape, stride, storage, |v| v as f32);
                Ok(Self::F32(data))
            }
            (Self::F32(storage), DType::F32) => {
                let data = unary_map(shape, stride, storage, |v| v);
                Ok(Self::F32(data))
            }
            (Self::F64(storage), DType::F32) => {
                let data = unary_map(shape, stride, storage, |v| v as f32);
                Ok(Self::F32(data))
            }
            (Self::U32(storage), DType::U32) => {
                let data = unary_map(shape, stride, storage, |v| v);
                Ok(Self::U32(data))
            }
            (Self::F32(storage), DType::U32) => {
                let data = unary_map(shape, stride, storage, |v| v as u32);
                Ok(Self::U32(data))
            }
            (Self::F64(storage), DType::U32) => {
                let data = unary_map(shape, stride, storage, |v| v as u32);
                Ok(Self::U32(data))
            }
            (Self::U32(storage), DType::F64) => {
                let data = unary_map(shape, stride, storage, |v| v as f64);
                Ok(Self::F64(data))
            }
            (Self::F32(storage), DType::F64) => {
                let data = unary_map(shape, stride, storage, |v| v as f64);
                Ok(Self::F64(data))
            }
            (Self::F64(storage), DType::F64) => {
                let data = unary_map(shape, stride, storage, |v| v);
                Ok(Self::F64(data))
            }
        }
    }

    pub(crate) fn divide_by_sum_over_dim(&mut self, shape: &Shape, dim: usize) -> Result<()> {
        // [self] stores data in a contiguous way.
        let dims = shape.dims();
        let elem_per_slice = dims[dim];
        let prod_pre_dim = dims[..dim].iter().product();
        let prod_post_dim = dims[dim + 1..].iter().product();
        match self {
            Self::F32(storage) => {
                for pre_idx in 0..prod_pre_dim {
                    for post_idx in 0..prod_post_dim {
                        let mut sum = 0f64;
                        let mut idx = pre_idx * prod_post_dim * elem_per_slice + post_idx;
                        for _ in 0..elem_per_slice {
                            sum += storage[idx] as f64;
                            idx += prod_post_dim
                        }
                        let sum = sum as f32;
                        let mut idx = pre_idx * prod_post_dim * elem_per_slice + post_idx;
                        for _ in 0..elem_per_slice {
                            storage[idx] /= sum;
                            idx += prod_post_dim
                        }
                    }
                }
            }
            Self::F64(storage) => {
                for pre_idx in 0..prod_pre_dim {
                    for post_idx in 0..prod_post_dim {
                        let mut sum = 0f64;
                        let mut idx = pre_idx * prod_post_dim * elem_per_slice + post_idx;
                        for _ in 0..elem_per_slice {
                            sum += storage[idx];
                            idx += prod_post_dim
                        }
                        let mut idx = pre_idx * prod_post_dim * elem_per_slice + post_idx;
                        for _ in 0..elem_per_slice {
                            storage[idx] /= sum;
                            idx += prod_post_dim
                        }
                    }
                }
            }
            Self::U32(_) => {}
        }
        Ok(())
    }

    pub(crate) fn affine_impl(
        &self,
        shape: &Shape,
        stride: &[usize],
        mul: f64,
        add: f64,
    ) -> Result<Self> {
        match self {
            Self::U32(storage) => {
                let mul = mul as u32;
                let add = add as u32;
                let data = unary_map(shape, stride, storage, |v| v * mul + add);
                Ok(Self::U32(data))
            }
            Self::F32(storage) => {
                let mul = mul as f32;
                let add = add as f32;
                let data = unary_map(shape, stride, storage, |v| v * mul + add);
                Ok(Self::F32(data))
            }
            Self::F64(storage) => {
                let data = unary_map(shape, stride, storage, |v| v * mul + add);
                Ok(Self::F64(data))
            }
        }
    }

    pub(crate) fn unary_impl<B: UnaryOp>(&self, shape: &Shape, stride: &[usize]) -> Result<Self> {
        match self {
            Self::F32(storage) => {
                let data = unary_map(shape, stride, storage, B::f32);
                Ok(Self::F32(data))
            }
            Self::F64(storage) => {
                let data = unary_map(shape, stride, storage, B::f64);
                Ok(Self::F64(data))
            }
            Self::U32(storage) => {
                let data = unary_map(shape, stride, storage, B::u32);
                Ok(Self::U32(data))
            }
        }
    }

    pub(crate) fn binary_impl<B: BinaryOp>(
        &self,
        rhs: &Self,
        shape: &Shape,
        lhs_stride: &[usize],
        rhs_stride: &[usize],
    ) -> Result<Self> {
        match (self, rhs) {
            (Self::F32(lhs), Self::F32(rhs)) => {
                let data = binary_map(shape, lhs_stride, rhs_stride, lhs, rhs, B::f32);
                Ok(Self::F32(data))
            }
            (Self::F64(lhs), Self::F64(rhs)) => {
                let data = binary_map(shape, lhs_stride, rhs_stride, lhs, rhs, B::f64);
                Ok(Self::F64(data))
            }
            (Self::U32(lhs), Self::U32(rhs)) => {
                let data = binary_map(shape, lhs_stride, rhs_stride, lhs, rhs, B::u32);
                Ok(Self::U32(data))
            }
            _ => {
                // This should be covered by the dtype check above.
                Err(Error::DTypeMismatchBinaryOp {
                    lhs: self.dtype(),
                    rhs: rhs.dtype(),
                    op: B::NAME,
                })
            }
        }
    }

    pub(crate) fn copy_strided_src(
        &self,
        dst: &mut Self,
        src_shape: &Shape,
        src_stride: &[usize],
        dst_offset: usize,
    ) -> Result<()> {
        if src_shape.rank() != src_stride.len() {
            panic!("incoherent shape and strides {src_shape:?} {src_stride:?}")
        }
        match (self, dst) {
            (Self::F32(src), Self::F32(dst)) => {
                if src_shape.is_contiguous(src_stride) {
                    dst[dst_offset..].copy_from_slice(src)
                } else {
                    let src_indexes = StridedIndex::new(src_shape.dims(), src_stride);
                    for (dst_index, src_index) in src_indexes.enumerate() {
                        dst[dst_index + dst_offset] = src[src_index]
                    }
                }
            }
            (Self::F64(src), Self::F64(dst)) => {
                if src_shape.is_contiguous(src_stride) {
                    dst[dst_offset..].copy_from_slice(src)
                } else {
                    let src_indexes = StridedIndex::new(src_shape.dims(), src_stride);
                    for (dst_index, src_index) in src_indexes.enumerate() {
                        dst[dst_index + dst_offset] = src[src_index]
                    }
                }
            }
            (_, dst) => {
                // This should be covered by the dtype check above.
                return Err(Error::DTypeMismatchBinaryOp {
                    lhs: self.dtype(),
                    rhs: dst.dtype(),
                    op: "copy_strided",
                });
            }
        }
        Ok(())
    }

    pub(crate) fn embedding_impl(
        &self,
        rhs: &Self,
        hidden_size: usize,
        vocab_size: usize,
    ) -> Result<Self> {
        match self {
            CpuStorage::U32(lhs) => match rhs {
                CpuStorage::F32(rhs) => {
                    let mut weights = Vec::with_capacity(lhs.len() * hidden_size);
                    for &index in lhs {
                        let index: usize = index.try_into()?;
                        if index >= vocab_size {
                            return Err(Error::InvalidIndex {
                                index,
                                vocab_size,
                                op: "embedding",
                            });
                        } else {
                            weights.extend(&rhs[hidden_size * index..hidden_size * (index + 1)]);
                        }
                    }
                    Ok(CpuStorage::F32(weights))
                }
                CpuStorage::F64(rhs) => {
                    let mut weights = Vec::with_capacity(lhs.len() * hidden_size);
                    for &index in lhs {
                        let index: usize = index.try_into()?;
                        if index >= vocab_size {
                            return Err(Error::InvalidIndex {
                                index,
                                vocab_size,
                                op: "embedding",
                            });
                        } else {
                            weights.extend(&rhs[hidden_size * index..hidden_size * (index + 1)]);
                        }
                    }
                    Ok(CpuStorage::F64(weights))
                }
                rhs => Err(Error::UnexpectedDType {
                    expected: DType::F32,
                    got: rhs.dtype(),
                }),
            },
            lhs => Err(Error::UnexpectedDType {
                expected: DType::U32,
                got: lhs.dtype(),
            }),
        }
    }

    pub(crate) fn matmul_impl(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_stride: &[usize],
        rhs_stride: &[usize],
    ) -> Result<Self> {
        let a_skip: usize = m * k;
        let b_skip: usize = n * k;
        let c_skip: usize = m * n;

        let rank = lhs_stride.len();
        let lhs_cs = lhs_stride[rank - 1];
        let lhs_rs = lhs_stride[rank - 2];

        let rhs_cs = rhs_stride[rank - 1];
        let rhs_rs = rhs_stride[rank - 2];

        if lhs_stride.len() > 2 {
            let lhs_batch_stride = &lhs_stride[..rank - 2];
            let rhs_batch_stride = &rhs_stride[..rank - 2];

            if lhs_batch_stride != [a_skip] || rhs_batch_stride != [b_skip] {
                // Temporary error before we support abitrary striding.
                return Err(Error::UnexpectedStriding);
            }
        }

        let mut dst = vec![0.0; b * m * n];

        let dst_shape: Shape = (m, n).into();
        let dst_strides = dst_shape.stride_contiguous();
        let dst_rs = dst_strides[0];
        let dst_cs = dst_strides[1];

        for step in 0..b {
            let lhs_p = &self.as_slice::<f32>()?[step * a_skip..];
            let rhs_p = &rhs.as_slice::<f32>()?[step * b_skip..];
            let dst_p = &mut dst[step * c_skip..];
            unsafe {
                gemm(
                    // m: usize,
                    m,
                    // n: usize,
                    n,
                    // k: usize,
                    k,
                    // dst: *mut T,
                    dst_p.as_mut_ptr(),
                    // dst_cs: isize,
                    dst_cs as isize,
                    // dst_rs: isize,
                    dst_rs as isize,
                    // read_dst: bool,
                    false,
                    // lhs: *const T,
                    lhs_p.as_ptr(),
                    // lhs_cs: isize,
                    lhs_cs as isize,
                    // lhs_rs: isize,
                    lhs_rs as isize,
                    // rhs: *const T,
                    rhs_p.as_ptr(),
                    // rhs_cs: isize,
                    rhs_cs as isize,
                    // rhs_rs: isize,
                    rhs_rs as isize,
                    // alpha: T,
                    1.0,
                    // beta: T,
                    1.0,
                    // conj_dst: bool,
                    false,
                    // conj_lhs: bool,
                    false,
                    // conj_rhs: bool,
                    true,
                    // parallelism: Parallelism
                    Parallelism::None,
                )
            }
        }

        let c = Self::F32(dst);
        Ok(c)
    }

    pub(crate) fn ones_impl(shape: &Shape, dtype: DType) -> Self {
        let elem_count = shape.elem_count();
        match dtype {
            DType::U32 => {
                let data = vec![1u32; elem_count];
                Self::U32(data)
            }
            DType::F32 => {
                let data = vec![1f32; elem_count];
                Self::F32(data)
            }
            DType::F64 => {
                let data = vec![1f64; elem_count];
                Self::F64(data)
            }
        }
    }

    pub(crate) fn zeros_impl(shape: &Shape, dtype: DType) -> Self {
        let elem_count = shape.elem_count();
        match dtype {
            DType::U32 => {
                let data = vec![0u32; elem_count];
                Self::U32(data)
            }
            DType::F32 => {
                let data = vec![0f32; elem_count];
                Self::F32(data)
            }
            DType::F64 => {
                let data = vec![0f64; elem_count];
                Self::F64(data)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Device, Tensor};

    #[test]
    fn simple_matmul() -> Result<()> {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let a = Tensor::from_slice(&data, (2, 2), &Device::Cpu)?;
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = Tensor::from_slice(&data, (2, 2), &Device::Cpu)?;

        let c = a.matmul(&b)?;
        assert_eq!(c.to_vec2::<f32>()?, &[&[7.0f32, 10.0], &[15.0, 22.0]]);

        let data = vec![1.0f32, 2.0];
        let a = Tensor::from_slice(&data, (2, 1), &Device::Cpu)?;
        let data = vec![3.0f32, 4.0];
        let b = Tensor::from_slice(&data, (1, 2), &Device::Cpu)?;
        let c = a.matmul(&b)?;
        assert_eq!(c.to_vec2::<f32>()?, &[&[3.0, 4.0], &[6.0, 8.0]]);

        let data: Vec<_> = (0..6).map(|i| i as f32).collect();
        let a = Tensor::from_slice(&data, (2, 3), &Device::Cpu)?;
        let data: Vec<_> = (0..6).map(|i| (i + 2) as f32).collect();
        let b = Tensor::from_slice(&data, (3, 2), &Device::Cpu)?;
        let c = a.matmul(&b)?;
        assert_eq!(c.to_vec2::<f32>()?, &[&[16., 19.], &[52., 64.]]);

        let data: Vec<_> = (0..12).map(|i| i as f32).collect();
        let a = Tensor::from_slice(&data, (2, 2, 3), &Device::Cpu)?;
        let data: Vec<_> = (0..12).map(|i| (i + 2) as f32).collect();
        let b = Tensor::from_slice(&data, (2, 3, 2), &Device::Cpu)?;
        let c = a.matmul(&b)?;
        assert_eq!(
            c.to_vec3::<f32>()?,
            &[&[&[16., 19.], &[52., 64.]], &[&[214., 235.], &[304., 334.]]]
        );
        Ok(())
    }
}
