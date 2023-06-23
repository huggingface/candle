use crate::op::{BinaryOp, UnaryOp};
use crate::{DType, Error, Result, Shape, StridedIndex};
use gemm::{gemm, Parallelism};

// TODO: Think about whether we would be better off with a dtype and
// a buffer as an owned slice of bytes.
// TODO: Maybe we should not implement [Clone] here and instead have an explicit allocator +
// intercept the oom errors to avoid panicking and provide a proper error.
#[derive(Debug, Clone)]
pub enum CpuStorage {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl CpuStorage {
    pub fn dtype(&self) -> DType {
        match self {
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

    pub(crate) fn affine_impl(
        &self,
        shape: &Shape,
        stride: &[usize],
        mul: f64,
        add: f64,
    ) -> Result<Self> {
        match self {
            Self::F32(storage) => {
                let index = StridedIndex::new(shape.dims(), stride);
                let mul = mul as f32;
                let add = add as f32;
                let data = index.map(|i| storage[i] * mul + add).collect();
                Ok(Self::F32(data))
            }
            Self::F64(storage) => {
                let index = StridedIndex::new(shape.dims(), stride);
                let data = index.map(|i| storage[i] * mul + add).collect();
                Ok(Self::F64(data))
            }
        }
    }

    pub(crate) fn unary_impl<B: UnaryOp>(&self, shape: &Shape, stride: &[usize]) -> Result<Self> {
        // TODO: Different code path for the contiguous case?
        match self {
            Self::F32(storage) => {
                let index = StridedIndex::new(shape.dims(), stride);
                let data = index.map(|i| B::f32(storage[i])).collect();
                Ok(Self::F32(data))
            }
            Self::F64(storage) => {
                let index = StridedIndex::new(shape.dims(), stride);
                let data = index.map(|i| B::f64(storage[i])).collect();
                Ok(Self::F64(data))
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
        // The ggml implementation has different paths based on whether the rhs is contiguous
        // or not, for now we only consider the general case but we should benchmark and do the
        // same if it helps.
        // https://github.com/ggerganov/llama.cpp/blob/aacdbd40562684665b6f7b8ba6695b7a2088bbb0/ggml.c#L7895
        match (self, rhs) {
            (Self::F32(lhs), Self::F32(rhs)) => {
                let lhs_index = StridedIndex::new(shape.dims(), lhs_stride);
                let rhs_index = StridedIndex::new(shape.dims(), rhs_stride);
                let data = lhs_index
                    .zip(rhs_index)
                    .map(|(lhs_i, rhs_i)| B::f32(lhs[lhs_i], rhs[rhs_i]))
                    .collect();
                Ok(Self::F32(data))
            }
            (Self::F64(lhs), Self::F64(rhs)) => {
                let lhs_index = StridedIndex::new(shape.dims(), lhs_stride);
                let rhs_index = StridedIndex::new(shape.dims(), rhs_stride);
                let data = lhs_index
                    .zip(rhs_index)
                    .map(|(lhs_i, rhs_i)| B::f64(lhs[lhs_i], rhs[rhs_i]))
                    .collect();
                Ok(Self::F64(data))
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
