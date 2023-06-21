use crate::storage::{BinaryOp, UnaryOp};
use crate::{DType, Error, Result, Shape, StridedIndex};
use ggblas::batched_sgemm;

// TODO: Think about whether we would be better off with a dtype and
// a buffer as an owned slice of bytes.
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

    pub(crate) fn matmul_impl(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_stride: &[usize],
        rhs_stride: &[usize],
    ) -> Result<Self> {
        println!("rhs {rhs:?}");
        println!("lhs_stride {lhs_stride:?}");
        println!("rhs_stride {rhs_stride:?}");
        // todo!("matmul");
        let a_skip: usize = m * k;
        let b_skip: usize = n * k;
        let c_skip: usize = m * n;

        let mut c = Self::F32(vec![0.0; b * m * n]);

        batched_sgemm(
            self.as_slice()?,
            a_skip,
            rhs.as_slice()?,
            b_skip,
            c.as_mut_slice()?,
            c_skip,
            m,
            n,
            k,
            b,
        );
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
        let a = Tensor::from_slice(&data, (2, 2), Device::Cpu)?;
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = Tensor::from_slice(&data, (2, 2), Device::Cpu)?;

        let c = a.matmul(&b)?;
        assert_eq!(c.to_vec2::<f32>()?, &[&[7.0f32, 10.0], &[15.0, 22.0]]);

        let data = vec![1.0f32, 2.0];
        let a = Tensor::from_slice(&data, (2, 1), Device::Cpu)?;
        let data = vec![3.0f32, 4.0];
        let b = Tensor::from_slice(&data, (1, 2), Device::Cpu)?;
        let c = a.matmul(&b)?;
        assert_eq!(c.to_vec2::<f32>()?, &[&[3.0, 4.0], &[6.0, 8.0]]);

        let data: Vec<_> = (0..6).map(|i| i as f32).collect();
        let a = Tensor::from_slice(&data, (2, 3), Device::Cpu)?;
        let data: Vec<_> = (0..6).map(|i| (i + 2) as f32).collect();
        let b = Tensor::from_slice(&data, (3, 2), Device::Cpu)?;
        let c = a.matmul(&b)?;
        assert_eq!(c.to_vec2::<f32>()?, &[&[16., 19.], &[52., 64.]]);

        let data: Vec<_> = (0..12).map(|i| i as f32).collect();
        let a = Tensor::from_slice(&data, (2, 2, 3), Device::Cpu)?;
        let data: Vec<_> = (0..12).map(|i| (i + 2) as f32).collect();
        let b = Tensor::from_slice(&data, (2, 3, 2), Device::Cpu)?;
        let c = a.matmul(&b)?;
        assert_eq!(
            c.to_vec3::<f32>()?,
            &[&[&[16., 19.], &[52., 64.]], &[&[214., 235.], &[304., 334.]]]
        );
        Ok(())
    }
}
