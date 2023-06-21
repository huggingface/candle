use crate::storage::{BinaryOp, UnaryOp};
use crate::{DType, Error, Result, Shape, StridedIndex};

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
