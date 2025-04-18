//! TensorScalar Enum and Trait
//!
use crate::{Result, Tensor, WithDType};
use half::{bf16, f16};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    U8(u8),
    U32(u32),
    I64(i64),
    BF16(bf16),
    F16(f16),
    F32(f32),
    F64(f64),
}

impl Scalar {
    pub fn dtype(&self) -> crate::DType {
        match self {
            Scalar::U8(_) => crate::DType::U8,
            Scalar::U32(_) => crate::DType::U32,
            Scalar::I64(_) => crate::DType::I64,
            Scalar::BF16(_) => crate::DType::BF16,
            Scalar::F16(_) => crate::DType::F16,
            Scalar::F32(_) => crate::DType::F32,
            Scalar::F64(_) => crate::DType::F64,
        }
    }
}

pub enum TensorScalar {
    Tensor(Tensor),
    Scalar(Tensor),
}

pub trait TensorOrScalar {
    fn to_tensor_scalar(self) -> Result<TensorScalar>;
}

impl TensorOrScalar for &Tensor {
    fn to_tensor_scalar(self) -> Result<TensorScalar> {
        Ok(TensorScalar::Tensor(self.clone()))
    }
}

impl<T: WithDType> TensorOrScalar for T {
    fn to_tensor_scalar(self) -> Result<TensorScalar> {
        let scalar = Tensor::new(self, &crate::Device::Cpu)?;
        Ok(TensorScalar::Scalar(scalar))
    }
}
