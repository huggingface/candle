//! TensorScalar Enum and Trait
//!
use crate::{DType, Result, Tensor, WithDType};
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

impl<T: WithDType> From<T> for Scalar {
    fn from(value: T) -> Self {
        value.to_scalar()
    }
}

impl Scalar {
    pub fn zero(dtype: DType) -> Self {
        match dtype {
            DType::U8 => Scalar::U8(0),
            DType::U32 => Scalar::U32(0),
            DType::I64 => Scalar::I64(0),
            DType::BF16 => Scalar::BF16(bf16::ZERO),
            DType::F16 => Scalar::F16(f16::ZERO),
            DType::F32 => Scalar::F32(0.0),
            DType::F64 => Scalar::F64(0.0),
        }
    }

    pub fn one(dtype: DType) -> Self {
        match dtype {
            DType::U8 => Scalar::U8(1),
            DType::U32 => Scalar::U32(1),
            DType::I64 => Scalar::I64(1),
            DType::BF16 => Scalar::BF16(bf16::ONE),
            DType::F16 => Scalar::F16(f16::ONE),
            DType::F32 => Scalar::F32(1.0),
            DType::F64 => Scalar::F64(1.0),
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Scalar::U8(_) => DType::U8,
            Scalar::U32(_) => DType::U32,
            Scalar::I64(_) => DType::I64,
            Scalar::BF16(_) => DType::BF16,
            Scalar::F16(_) => DType::F16,
            Scalar::F32(_) => DType::F32,
            Scalar::F64(_) => DType::F64,
        }
    }

    pub fn to_f64(&self) -> f64 {
        match self {
            Scalar::U8(v) => *v as f64,
            Scalar::U32(v) => *v as f64,
            Scalar::I64(v) => *v as f64,
            Scalar::BF16(v) => v.to_f64(),
            Scalar::F16(v) => v.to_f64(),
            Scalar::F32(v) => *v as f64,
            Scalar::F64(v) => *v,
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
