use crate::{CpuStorage, Error, Result};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DType {
    U8,
    U32,
    BF16,
    F16,
    F32,
    F64,
}

#[derive(Debug, PartialEq, Eq)]
pub struct DTypeParseError;

impl std::str::FromStr for DType {
    type Err = DTypeParseError;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "u8" => Ok(Self::U8),
            "u32" => Ok(Self::U32),
            "bf16" => Ok(Self::BF16),
            "f16" => Ok(Self::F16),
            "f32" => Ok(Self::F32),
            "f64" => Ok(Self::F64),
            _ => Err(DTypeParseError),
        }
    }
}

impl DType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::U32 => "u32",
            Self::BF16 => "bf16",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }

    pub fn size_in_bytes(&self) -> usize {
        match self {
            Self::U8 => 4,
            Self::U32 => 4,
            Self::BF16 => 2,
            Self::F16 => 2,
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

pub trait WithDType: Sized + Copy + num_traits::NumAssign + 'static {
    const DTYPE: DType;

    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn to_cpu_storage_owned(data: Vec<Self>) -> CpuStorage;

    fn to_cpu_storage(data: &[Self]) -> CpuStorage {
        Self::to_cpu_storage_owned(data.to_vec())
    }

    fn cpu_storage_as_slice(s: &CpuStorage) -> Result<&[Self]>;
    fn cpu_storage_data(s: CpuStorage) -> Result<Vec<Self>>;
}

macro_rules! with_dtype {
    ($ty:ty, $dtype:ident, $from_f64:expr, $to_f64:expr) => {
        impl WithDType for $ty {
            const DTYPE: DType = DType::$dtype;

            fn from_f64(v: f64) -> Self {
                $from_f64(v)
            }

            fn to_f64(self) -> f64 {
                $to_f64(self)
            }

            fn to_cpu_storage_owned(data: Vec<Self>) -> CpuStorage {
                CpuStorage::$dtype(data)
            }

            fn cpu_storage_data(s: CpuStorage) -> Result<Vec<Self>> {
                match s {
                    CpuStorage::$dtype(data) => Ok(data),
                    _ => Err(Error::UnexpectedDType {
                        expected: DType::$dtype,
                        got: s.dtype(),
                        msg: "unexpected dtype",
                    }),
                }
            }

            fn cpu_storage_as_slice(s: &CpuStorage) -> Result<&[Self]> {
                match s {
                    CpuStorage::$dtype(data) => Ok(data),
                    _ => Err(Error::UnexpectedDType {
                        expected: DType::$dtype,
                        got: s.dtype(),
                        msg: "unexpected dtype",
                    }),
                }
            }
        }
    };
}
use half::{bf16, f16};

with_dtype!(u8, U8, |v: f64| v as u8, |v: u8| v as f64);
with_dtype!(u32, U32, |v: f64| v as u32, |v: u32| v as f64);
with_dtype!(f16, F16, f16::from_f64, f16::to_f64);
with_dtype!(bf16, BF16, bf16::from_f64, bf16::to_f64);
with_dtype!(f32, F32, |v: f64| v as f32, |v: f32| v as f64);
with_dtype!(f64, F64, |v: f64| v, |v: f64| v);
