use crate::{CpuStorage, Error, Result};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DType {
    U32,
    BF16,
    F16,
    F32,
    F64,
}

impl DType {
    pub fn size_in_bytes(&self) -> usize {
        match self {
            Self::U32 => 4,
            Self::BF16 => 2,
            Self::F16 => 2,
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

pub trait WithDType: Sized + Copy {
    const DTYPE: DType;

    fn to_cpu_storage_owned(data: Vec<Self>) -> CpuStorage;

    fn to_cpu_storage(data: &[Self]) -> CpuStorage {
        Self::to_cpu_storage_owned(data.to_vec())
    }

    fn cpu_storage_as_slice(s: &CpuStorage) -> Result<&[Self]>;
    fn cpu_storage_as_mut_slice(s: &mut CpuStorage) -> Result<&mut [Self]>;
    fn cpu_storage_data(s: CpuStorage) -> Result<Vec<Self>>;
}

macro_rules! with_dtype {
    ($ty:ty, $dtype:ident) => {
        impl WithDType for $ty {
            const DTYPE: DType = DType::$dtype;

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

            fn cpu_storage_as_mut_slice(s: &mut CpuStorage) -> Result<&mut [Self]> {
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
with_dtype!(u32, U32);
with_dtype!(half::f16, F16);
with_dtype!(half::bf16, BF16);
with_dtype!(f32, F32);
with_dtype!(f64, F64);
