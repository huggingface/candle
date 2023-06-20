use crate::{CpuStorage, Error, Result};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
}

impl DType {
    pub fn size_in_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

pub trait WithDType: Sized + Copy {
    const DTYPE: DType;

    fn to_cpu_storage(data: &[Self]) -> CpuStorage;

    fn cpu_storage_as_slice(s: &CpuStorage) -> Result<&[Self]>;
}

impl WithDType for f32 {
    const DTYPE: DType = DType::F32;

    fn to_cpu_storage(data: &[Self]) -> CpuStorage {
        CpuStorage::F32(data.to_vec())
    }

    fn cpu_storage_as_slice(s: &CpuStorage) -> Result<&[Self]> {
        match s {
            CpuStorage::F32(data) => Ok(data),
            _ => Err(Error::UnexpectedDType {
                expected: DType::F32,
                got: s.dtype(),
            }),
        }
    }
}

impl WithDType for f64 {
    const DTYPE: DType = DType::F64;

    fn to_cpu_storage(data: &[Self]) -> CpuStorage {
        CpuStorage::F64(data.to_vec())
    }

    fn cpu_storage_as_slice(s: &CpuStorage) -> Result<&[Self]> {
        match s {
            CpuStorage::F64(data) => Ok(data),
            _ => Err(Error::UnexpectedDType {
                expected: DType::F64,
                got: s.dtype(),
            }),
        }
    }
}
