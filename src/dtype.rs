use crate::CpuStorage;

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
}

impl WithDType for f32 {
    const DTYPE: DType = DType::F32;

    fn to_cpu_storage(data: &[Self]) -> CpuStorage {
        CpuStorage::F32(data.to_vec())
    }
}

impl WithDType for f64 {
    const DTYPE: DType = DType::F64;

    fn to_cpu_storage(data: &[Self]) -> CpuStorage {
        CpuStorage::F64(data.to_vec())
    }
}
