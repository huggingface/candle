use crate::{DType, Device};

// TODO: Think about whether we would be better off with a dtype and
// a buffer as an owned slice of bytes.
pub(crate) enum CpuStorage {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl CpuStorage {
    pub(crate) fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
        }
    }
}

#[allow(dead_code)]
pub(crate) enum Storage {
    Cpu(CpuStorage),
}

impl Storage {
    pub(crate) fn device(&self) -> Device {
        match self {
            Self::Cpu(_) => Device::Cpu,
        }
    }

    pub(crate) fn dtype(&self) -> DType {
        match self {
            Self::Cpu(storage) => storage.dtype(),
        }
    }
}
