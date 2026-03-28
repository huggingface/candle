use crate::DType;
use rocm_rs::hip::error::Error as HipError;

#[derive(Debug, thiserror::Error)]
pub enum RocmError {
    #[error("HIP error: {0}")]
    Hip(#[from] HipError),

    #[error("ROCm kernel not found: {0}")]
    KernelNotFound(String),

    #[error("ROCm module not found: {0}")]
    ModuleNotFound(String),

    #[error("ROCm device not available")]
    DeviceNotAvailable,

    #[error("dtype {0:?} not supported for ROCm")]
    UnsupportedDType(DType),

    #[error("internal error: {0}")]
    Internal(String),
}

impl From<RocmError> for crate::Error {
    fn from(e: RocmError) -> Self {
        crate::Error::Msg(e.to_string())
    }
}

impl From<HipError> for crate::Error {
    fn from(e: HipError) -> Self {
        crate::Error::Msg(format!("HIP error: {}", e))
    }
}

pub trait WrapErr<T> {
    fn w(self) -> crate::Result<T>;
}

impl<T> WrapErr<T> for Result<T, RocmError> {
    fn w(self) -> crate::Result<T> {
        self.map_err(|e| crate::Error::Msg(e.to_string()))
    }
}
