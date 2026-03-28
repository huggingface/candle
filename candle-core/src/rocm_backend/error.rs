use rocm_rs::hip::error::Error as HipError;

/// Error wrapper for ROCm operations
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

impl From<crate::Error> for RocmError {
    fn from(e: crate::Error) -> Self {
        RocmError::Internal(e.to_string())
    }
}

/// Helper trait to convert errors
pub trait WrapErr<T> {
    fn w(self) -> Result<T, crate::Error>;
}

impl<T> WrapErr<T> for Result<T, RocmError> {
    fn w(self) -> Result<T, crate::Error> {
        self.map_err(|e| crate::Error::Msg(e.to_string()))
    }
}

use crate::DType;
