use thiserror::Error;

#[derive(Debug, Error)]
pub enum RocmKernelError {
    #[error("ROCm error: {0}")]
    Rocm(String),
    #[error("Internal error: {0}")]
    Internal(String),
    #[error("Kernel compilation failed: {0}")]
    Compilation(String),
    #[error("Kernel launch failed: {0}")]
    Launch(String),
    #[error("Unsupported dtype: {0:?}")]
    UnsupportedDType(String),
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
}

impl From<rocm_rs::hip::Error> for RocmKernelError {
    fn from(e: rocm_rs::hip::Error) -> Self {
        RocmKernelError::Rocm(e.to_string())
    }
}

pub trait WrapErr<T> {
    fn w(self) -> Result<T, RocmKernelError>;
}

impl<T> WrapErr<T> for Result<T, rocm_rs::hip::Error> {
    fn w(self) -> Result<T, RocmKernelError> {
        self.map_err(|e| e.into())
    }
}
