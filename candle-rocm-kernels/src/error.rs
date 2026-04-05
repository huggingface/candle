use thiserror::Error;

#[derive(Debug, Error)]
pub enum KernelError {
    #[error("ROCm error: {0}")]
    Rocm(String),
    #[error("IO error: {0}")]
    Io(String),
    #[error("Kernel compilation failed: {0}")]
    Compilation(String),
    #[error("Kernel launch failed: {0}")]
    Launch(String),
    #[error("Internal error: {0}")]
    Internal(String),
    #[error("Unsupported dtype: {0}")]
    UnsupportedDType(String),
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
}

impl From<rocm_rs::hip::Error> for KernelError {
    fn from(e: rocm_rs::hip::Error) -> Self {
        KernelError::Rocm(e.to_string())
    }
}
