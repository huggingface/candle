// candle-core/src/rocm_backend/error.rs
// Created by: TEAM-488 (Phase 1)
// Updated by: TEAM-492 (Phase 2 Step 3) - Added KernelError
// ROCm error handling - wraps rocm-rs errors

use thiserror::Error;

#[derive(Debug, Error)]
pub enum RocmError {
    #[error("ROCm HIP error: {0}")]
    Hip(#[from] rocm_rs::hip::Error),

    #[error("ROCm BLAS error: {0}")]
    Blas(#[from] rocm_rs::rocblas::Error),

    #[error("Kernel error: {0}")]
    KernelError(String),

    #[error("Device {0} not found")]
    DeviceNotFound(usize),

    #[error("Out of memory: requested {requested} bytes")]
    OutOfMemory { requested: usize },

    #[error("Invalid device ID: {0}")]
    InvalidDeviceId(usize),

    #[error("Size mismatch: expected {expected}, got {actual}")]
    SizeMismatch { expected: usize, actual: usize },

    #[error("ROCm error: {0}")]
    Other(String),
}

impl From<RocmError> for crate::Error {
    fn from(err: RocmError) -> Self {
        crate::Error::Msg(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, RocmError>;
