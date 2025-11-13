//! ROCm error handling
//! Created by: TEAM-488 (Phase 1 - Device integration)
//! Updated by: TEAM-492 (Kernel operations)
//! Updated by: TEAM-498 (CUDA parity - comprehensive error types)
//! CUDA parity: cuda_backend/error.rs

use crate::{DType, Layout};
use thiserror::Error;

/// ROCm backend errors (matches CUDA backend pattern)
#[derive(Debug, Error)]
pub enum RocmError {
    // ===== rocm-rs library errors (transparent wrapping) =====
    
    #[error(transparent)]
    Hip(#[from] rocm_rs::hip::Error),

    #[error(transparent)]
    Blas(#[from] rocm_rs::rocblas::Error),

    #[error(transparent)]
    MIOpen(#[from] rocm_rs::miopen::Error),

    // ROCm-specific libraries (CUDA doesn't have equivalents)
    // Uncomment when we use them:
    //
    // #[error(transparent)]
    // Rand(#[from] rocm_rs::rocrand::Error),  // rocRAND (random number generation)
    //
    // #[error(transparent)]
    // Fft(#[from] rocm_rs::rocfft::Error),    // rocFFT (Fast Fourier Transform)
    //
    // #[error(transparent)]
    // Sparse(#[from] rocm_rs::rocsparse::Error),  // rocSPARSE (sparse linear algebra)

    // ===== Kernel and module errors =====
    
    #[error("missing kernel '{module_name}'")]
    MissingKernel { module_name: String },

    #[error("kernel error: {0}")]
    KernelError(String),

    #[error("{hip} when loading {module_name}")]
    Load {
        hip: rocm_rs::hip::Error,
        module_name: String,
    },

    // ===== Type and operation errors =====
    
    #[error("unsupported dtype {dtype:?} for {op}")]
    UnsupportedDtype { dtype: DType, op: &'static str },

    #[error("{msg}, expected: {expected:?}, got: {got:?}")]
    UnexpectedDType {
        msg: &'static str,
        expected: DType,
        got: DType,
    },

    // ===== Matrix multiplication errors =====
    
    #[error("matmul is only supported for contiguous tensors lstride: {lhs_stride:?} rstride: {rhs_stride:?} mnk: {mnk:?}")]
    MatMulNonContiguous {
        lhs_stride: Layout,
        rhs_stride: Layout,
        mnk: (usize, usize, usize),
    },

    // ===== Device and memory errors =====
    
    #[error("device {0} not found")]
    DeviceNotFound(usize),

    #[error("invalid device ID: {0}")]
    InvalidDeviceId(usize),

    #[error("out of memory: requested {requested} bytes")]
    OutOfMemory { requested: usize },

    // ===== Shape and size errors =====
    
    #[error("size mismatch: expected {expected}, got {actual}")]
    SizeMismatch { expected: usize, actual: usize },

    #[error("shape mismatch: {msg}")]
    ShapeMismatch { msg: String },

    // ===== Internal errors =====
    
    #[error("internal error '{0}'")]
    InternalError(&'static str),

    #[error("ROCm error: {0}")]
    Other(String),
}

impl From<RocmError> for crate::Error {
    fn from(val: RocmError) -> Self {
        crate::Error::Msg(val.to_string())
    }
}

/// Helper trait for wrapping errors (matches CUDA's WrapErr pattern)
pub trait WrapErr<O> {
    fn w(self) -> std::result::Result<O, crate::Error>;
}

impl<O, E: Into<RocmError>> WrapErr<O> for std::result::Result<O, E> {
    fn w(self) -> std::result::Result<O, crate::Error> {
        self.map_err(|e| crate::Error::Msg(e.into().to_string()))
    }
}

pub type Result<T> = std::result::Result<T, RocmError>;
