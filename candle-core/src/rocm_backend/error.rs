use crate::{DType, Layout};
use rocm_rs::hip::error::Error as HipError;

#[derive(Debug, thiserror::Error)]
pub enum RocmError {
    #[error("HIP error: {0}")]
    Hip(#[from] HipError),

    #[error("rocBLAS error: {0}")]
    Rocblas(String),

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

    #[error("matmul is only supported for contiguous tensors lstride: {lhs_stride:?} rstride: {rhs_stride:?} mnk: {mnk:?}")]
    MatMulNonContiguous {
        lhs_stride: Layout,
        rhs_stride: Layout,
        mnk: (usize, usize, usize),
    },
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
