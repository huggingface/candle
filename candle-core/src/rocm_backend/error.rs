//! The backend's own error type, and the bridge into [`crate::Error`].
//!
//! Everything this backend raises goes out as `crate::Error::Rocm(RocmError)`,
//! built through [`WrapErr::w`] or `?`. Both routes call `.bt()`, so a ROCm
//! failure carries a backtrace like every other backend's, and callers can match
//! on `Error::Rocm(_)` instead of scraping a string. `Error::Rocm` and the
//! variants below that wrap another error are `#[error(transparent)]`, so the
//! rendered message is exactly the one written at the failing call site.

use crate::Layout;
use candle_rocm_kernels::KernelError;
use rocm_rs::hip::error::Error as HipError;

#[derive(Debug, thiserror::Error)]
pub enum RocmError {
    #[error("HIP error: {0}")]
    Hip(#[from] HipError),

    #[error("rocBLAS error: {0}")]
    Rocblas(String),

    #[error("MIOpen error: {0}")]
    MIOpen(String),

    /// Kernel compilation, module loading and function lookup.
    #[error(transparent)]
    Kernel(#[from] KernelError),

    /// A failure this backend detected itself, message already formatted.
    ///
    /// Display is a bare `{0}` rather than a prefix so the specific wording of
    /// the call site survives verbatim; the `Error::Rocm` wrapper is what says
    /// where it came from.
    #[error("{0}")]
    Internal(String),

    // The two layouts are boxed to keep `RocmError` (and every `Result` built on
    // it) small; inline they make this variant ~136 bytes.
    #[error("matmul is only supported for contiguous tensors lstride: {lhs_stride:?} rstride: {rhs_stride:?} mnk: {mnk:?}")]
    MatMulNonContiguous {
        lhs_stride: Box<Layout>,
        rhs_stride: Box<Layout>,
        mnk: (usize, usize, usize),
    },
}

impl From<RocmError> for crate::Error {
    fn from(val: RocmError) -> Self {
        crate::Error::Rocm(Box::new(val)).bt()
    }
}

impl From<HipError> for crate::Error {
    fn from(e: HipError) -> Self {
        RocmError::Hip(e).into()
    }
}

/// A message this backend raised itself, as a `crate::Error`.
///
/// Replaces the `crate::Error::Msg(format!(…))` these call sites used to build:
/// the wording is unchanged, but the error now carries a backtrace and arrives
/// as `Error::Rocm` so it can be matched on.
pub(crate) fn rocm_error(msg: impl Into<String>) -> crate::Error {
    RocmError::Internal(msg.into()).into()
}

pub trait WrapErr<O> {
    fn w(self) -> std::result::Result<O, crate::Error>;
}

/// Generic over the source error rather than fixed to `RocmError`, so a raw
/// `HipError` or `KernelError` can be wrapped with `.w()` directly. This mirrors
/// `cuda_backend`'s `WrapErr`.
impl<O, E: Into<RocmError>> WrapErr<O> for std::result::Result<O, E> {
    fn w(self) -> std::result::Result<O, crate::Error> {
        self.map_err(|e| crate::Error::Rocm(Box::new(e.into())).bt())
    }
}

#[cfg(test)]
mod tests {
    use super::{RocmError, WrapErr};

    /// The whole point of the variant: a ROCm failure must be matchable as one
    /// rather than only readable as a string.
    #[test]
    fn a_rocm_error_arrives_as_the_rocm_variant() {
        let err: crate::Error = RocmError::Internal("boom".to_string()).into();
        assert!(matches!(err, crate::Error::Rocm(_)));
    }

    /// `Internal` renders bare and `Error::Rocm` is transparent, so a message
    /// written at a call site reaches the user unchanged.
    #[test]
    fn the_call_sites_wording_survives_the_wrapping() {
        let err: crate::Error =
            RocmError::Internal("Failed to allocate ROCm memory: out of memory".to_string()).into();
        assert_eq!(
            err.to_string(),
            "Failed to allocate ROCm memory: out of memory"
        );
    }

    /// `w()` is generic over `E: Into<RocmError>`, so it applies to the source
    /// error types directly and not just to a `RocmError` the caller built.
    #[test]
    fn wrap_err_accepts_any_convertible_source() {
        let raw: std::result::Result<(), candle_rocm_kernels::KernelError> =
            Err(candle_rocm_kernels::KernelError::Compilation("nope".into()));
        let err = raw.w().unwrap_err();
        assert!(matches!(err, crate::Error::Rocm(_)));
        assert_eq!(err.to_string(), "Kernel compilation failed: nope");
    }
}
