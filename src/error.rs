use crate::{DType, DeviceLocation, Shape};

/// Main library error type.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("unexpected dtype, expected: {expected:?}, got: {got:?}")]
    UnexpectedDType { expected: DType, got: DType },

    #[error("{op} only supports contiguous tensors")]
    RequiresContiguous { op: &'static str },

    #[error("{op} expects at least one tensor")]
    OpRequiresAtLeastOneTensor { op: &'static str },

    #[error("backward is not supported for {op}")]
    BackwardNotSupported { op: &'static str },

    #[error("the candle crate has not been built with cuda support")]
    NotCompiledWithCudaSupport,

    #[error(
        "Shape mismatch, got buffer of size {buffer_size} which is compatible with shape {shape:?}"
    )]
    ShapeMismatch { buffer_size: usize, shape: Shape },

    #[error("shape mismatch in {op}, lhs: {lhs:?}, rhs: {rhs:?}")]
    ShapeMismatchBinaryOp {
        lhs: Shape,
        rhs: Shape,
        op: &'static str,
    },

    #[error("shape mismatch in cat for dim {dim}, shape for arg 1: {first_shape:?} shape for arg {n}: {nth_shape:?}")]
    ShapeMismatchCat {
        dim: usize,
        first_shape: Shape,
        n: usize,
        nth_shape: Shape,
    },

    #[error("device mismatch in {op}, lhs: {lhs:?}, rhs: {rhs:?}")]
    DeviceMismatchBinaryOp {
        lhs: DeviceLocation,
        rhs: DeviceLocation,
        op: &'static str,
    },

    #[error("dtype mismatch in {op}, lhs: {lhs:?}, rhs: {rhs:?}")]
    DTypeMismatchBinaryOp {
        lhs: DType,
        rhs: DType,
        op: &'static str,
    },

    #[error("unexpected rank, expected: {expected}, got: {got} ({shape:?})")]
    UnexpectedNumberOfDims {
        expected: usize,
        got: usize,
        shape: Shape,
    },

    // TODO this is temporary when we support arbitrary matmul
    #[error("temporary error where matmul doesn't support arbitrary striding")]
    UnexpectedStriding,

    #[error(transparent)]
    Cuda(#[from] crate::CudaError),
}

pub type Result<T> = std::result::Result<T, Error>;
