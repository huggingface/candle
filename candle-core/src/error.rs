use crate::{DType, DeviceLocation, Shape};

/// Main library error type.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("{msg}, expected: {expected:?}, got: {got:?}")]
    UnexpectedDType {
        msg: &'static str,
        expected: DType,
        got: DType,
    },

    #[error("{msg}, expected: {expected:?}, got: {got:?}")]
    UnexpectedShape {
        msg: String,
        expected: Shape,
        got: Shape,
    },

    #[error("{op}: dimension index {dim} out of range for {shape:?}")]
    DimOutOfRange {
        shape: Shape,
        dim: usize,
        op: &'static str,
    },

    #[error("invalid args for narrow: {shape:?}, dim: {dim}, start: {start}, len:{len}")]
    NarrowInvalidArgs {
        shape: Shape,
        dim: usize,
        start: usize,
        len: usize,
    },

    #[error("{op} only supports contiguous tensors")]
    RequiresContiguous { op: &'static str },

    #[error("{op} expects at least one tensor")]
    OpRequiresAtLeastOneTensor { op: &'static str },

    #[error("backward is not supported for {op}")]
    BackwardNotSupported { op: &'static str },

    #[error("{op} invalid index {index} with vocab {vocab_size}")]
    InvalidIndex {
        op: &'static str,
        index: usize,
        vocab_size: usize,
    },

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
    #[error("temporary error where matmul doesn't support arbitrary striding {lhs_stride:?} x {rhs_stride:?}")]
    UnexpectedStriding {
        lhs_stride: Vec<usize>,
        rhs_stride: Vec<usize>,
    },

    #[error(transparent)]
    Cuda(#[from] crate::CudaError),

    #[error(transparent)]
    TryFromIntError(#[from] core::num::TryFromIntError),

    #[error("npy/npz error {0}")]
    Npy(String),

    /// Zip file format error.
    #[error(transparent)]
    Zip(#[from] zip::result::ZipError),

    /// Integer parse error.
    #[error(transparent)]
    ParseInt(#[from] std::num::ParseIntError),

    /// I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// SafeTensor error.
    #[error(transparent)]
    SafeTensor(#[from] safetensors::SafeTensorError),

    #[error("unsupported safetensor dtype {0:?}")]
    UnsupportedSafeTensorDtype(safetensors::Dtype),

    #[error("unsupported dtype {0:?} for op {1}")]
    UnsupportedDTypeForOp(DType, &'static str),

    #[error("cannot broadcast {src_shape:?} to {dst_shape:?}")]
    BroadcastIncompatibleShapes { src_shape: Shape, dst_shape: Shape },

    #[error("matmul is only supported for contiguous tensors lstride: {lhs_stride:?} rstride: {rhs_stride:?} mnk: {mnk:?}")]
    MatMulNonContiguous {
        lhs_stride: Vec<usize>,
        rhs_stride: Vec<usize>,
        mnk: (usize, usize, usize),
    },
}

pub type Result<T> = std::result::Result<T, Error>;
