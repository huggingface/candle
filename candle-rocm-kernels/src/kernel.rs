//! Kernel sources and types for ROCm operations.

/// Trait for kernel source definitions.
///
/// This trait simplifies the old enum-based approach by using
/// compile-time constants instead of runtime matching.
pub trait KernelSource {
    /// Unique kernel name (used for caching)
    const NAME: &'static str;
    /// The HIP source code
    const CODE: &'static str;
}

/// Binary operations kernel source
pub struct BinaryKernel;
impl KernelSource for BinaryKernel {
    const NAME: &'static str = "binary";
    const CODE: &'static str = include_str!("kernels/binary.hip");
}

/// Unary operations kernel source
pub struct UnaryKernel;
impl KernelSource for UnaryKernel {
    const NAME: &'static str = "unary";
    const CODE: &'static str = include_str!("kernels/unary.hip");
}

/// Affine operations kernel source
pub struct AffineKernel;
impl KernelSource for AffineKernel {
    const NAME: &'static str = "affine";
    const CODE: &'static str = include_str!("kernels/affine.hip");
}

/// Fill operations kernel source
pub struct FillKernel;
impl KernelSource for FillKernel {
    const NAME: &'static str = "fill";
    const CODE: &'static str = include_str!("kernels/fill.hip");
}

/// Reduce operations kernel source
pub struct ReduceKernel;
impl KernelSource for ReduceKernel {
    const NAME: &'static str = "reduce";
    const CODE: &'static str = include_str!("kernels/reduce.hip");
}

/// Binary operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Minimum,
    Maximum,
}

impl BinaryOp {
    /// Get the kernel function name for this operation
    pub fn kernel_name(&self) -> &'static str {
        match self {
            BinaryOp::Add => "badd",
            BinaryOp::Sub => "bsub",
            BinaryOp::Mul => "bmul",
            BinaryOp::Div => "bdiv",
            BinaryOp::Minimum => "bminimum",
            BinaryOp::Maximum => "bmaximum",
        }
    }
}

/// Unary operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Copy,
    Relu,
    Sigmoid,
    Tan,
    Exp,
    Log,
    Sin,
    Cos,
    Sqrt,
    Abs,
    Neg,
    Recip,
    Floor,
    Ceil,
    Round,
    Gelu,
    Silu,
    Erf,
}

impl UnaryOp {
    /// Get the kernel function name for this operation
    pub fn kernel_name(&self) -> &'static str {
        match self {
            UnaryOp::Copy => "ucopy",
            UnaryOp::Relu => "urelu",
            UnaryOp::Sigmoid => "usigmoid",
            UnaryOp::Tan => "utan",
            UnaryOp::Exp => "uexp",
            UnaryOp::Log => "ulog",
            UnaryOp::Sin => "usin",
            UnaryOp::Cos => "ucos",
            UnaryOp::Sqrt => "usqrt",
            UnaryOp::Abs => "uabs",
            UnaryOp::Neg => "uneg",
            UnaryOp::Recip => "urecip",
            UnaryOp::Floor => "ufloor",
            UnaryOp::Ceil => "uceil",
            UnaryOp::Round => "uround",
            UnaryOp::Gelu => "ugelu",
            UnaryOp::Silu => "usilu",
            UnaryOp::Erf => "uerf",
        }
    }
}

/// Data types supported by kernels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    BF16,
    F16,
    F32,
    F64,
    I64,
    U32,
    U8,
}

impl DType {
    /// Get the size of this dtype in bytes
    pub fn size_in_bytes(&self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U32 => 4,
            Self::I64 => 8,
            Self::BF16 => 2,
            Self::F16 => 2,
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

/// Get the dtype suffix for kernel function naming
pub fn dtype_suffix<T: Copy + Send + Sync + 'static>() -> &'static str {
    let type_name = std::any::type_name::<T>();
    if type_name.contains("f32") {
        "f32"
    } else if type_name.contains("f64") {
        "f64"
    } else if type_name.contains("u8") {
        "u8"
    } else if type_name.contains("u32") {
        "u32"
    } else if type_name.contains("i64") {
        "i64"
    } else {
        panic!("Unsupported dtype for kernel: {}", type_name)
    }
}
