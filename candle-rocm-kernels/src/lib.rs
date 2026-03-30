pub mod cache;
pub mod error;
pub mod kernels;
pub mod manager;
pub mod source;
pub mod utils;

pub use cache::CacheManager;
pub use error::RocmKernelError;
pub use kernels::binary::{BinaryKernels, BinaryOp};
pub use kernels::unary::{UnaryKernels, UnaryOp};
pub use manager::KernelManager;
pub use utils::BufferOffset;

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
