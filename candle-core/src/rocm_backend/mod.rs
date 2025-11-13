//! ROCm Backend for Candle
//!
//! ## Module Structure
//! - `device`: ROCm device management
//! - `error`: Error types
//! - `kernels`: HIP kernel launch wrappers
//! - `miopen`: MIOpen operations (conv2d, pooling)
//! - `ops`: Operation structs (Binary, Cmp, Reduce, Unary)
//! - `rocblas`: rocBLAS operations (matmul)
//! - `storage`: Storage types (RocmStorage, RocmStorageSlice)
//! - `utils`: Utility traits

pub mod device;
pub mod error;
pub mod kernels;
pub mod miopen;
pub mod ops;
pub mod rocblas;
pub mod storage;
pub mod utils;

// Re-exports
pub use device::{device_count, is_available, runtime_version, RocmDevice};
pub use error::RocmError;
pub use storage::{RocmStorage, RocmStorageSlice};

// Re-export rocm-rs types we use directly
pub use rocm_rs::hip::{Dim3, DeviceMemory, Function, Module, Stream};

// Type alias for convenience (matches CUDA backend pattern)
pub type S = RocmStorageSlice;
pub type Result<T> = std::result::Result<T, RocmError>;
