//! ROCm Backend for Candle
//!
//! ## Module Structure
//! - `device`: ROCm device management
//! - `error`: Error types
//! - `kernels`: HIP kernel launch wrappers
//! - `miopen`: MIOpen operations (conv2d, pooling)
//! - `ops`: Operation structs (Binary, Cmp, Reduce, Unary)
//! - `rocblas`: rocBLAS operations (matmul)
//! - `storage`: RocmStorage struct and BackendStorage impl
//! - `storage_slice`: Storage slice enum
//! - `utils`: Utility traits
//!
//! Created by: TEAM-496

pub mod device;
pub mod error;
pub mod kernels;
pub mod miopen;
pub mod ops;
pub mod rocblas;
pub mod storage;
pub mod storage_slice;
pub mod utils;

// Re-exports
pub use device::{device_count, is_available, runtime_version, RocmDevice};
pub use error::RocmError;
pub use storage::RocmStorage;
pub use storage_slice::RocmStorageSlice;

// Re-export rocm-rs types we use directly
pub use rocm_rs::hip::{Dim3, DeviceMemory, Function, Module, Stream};

// Type alias for convenience (matches CUDA backend pattern)
pub type S = RocmStorageSlice;
pub type Result<T> = std::result::Result<T, RocmError>;
