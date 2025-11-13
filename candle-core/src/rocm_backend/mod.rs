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

// TEAM-507: Import candle-kernels for CUDA parity
// This gives access to pre-compiled HSACO modules (AFFINE, BINARY, CAST, etc.)
pub use candle_kernels as kernels_module;

mod backend_device;
pub mod device;
pub mod error;
pub mod kernels;
pub mod miopen;
pub mod ops;
pub mod rocblas;
pub mod storage;
pub mod utils;

// Re-exports (matches CUDA backend pattern)
pub use device::{device_count, is_available, runtime_version, RocmDevice};
pub use error::{RocmError, WrapErr};
pub use storage::{RocmStorage, RocmStorageSlice};

// Re-export rocm-rs crate (matches CUDA's `pub use cudarc;`)
// TEAM-498: Added for CUDA parity - allows users to access rocm-rs directly
pub use rocm_rs;

// Type alias for convenience (matches CUDA backend pattern)
pub type S = RocmStorageSlice;
pub type Result<T> = std::result::Result<T, RocmError>;
