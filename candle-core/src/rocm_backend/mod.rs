// candle-core/src/rocm_backend/mod.rs
// Created by: TEAM-488 (Phase 1)
// Updated by: TEAM-489 (Phase 2 Step 3) - Added kernel imports
// Updated by: TEAM-492 (Phase 2 Step 3) - Direct kernel loading
// Updated by: TEAM-493 (Phase 3) - Added utils module for Map traits
// ROCm backend using rocm-rs - thin wrappers, don't reimplement

pub mod device;
pub mod error;
pub mod kernels;
pub mod storage_slice;
pub mod utils;

pub use device::{device_count, is_available, runtime_version, RocmDevice};
pub use error::RocmError;
pub use storage_slice::RocmStorageSlice;

// Re-export rocm-rs types we use directly
pub use rocm_rs::hip::{Dim3, DeviceMemory, Function, Module, Stream};

// Type alias for convenience (matches CUDA backend pattern)
pub type S = RocmStorageSlice;
pub type Result<T> = std::result::Result<T, RocmError>;
