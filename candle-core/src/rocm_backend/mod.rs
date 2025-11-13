// candle-core/src/rocm_backend/mod.rs
// Created by: TEAM-488 (Phase 1)
// Updated by: TEAM-489 (Phase 2 Step 3) - Added kernel imports
// ROCm backend using rocm-rs - thin wrappers, don't reimplement

pub mod device;
pub mod error;
pub mod storage_slice;

pub use device::{device_count, is_available, runtime_version, RocmDevice};
pub use error::{Result, RocmError};
pub use storage_slice::RocmStorageSlice;

// Re-export rocm-rs types we use directly
pub use rocm_rs::hip::{Dim3, DeviceMemory, Function, Module, Stream};

// TEAM-489: Import rocm-rs kernel operations
pub use rocm_rs::rocarray::kernels;

// Type alias for convenience (matches CUDA backend pattern)
pub type S = RocmStorageSlice;
