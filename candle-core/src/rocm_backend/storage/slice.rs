//! ROCm storage slice enum
//! Created by: TEAM-488 (Phase 1 - Device integration)
//! Modified by: TEAM-491-496 (Kernel work and backend implementation)
//! CUDA parity verified by: TEAM-497
//! Arc-based memory: TEAM-500
//!
//! Holds device memory for different tensor dtypes.
//!
//! ## Architectural Note:
//! TEAM-500: Arc-based memory sharing implemented!
//! ROCm now uses Arc internally (like CUDA), making clone() cheap.
//! Backwards compatibility maintained via Arc::make_mut() for mutable operations.

use float8::F8E4M3;
use half::{bf16, f16};
use rocm_rs::hip::DeviceMemory;

// Created by: TEAM-488 | CUDA parity verified by: TEAM-497 (cuda_backend/mod.rs:66-75)
/// ROCm storage slice for different data types
#[derive(Debug)]
pub enum RocmStorageSlice {
    U8(DeviceMemory<u8>),
    U32(DeviceMemory<u32>),
    I64(DeviceMemory<i64>),
    BF16(DeviceMemory<bf16>),
    F16(DeviceMemory<f16>),
    F32(DeviceMemory<f32>),
    F64(DeviceMemory<f64>),
    F8E4M3(DeviceMemory<F8E4M3>),
}

// Created by: TEAM-488 | CUDA parity verified by: TEAM-497
impl RocmStorageSlice {
    /// Get the size in bytes
    pub fn size_in_bytes(&self) -> usize {
        match self {
            Self::U8(s) => s.size(),
            Self::U32(s) => s.size(),
            Self::I64(s) => s.size(),
            Self::BF16(s) => s.size(),
            Self::F16(s) => s.size(),
            Self::F32(s) => s.size(),
            Self::F64(s) => s.size(),
            Self::F8E4M3(s) => s.size(),
        }
    }

    /// Get the element count
    pub fn len(&self) -> usize {
        match self {
            Self::U8(s) => s.count(),
            Self::U32(s) => s.count(),
            Self::I64(s) => s.count(),
            Self::BF16(s) => s.count(),
            Self::F16(s) => s.count(),
            Self::F32(s) => s.count(),
            Self::F64(s) => s.count(),
            Self::F8E4M3(s) => s.count(),
        }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// TEAM-500: Removed explicit Clone implementation
// DeviceMemory now uses Arc internally, so clone is cheap (just ref count increment)
// The old implementation did expensive device-to-device copies
// Now we get CUDA parity: cheap clones via Arc::clone()
// Backwards compatibility maintained: Arc::make_mut() creates independent copies when needed
impl Clone for RocmStorageSlice {
    fn clone(&self) -> Self {
        match self {
            Self::U8(s) => Self::U8(s.clone()),
            Self::U32(s) => Self::U32(s.clone()),
            Self::I64(s) => Self::I64(s.clone()),
            Self::BF16(s) => Self::BF16(s.clone()),
            Self::F16(s) => Self::F16(s.clone()),
            Self::F32(s) => Self::F32(s.clone()),
            Self::F64(s) => Self::F64(s.clone()),
            Self::F8E4M3(s) => Self::F8E4M3(s.clone()),
        }
    }
}
