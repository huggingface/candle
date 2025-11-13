//! ROCm storage slice enum
//! Created by: TEAM-488 (Phase 1 - Device integration)
//! Modified by: TEAM-491-496 (Kernel work and backend implementation)
//! CUDA parity verified by: TEAM-497
//!
//! Holds device memory for different tensor dtypes.
//!
//! ## Architectural Note:
//! CUDA uses Arc-based sharing (CudaSlice with internal Arc), making clone() cheap.
//! ROCm currently uses explicit device-to-device copies, making clone() expensive.
//! TODO: Refactor rocm-rs DeviceMemory to use Arc internally for parity with CUDA.

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

// Created by: TEAM-488 | Modified by: TEAM-491-496
// PARITY: CPU (cpu_backend/mod.rs:21) and Metal (metal_backend/mod.rs:73) both derive Clone
// ARCHITECTURAL DIFFERENCE: CUDA uses Arc-based sharing (cheap clone via ref counting)
//                           ROCm uses explicit GPU memory copy (expensive clone operation)
// TODO: Refactor rocm-rs DeviceMemory to use Arc internally to match CUDA architecture
impl Clone for RocmStorageSlice {
    fn clone(&self) -> Self {
        match self {
            Self::U8(s) => {
                let mut new = DeviceMemory::new(s.count()).expect("allocation failed");
                new.copy_from_device(s).expect("copy failed");
                Self::U8(new)
            }
            Self::U32(s) => {
                let mut new = DeviceMemory::new(s.count()).expect("allocation failed");
                new.copy_from_device(s).expect("copy failed");
                Self::U32(new)
            }
            Self::I64(s) => {
                let mut new = DeviceMemory::new(s.count()).expect("allocation failed");
                new.copy_from_device(s).expect("copy failed");
                Self::I64(new)
            }
            Self::BF16(s) => {
                let mut new = DeviceMemory::new(s.count()).expect("allocation failed");
                new.copy_from_device(s).expect("copy failed");
                Self::BF16(new)
            }
            Self::F16(s) => {
                let mut new = DeviceMemory::new(s.count()).expect("allocation failed");
                new.copy_from_device(s).expect("copy failed");
                Self::F16(new)
            }
            Self::F32(s) => {
                let mut new = DeviceMemory::new(s.count()).expect("allocation failed");
                new.copy_from_device(s).expect("copy failed");
                Self::F32(new)
            }
            Self::F64(s) => {
                let mut new = DeviceMemory::new(s.count()).expect("allocation failed");
                new.copy_from_device(s).expect("copy failed");
                Self::F64(new)
            }
            Self::F8E4M3(s) => {
                let mut new = DeviceMemory::new(s.count()).expect("allocation failed");
                new.copy_from_device(s).expect("copy failed");
                Self::F8E4M3(new)
            }
        }
    }
}
