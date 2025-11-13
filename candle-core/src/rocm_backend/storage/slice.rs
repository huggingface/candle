//! ROCm storage slice enum
//!
//! Holds device memory for different tensor dtypes.

use float8::F8E4M3;
use half::{bf16, f16};
use rocm_rs::hip::DeviceMemory;

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

// Clone implementation
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
