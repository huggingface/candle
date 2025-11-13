// candle-core/src/rocm_backend/storage.rs
// Created by: TEAM-488 (Phase 1)
// Wraps rocm-rs DeviceMemory - thin wrapper, don't reimplement

use super::device::RocmDevice;
use super::error::{Result, RocmError};
use crate::DType;
use rocm_rs::hip::DeviceMemory as HipMemory;

/// ROCm storage for tensors
/// 
/// This is a thin wrapper around rocm_rs::hip::DeviceMemory.
/// We don't reimplement - we just wrap the existing API.
pub struct RocmStorage {
    data: HipMemory<u8>,
    dtype: DType,
    device: RocmDevice,
}

impl RocmStorage {
    /// Create new ROCm storage
    pub fn new(size: usize, dtype: DType, device: &RocmDevice) -> Result<Self> {
        let data = HipMemory::new(size)?;
        Ok(Self {
            data,
            dtype,
            device: device.clone(),
        })
    }

    /// Get size in bytes
    pub fn size(&self) -> usize {
        self.data.size()
    }

    /// Get data type
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get device
    pub fn device(&self) -> &RocmDevice {
        &self.device
    }

    /// Copy from host to device
    pub fn copy_from_host(&mut self, src: &[u8]) -> Result<()> {
        if src.len() != self.size() {
            return Err(RocmError::SizeMismatch {
                expected: self.size(),
                actual: src.len(),
            });
        }
        
        self.data.copy_from_host(src)?;
        Ok(())
    }

    /// Copy from device to host
    pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<()> {
        if dst.len() != self.size() {
            return Err(RocmError::SizeMismatch {
                expected: self.size(),
                actual: dst.len(),
            });
        }
        
        self.data.copy_to_host(dst)?;
        Ok(())
    }

    /// Copy from another device storage
    pub fn copy_from_device(&mut self, src: &RocmStorage) -> Result<()> {
        if src.size() != self.size() {
            return Err(RocmError::SizeMismatch {
                expected: self.size(),
                actual: src.size(),
            });
        }
        
        self.data.copy_from_device(&src.data)?;
        Ok(())
    }

    /// Get underlying HIP memory (for kernel operations)
    /// 
    /// This allows direct access to rocm-rs APIs when needed.
    pub fn hip_memory(&self) -> &HipMemory<u8> {
        &self.data
    }

    /// Get mutable underlying HIP memory
    pub fn hip_memory_mut(&mut self) -> &mut HipMemory<u8> {
        &mut self.data
    }

    /// Get raw device pointer
    pub fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    /// Get mutable raw device pointer
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.data.as_mut_ptr()
    }
}

impl Clone for RocmStorage {
    fn clone(&self) -> Self {
        let mut new_data = HipMemory::new(self.size()).expect("Failed to allocate memory");
        new_data
            .copy_from_device(&self.data)
            .expect("Failed to copy data");

        Self {
            data: new_data,
            dtype: self.dtype,
            device: self.device.clone(),
        }
    }
}
