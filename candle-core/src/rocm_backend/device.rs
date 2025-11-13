//! ROCm device wrapper
//! Created by: TEAM-488 (Phase 1 - Device integration)
//! CUDA parity verified by: TEAM-497, TEAM-498
//! 
//! Note: Like CUDA's CudaDevice, this is a thin wrapper around the underlying library.
//! CUDA exposes `cudarc` directly, we expose `rocm_rs` directly (see mod.rs).
//! Users can access rocm-rs APIs directly when needed via `hip_device()` or `rocm_rs::*`.

use super::error::{Result, RocmError};
use rocm_rs::hip::{Device as HipDevice, DeviceProperties as HipProps};

/// Candle wrapper for ROCm device
/// 
/// This is a thin wrapper around rocm_rs::hip::Device.
/// We don't reimplement - we just wrap the existing API.
#[derive(Debug, Clone)]
pub struct RocmDevice {
    inner: HipDevice,
}

impl RocmDevice {
    // Created by: TEAM-488 | CUDA parity: cuda_backend/device.rs:262-279 (BackendDevice::new)
    /// Create a new ROCm device
    pub fn new(id: usize) -> Result<Self> {
        let inner = HipDevice::new(id as i32)?;
        inner.set_current()?;
        Ok(Self { inner })
    }

    // Created by: TEAM-488 | CUDA parity: cuda_backend/device.rs:188-190
    /// Get device ID
    pub fn id(&self) -> usize {
        self.inner.id() as usize
    }

    // Created by: TEAM-488 | ROCm-specific (CUDA uses cudarc properties directly)
    /// Get device name
    pub fn name(&self) -> Result<String> {
        let props = self.inner.properties()?;
        Ok(props.name)
    }

    // Created by: TEAM-488 | ROCm-specific (CUDA uses cudarc properties directly)
    /// Get compute capability (major, minor)
    pub fn compute_capability(&self) -> Result<(i32, i32)> {
        let props = self.inner.properties()?;
        Ok((props.major, props.minor))
    }

    // Created by: TEAM-488 | ROCm-specific (CUDA uses cudarc stream sync directly)
    /// Synchronize device (wait for all operations to complete)
    pub fn synchronize(&self) -> Result<()> {
        self.inner.synchronize()?;
        Ok(())
    }

    // Created by: TEAM-488 | ROCm-specific (CUDA uses cudarc memory info directly)
    /// Get total memory in bytes
    pub fn total_memory(&self) -> Result<usize> {
        let info = rocm_rs::hip::memory_info()?;
        Ok(info.total)
    }

    // Created by: TEAM-488 | ROCm-specific (CUDA uses cudarc memory info directly)
    /// Get free memory in bytes
    pub fn free_memory(&self) -> Result<usize> {
        let info = rocm_rs::hip::memory_info()?;
        Ok(info.free)
    }

    // Created by: TEAM-488 | ROCm-specific (no CUDA equivalent)
    /// Get underlying rocm-rs device (for kernel operations)
    /// 
    /// This allows direct access to rocm-rs APIs when needed.
    pub fn hip_device(&self) -> &HipDevice {
        &self.inner
    }

    // Created by: TEAM-502 | CUDA parity: cuda_backend/device.rs:56-58
    /// Allocate device memory
    pub unsafe fn alloc<T>(&self, len: usize) -> Result<rocm_rs::hip::DeviceMemory<T>> {
        self.inner.alloc::<T>(len).map_err(RocmError::from)
    }

    // Created by: TEAM-502 | CUDA parity: cuda_backend/device.rs:59-65
    /// Allocate zero-initialized device memory
    pub fn alloc_zeros<T>(&self, len: usize) -> Result<rocm_rs::hip::DeviceMemory<T>>
    where
        T: Default + Clone,
    {
        self.inner.alloc_zeros::<T>(len).map_err(RocmError::from)
    }

    // Created by: TEAM-502 | CUDA parity: cuda_backend/device.rs (memcpy operations)
    /// Copy data from host (stack) to device
    pub fn memcpy_stod<T: Clone>(&self, src: &[T]) -> Result<rocm_rs::hip::DeviceMemory<T>> {
        let mut dst = unsafe { self.alloc::<T>(src.len())? };
        self.inner.memcpy_htod(src, &mut dst).map_err(RocmError::from)?;
        Ok(dst)
    }

    // Created by: TEAM-502 | CUDA parity: cuda_backend/device.rs (memcpy operations)
    /// Copy data from host to device
    pub fn memcpy_htod<T>(&self, src: &[T], dst: &mut rocm_rs::hip::DeviceMemory<T>) -> Result<()> {
        self.inner.memcpy_htod(src, dst).map_err(RocmError::from)
    }

    // Created by: TEAM-502 | CUDA parity: cuda_backend/device.rs (memcpy operations)
    /// Copy data from device to host (vector)
    pub fn memcpy_dtov<T: Clone>(&self, src: &rocm_rs::hip::DeviceMemory<T>) -> Result<Vec<T>> {
        self.inner.memcpy_dtoh(src).map_err(RocmError::from)
    }

    // Created by: TEAM-502 | CUDA parity: cuda_backend/device.rs (kernel loading)
    /// Get or load a kernel function from HSACO binary
    pub fn get_or_load_func(&self, name: &str, hsaco: &[u8]) -> Result<rocm_rs::hip::Function> {
        let module = self.inner.load_module(hsaco).map_err(RocmError::from)?;
        module.get_function(name).map_err(RocmError::from)
    }
}

impl PartialEq for RocmDevice {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl Eq for RocmDevice {}

impl std::hash::Hash for RocmDevice {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id().hash(state);
    }
}

// Created by: TEAM-488 | ROCm-specific (CUDA doesn't export device_count publicly)
/// Get number of ROCm devices
pub fn device_count() -> Result<usize> {
    let count = rocm_rs::hip::device_count()?;
    Ok(count as usize)
}

// Created by: TEAM-488 | ROCm-specific (CUDA doesn't export is_available publicly)
/// Check if ROCm is available
pub fn is_available() -> bool {
    rocm_rs::hip::is_hip_available()
}

// Created by: TEAM-488 | ROCm-specific (CUDA doesn't export runtime_version publicly)
/// Get ROCm runtime version
pub fn runtime_version() -> Result<i32> {
    let version = rocm_rs::hip::runtime_version()?;
    Ok(version)
}
