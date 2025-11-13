// candle-core/src/rocm_backend/device.rs
// Created by: TEAM-488 (Phase 1)
// Wraps rocm-rs Device - thin wrapper, don't reimplement

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
    /// Create a new ROCm device
    pub fn new(id: usize) -> Result<Self> {
        let inner = HipDevice::new(id as i32)?;
        inner.set_current()?;
        Ok(Self { inner })
    }

    /// Get device ID
    pub fn id(&self) -> usize {
        self.inner.id() as usize
    }

    /// Get device name
    pub fn name(&self) -> Result<String> {
        let props = self.inner.properties()?;
        Ok(props.name)
    }

    /// Get compute capability (major, minor)
    pub fn compute_capability(&self) -> Result<(i32, i32)> {
        let props = self.inner.properties()?;
        Ok((props.major, props.minor))
    }

    /// Synchronize device (wait for all operations to complete)
    pub fn synchronize(&self) -> Result<()> {
        self.inner.synchronize()?;
        Ok(())
    }

    /// Get total memory in bytes
    pub fn total_memory(&self) -> Result<usize> {
        let info = rocm_rs::hip::memory_info()?;
        Ok(info.total)
    }

    /// Get free memory in bytes
    pub fn free_memory(&self) -> Result<usize> {
        let info = rocm_rs::hip::memory_info()?;
        Ok(info.free)
    }

    /// Get underlying rocm-rs device (for kernel operations)
    /// 
    /// This allows direct access to rocm-rs APIs when needed.
    pub fn hip_device(&self) -> &HipDevice {
        &self.inner
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

/// Get number of ROCm devices
pub fn device_count() -> Result<usize> {
    let count = rocm_rs::hip::device_count()?;
    Ok(count as usize)
}

/// Check if ROCm is available
pub fn is_available() -> bool {
    rocm_rs::hip::is_hip_available()
}

/// Get ROCm runtime version
pub fn runtime_version() -> Result<i32> {
    let version = rocm_rs::hip::runtime_version()?;
    Ok(version)
}
