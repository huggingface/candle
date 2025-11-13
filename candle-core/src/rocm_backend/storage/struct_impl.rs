//! RocmStorage struct definition and basic methods
//! Created by: TEAM-488 (Phase 1 - Device integration)
//! Modified by: TEAM-491-496 (Kernel work and backend implementation)
//! CUDA parity verified by: TEAM-497

use super::RocmStorageSlice;
use crate::rocm_backend::{miopen, RocmDevice, RocmError};
use crate::Result;

type S = RocmStorageSlice;

// Created by: TEAM-488 | CUDA parity verified by: TEAM-497
// CUDA: cuda_backend/mod.rs:1132-1135
/// ROCm storage - wraps storage slice + device
#[derive(Debug)]
pub struct RocmStorage {
    pub(crate) slice: RocmStorageSlice,
    pub(crate) device: RocmDevice,
}

// Created by: TEAM-488 | CUDA parity verified by: TEAM-497
impl RocmStorage {
    pub fn new(slice: RocmStorageSlice, device: RocmDevice) -> Self {
        Self { slice, device }
    }

    // TEAM-497: Device accessor (CUDA: cuda_backend/mod.rs:1314-1316)
    pub fn device(&self) -> &RocmDevice {
        &self.device
    }

    // TEAM-497: DType accessor (CUDA: cuda_backend/mod.rs:1301-1312)
    pub fn dtype(&self) -> crate::DType {
        match &self.slice {
            S::U8(_) => crate::DType::U8,
            S::U32(_) => crate::DType::U32,
            S::I64(_) => crate::DType::I64,
            S::BF16(_) => crate::DType::BF16,
            S::F16(_) => crate::DType::F16,
            S::F32(_) => crate::DType::F32,
            S::F64(_) => crate::DType::F64,
            S::F8E4M3(_) => crate::DType::F8E4M3,
        }
    }

    // TEAM-497: ROCm-specific helper for pooling (CUDA uses separate avg_pool2d/max_pool2d)
    // CUDA avg_pool2d: cuda_backend/mod.rs:1879-1890
    // CUDA max_pool2d: cuda_backend/mod.rs:1892-1903
    /// Helper method for pooling operations
    pub(super) fn pool2d(
        &self,
        layout: &crate::Layout,
        k: (usize, usize),
        stride: (usize, usize),
        mode: rocm_rs::miopen::ffi::miopenPoolingMode_t,
    ) -> Result<Self> {
        miopen::pool2d(self, layout, k, stride, mode)
    }

    // Created by: TEAM-502 | CUDA parity: cuda_backend/mod.rs:1137-1191 (wrap_cuda_slice pattern)
    /// Wrap a DeviceMemory<f32> into RocmStorage
    pub fn wrap_rocm_slice(mem: rocm_rs::hip::DeviceMemory<f32>, device: RocmDevice) -> Self {
        Self { slice: RocmStorageSlice::F32(mem), device }
    }

    // Created by: TEAM-502 | CUDA parity: cuda_backend/mod.rs:1137-1191 (wrap_cuda_slice pattern)
    /// Wrap a DeviceMemory<f16> into RocmStorage
    pub fn wrap_rocm_slice_f16(
        mem: rocm_rs::hip::DeviceMemory<half::f16>,
        device: RocmDevice,
    ) -> Self {
        Self { slice: RocmStorageSlice::F16(mem), device }
    }

    // Created by: TEAM-502 | CUDA parity: cuda_backend/mod.rs (as_cuda_slice pattern)
    /// Get a reference to the underlying DeviceMemory<f32>
    pub fn as_hip_slice<T>(&self) -> Result<&rocm_rs::hip::DeviceMemory<T>>
    where
        T: 'static,
    {
        use std::any::TypeId;

        match &self.slice {
            S::F32(mem) if TypeId::of::<T>() == TypeId::of::<f32>() => {
                // SAFETY: We've checked the type matches
                Ok(unsafe {
                    &*(mem as *const rocm_rs::hip::DeviceMemory<f32>
                        as *const rocm_rs::hip::DeviceMemory<T>)
                })
            }
            S::F16(mem) if TypeId::of::<T>() == TypeId::of::<half::f16>() => {
                // SAFETY: We've checked the type matches
                Ok(unsafe {
                    &*(mem as *const rocm_rs::hip::DeviceMemory<half::f16>
                        as *const rocm_rs::hip::DeviceMemory<T>)
                })
            }
            S::F64(mem) if TypeId::of::<T>() == TypeId::of::<f64>() => {
                // SAFETY: We've checked the type matches
                Ok(unsafe {
                    &*(mem as *const rocm_rs::hip::DeviceMemory<f64>
                        as *const rocm_rs::hip::DeviceMemory<T>)
                })
            }
            S::U8(mem) if TypeId::of::<T>() == TypeId::of::<u8>() => {
                // SAFETY: We've checked the type matches
                Ok(unsafe {
                    &*(mem as *const rocm_rs::hip::DeviceMemory<u8>
                        as *const rocm_rs::hip::DeviceMemory<T>)
                })
            }
            S::U32(mem) if TypeId::of::<T>() == TypeId::of::<u32>() => {
                // SAFETY: We've checked the type matches
                Ok(unsafe {
                    &*(mem as *const rocm_rs::hip::DeviceMemory<u32>
                        as *const rocm_rs::hip::DeviceMemory<T>)
                })
            }
            S::I64(mem) if TypeId::of::<T>() == TypeId::of::<i64>() => {
                // SAFETY: We've checked the type matches
                Ok(unsafe {
                    &*(mem as *const rocm_rs::hip::DeviceMemory<i64>
                        as *const rocm_rs::hip::DeviceMemory<T>)
                })
            }
            S::BF16(mem) if TypeId::of::<T>() == TypeId::of::<half::bf16>() => {
                // SAFETY: We've checked the type matches
                Ok(unsafe {
                    &*(mem as *const rocm_rs::hip::DeviceMemory<half::bf16>
                        as *const rocm_rs::hip::DeviceMemory<T>)
                })
            }
            _ => Err(crate::Error::UnexpectedDType {
                msg: format!(
                    "Type mismatch in as_hip_slice: expected {:?}, got {:?}",
                    std::any::type_name::<T>(),
                    self.dtype()
                ),
            }
            .bt()),
        }
    }
}
