//! RocmStorage struct definition and basic methods

use crate::rocm_backend::{miopen, RocmDevice, RocmError};
use crate::Result;
use super::RocmStorageSlice;

type S = RocmStorageSlice;

/// ROCm storage - wraps storage slice + device
#[derive(Debug)]
pub struct RocmStorage {
    pub(crate) slice: RocmStorageSlice,
    pub(crate) device: RocmDevice,
}

impl RocmStorage {
    pub fn new(slice: RocmStorageSlice, device: RocmDevice) -> Self {
        Self { slice, device }
    }

    pub fn device(&self) -> &RocmDevice {
        &self.device
    }

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
}
