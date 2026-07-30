//! rocrand-backed random tensor generation.
//!
//! Split out of `device.rs` so that file stays close to the plain device
//! plumbing; `BackendDevice::rand_uniform`/`rand_normal` just delegate here.

use crate::backend::BackendStorage;
use crate::{DType, Layout, Result, Shape};

use super::wrappers::SendSyncDeviceMemory;
use super::{Affine, RocmDevice, RocmStorage, RocmStorageSlice};

/// rocrand only implements the float and double generators, so f16/bf16 are
/// produced in f32 and cast down afterwards.
///
/// The CUDA backend rejects f16/bf16 outright (curand has the same limitation).
/// Accepting them here is a deliberate improvement over strict parity, not a
/// parity fix — `Tensor::randn`/`rand` in half precision are common enough that
/// erroring out is worse than the one extra cast.
fn generation_dtype(dtype: DType) -> DType {
    match dtype {
        DType::F16 | DType::BF16 => DType::F32,
        dtype => dtype,
    }
}

impl RocmDevice {
    /// Shrink `data` to `elem_count` elements, allocating a new buffer when it
    /// is longer.
    ///
    /// The normal generators below have to round the element count up to an
    /// even number. Handing that oversized buffer back would corrupt the
    /// tensor's storage length: `clone_dtoh` sizes its host `Vec` from the
    /// *allocation*, so `to_cpu_storage` would return more elements than the
    /// shape has.
    fn shrink_to<T: Copy>(
        &self,
        data: SendSyncDeviceMemory<T>,
        elem_count: usize,
    ) -> Result<SendSyncDeviceMemory<T>> {
        if data.count() == elem_count {
            return Ok(data);
        }
        let mut dst = self.alloc::<T>(elem_count)?;
        // `copy_from_device` clamps to the shorter of the two allocations, so
        // this keeps the first `elem_count` elements and drops the rest.
        dst.copy_from_device(&data)
            .map_err(|e| crate::Error::Msg(format!("Failed to copy device to device: {}", e)))?;
        Ok(dst)
    }

    /// Cast a freshly generated f32 buffer down to `dtype` when the caller asked
    /// for a dtype rocrand cannot produce directly.
    fn cast_generated(
        &self,
        slice: RocmStorageSlice,
        layout: &Layout,
        dtype: DType,
    ) -> Result<RocmStorage> {
        let storage = RocmStorage {
            slice,
            device: self.clone(),
        };
        if storage.dtype() == dtype {
            Ok(storage)
        } else {
            storage.to_dtype(layout, dtype)
        }
    }

    pub(crate) fn rand_uniform_impl(
        &self,
        shape: &Shape,
        dtype: DType,
        lo: f64,
        hi: f64,
    ) -> Result<RocmStorage> {
        let elem_count = shape.elem_count();
        let layout = Layout::contiguous(shape);
        let slice = match generation_dtype(dtype) {
            DType::F32 => {
                let mut data = self.alloc::<f32>(elem_count)?;
                self.rocrand()?.generate_uniform(&mut data).map_err(|e| {
                    crate::Error::Msg(format!("rocrand generate_uniform failed: {}", e))
                })?;
                RocmStorageSlice::F32(data)
            }
            DType::F64 => {
                let mut data = self.alloc::<f64>(elem_count)?;
                self.rocrand()?
                    .generate_uniform_double(&mut data)
                    .map_err(|e| {
                        crate::Error::Msg(format!("rocrand generate_uniform_double failed: {}", e))
                    })?;
                RocmStorageSlice::F64(data)
            }
            dtype => {
                return Err(crate::Error::Msg(format!(
                    "dtype {:?} not supported for rocm rand_uniform",
                    dtype
                )))
            }
        };
        // The range has to be applied before the down-cast so the scaling keeps
        // f32 precision.
        let slice = if lo == 0. && hi == 1.0 {
            slice
        } else {
            Affine(hi - lo, lo).map(&slice, self, &layout)?
        };
        self.cast_generated(slice, &layout, dtype)
    }

    pub(crate) fn rand_normal_impl(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<RocmStorage> {
        let elem_count = shape.elem_count();
        let layout = Layout::contiguous(shape);
        // rocrand can only generate an even number of normal values.
        let elem_count_round = elem_count.next_multiple_of(2);
        let slice = match generation_dtype(dtype) {
            DType::F32 => {
                let mut data = self.alloc::<f32>(elem_count_round)?;
                self.rocrand()?
                    .generate_normal(&mut data, mean as f32, std as f32)
                    .map_err(|e| {
                        crate::Error::Msg(format!("rocrand generate_normal failed: {}", e))
                    })?;
                RocmStorageSlice::F32(self.shrink_to(data, elem_count)?)
            }
            DType::F64 => {
                let mut data = self.alloc::<f64>(elem_count_round)?;
                self.rocrand()?
                    .generate_normal_double(&mut data, mean, std)
                    .map_err(|e| {
                        crate::Error::Msg(format!("rocrand generate_normal_double failed: {}", e))
                    })?;
                RocmStorageSlice::F64(self.shrink_to(data, elem_count)?)
            }
            dtype => {
                return Err(crate::Error::Msg(format!(
                    "dtype {:?} not supported for rocm rand_normal",
                    dtype
                )))
            }
        };
        self.cast_generated(slice, &layout, dtype)
    }
}
