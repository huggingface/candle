//! rocrand-backed random tensor generation.
//!
//! Split out of `device.rs` so that file stays close to the plain device
//! plumbing; `BackendDevice::rand_uniform`/`rand_normal` just delegate here.

use crate::backend::BackendStorage;
use crate::{DType, Layout, Result, Shape};

use super::alloc::SendSyncDeviceMemory;
use super::{rocm_error, Affine, RocmDevice, RocmStorage, RocmStorageSlice};
use rocm_rs::rocrand::bindings;
use rocm_rs::rocrand::Generator;

/// Calls a raw rocrand generator entry point against a device buffer.
///
/// `rocm-rs`' typed `generate_*` wrappers take its own `DeviceMemory`, which
/// this backend no longer allocates — its `Drop` calls the blocking `hipFree`.
/// The wrappers do nothing but unwrap the pointer and the count, so going
/// straight to the C entry point costs nothing and keeps the buffers
/// stream-ordered.
///
/// The generator runs on the owning `RocmDevice`'s stream — `RocmDevice::new`
/// binds it there with `rocrand_set_stream` — so its writes are ordered against
/// the buffer's allocation and against whatever consumes the buffer next, which
/// is what the caching allocator's block reuse relies on (see `super::alloc`).
///
/// # Safety
/// `call` must be handed the pointer and element count of `data` and must write
/// no more than that.
unsafe fn generate<T, F>(
    rng: &super::wrappers::SendSyncPseudoRng,
    data: &SendSyncDeviceMemory<T>,
    what: &str,
    call: F,
) -> Result<()>
where
    F: FnOnce(bindings::rocrand_generator, *mut T, usize) -> bindings::rocrand_status,
{
    let status = call(rng.as_ptr(), data.as_ptr() as *mut T, data.count());
    if status != bindings::rocrand_status_ROCRAND_STATUS_SUCCESS {
        return Err(rocm_error(format!(
            "rocrand {what} failed with status {status}"
        )));
    }
    Ok(())
}

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
            .map_err(|e| rocm_error(format!("Failed to copy device to device: {}", e)))?;
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
                let data = self.alloc::<f32>(elem_count)?;
                // SAFETY: `generate` passes the buffer's own pointer and count.
                unsafe {
                    generate(&*self.rocrand()?, &data, "generate_uniform", |g, p, n| {
                        bindings::rocrand_generate_uniform(g, p, n)
                    })?
                };
                RocmStorageSlice::F32(data)
            }
            DType::F64 => {
                let data = self.alloc::<f64>(elem_count)?;
                // SAFETY: as above.
                unsafe {
                    generate(
                        &*self.rocrand()?,
                        &data,
                        "generate_uniform_double",
                        |g, p, n| bindings::rocrand_generate_uniform_double(g, p, n),
                    )?
                };
                RocmStorageSlice::F64(data)
            }
            dtype => {
                return Err(rocm_error(format!(
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
                let data = self.alloc::<f32>(elem_count_round)?;
                // SAFETY: `generate` passes the buffer's own pointer and count.
                unsafe {
                    generate(&*self.rocrand()?, &data, "generate_normal", |g, p, n| {
                        bindings::rocrand_generate_normal(g, p, n, mean as f32, std as f32)
                    })?
                };
                RocmStorageSlice::F32(self.shrink_to(data, elem_count)?)
            }
            DType::F64 => {
                let data = self.alloc::<f64>(elem_count_round)?;
                // SAFETY: as above.
                unsafe {
                    generate(
                        &*self.rocrand()?,
                        &data,
                        "generate_normal_double",
                        |g, p, n| bindings::rocrand_generate_normal_double(g, p, n, mean, std),
                    )?
                };
                RocmStorageSlice::F64(self.shrink_to(data, elem_count)?)
            }
            dtype => {
                return Err(rocm_error(format!(
                    "dtype {:?} not supported for rocm rand_normal",
                    dtype
                )))
            }
        };
        self.cast_generated(slice, &layout, dtype)
    }
}
