//! The `[dims, strides…]` buffers every strided kernel takes.
//!
//! These are tiny — a few dozen bytes — but they used to cost a `hipMalloc` and
//! a blocking host-to-device copy on *every* launch. They are now cached per
//! device; see [`params_from_vec`].

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::{rocm_error, RocmDevice, SendSyncDeviceMemory};
use crate::{Layout, Result};

/// Cached device copy of a `[dims, strides…]` parameter vector.
///
/// Every launcher used to `hipMalloc` + blocking `hipMemcpy` a fresh buffer to
/// move at most a few dozen bytes. The same tuples recur on every token of an
/// inference loop and conv shapes are static, so the hit rate is effectively
/// 100%. `cuda_backend`'s `CUDA_PARAM_CACHE` is the same idea; the differences
/// are that this caches unconditionally (CUDA gates it on graph capture, which
/// is what it was built for) and that entries above [`MAX_CACHED_PARAM_BYTES`]
/// fall back to a private allocation instead of being retained forever.
pub enum ParamBuffer {
    Owned(SendSyncDeviceMemory<usize>),
    Cached(Arc<SendSyncDeviceMemory<usize>>),
    Null,
}

impl ParamBuffer {
    /// True when the layout was contiguous and the kernel gets a null `info`.
    pub fn is_null(&self) -> bool {
        matches!(self, ParamBuffer::Null)
    }

    pub fn as_ptr(&self) -> *const usize {
        match self {
            ParamBuffer::Owned(m) => m.as_ptr() as *const usize,
            ParamBuffer::Cached(m) => m.as_ptr() as *const usize,
            ParamBuffer::Null => std::ptr::null(),
        }
    }
}

/// Matches `cuda_backend`'s 4096-byte cap, i.e. 512 `usize`s — far above any
/// real `[dims, strides]` vector, so the bound only ever catches a pathological
/// rank.
const MAX_CACHED_PARAM_BYTES: usize = 4096;

pub(crate) type ParamCache = Mutex<HashMap<Vec<usize>, Arc<SendSyncDeviceMemory<usize>>>>;

/// Upload `params`, reusing a previous upload of the same vector when possible.
///
/// The cache hangs off the `RocmDevice`, not off a process-global map keyed by
/// device id as `cuda_backend` does. Stream-ordered pool memory must not outlive
/// the pool and stream it was allocated from, and a global cache does exactly
/// that: it keeps buffers from a device that has already been torn down, which
/// on ROCm faults the *next* device created in the process. Tying the cache to
/// the device makes that unrepresentable, and drops the device id from the key.
pub fn params_from_vec(dev: &RocmDevice, params: Vec<usize>) -> Result<ParamBuffer> {
    if std::mem::size_of_val(params.as_slice()) > MAX_CACHED_PARAM_BYTES {
        return Ok(ParamBuffer::Owned(dev.clone_htod(&params)?));
    }

    let cache = dev.param_cache();
    {
        let cache = cache
            .lock()
            .map_err(|_| rocm_error("ROCm parameter cache is poisoned".to_string()))?;
        if let Some(buffer) = cache.get(&params) {
            return Ok(ParamBuffer::Cached(buffer.clone()));
        }
    }

    let buffer = Arc::new(dev.clone_htod(&params)?);
    let mut cache = cache
        .lock()
        .map_err(|_| rocm_error("ROCm parameter cache is poisoned".to_string()))?;
    // Another thread may have inserted the same key while this one uploaded;
    // keep whichever landed first so the two threads share one buffer.
    Ok(ParamBuffer::Cached(
        cache.entry(params).or_insert(buffer).clone(),
    ))
}

/// The `[dims, strides]` prefix a one-operand strided kernel takes.
///
/// A contiguous layout gets no buffer at all: the kernels test `info` for null
/// and index linearly when it is.
pub(super) fn dims_and_strides(dev: &RocmDevice, layout: &Layout) -> Result<ParamBuffer> {
    if layout.is_contiguous() {
        return Ok(ParamBuffer::Null);
    }
    let dims = layout.shape().dims();
    let mut data = Vec::with_capacity(dims.len() * 2);
    data.extend_from_slice(dims);
    data.extend_from_slice(layout.stride());
    params_from_vec(dev, data)
}

pub(super) fn dims_and_strides_pair(
    dev: &RocmDevice,
    l1: &Layout,
    l2: &Layout,
) -> Result<ParamBuffer> {
    if l1.is_contiguous() && l2.is_contiguous() {
        return Ok(ParamBuffer::Null);
    }
    let dims = l1.shape().dims();
    let mut data = Vec::with_capacity(dims.len() * 3);
    data.extend_from_slice(dims);
    data.extend_from_slice(l1.stride());
    data.extend_from_slice(l2.stride());
    params_from_vec(dev, data)
}

/// Concatenates `parts` and uploads them as a kernel `info` buffer.
///
/// The conv, pool and upsample kernels dereference `info` unconditionally, so
/// unlike the elementwise ops there is no null shortcut for a contiguous
/// layout — which is exactly why routing this through the parameter cache
/// matters more here than anywhere else. Conv and pool shapes are fixed for the
/// life of a model, so after the first forward pass every call is a hit.
pub(crate) fn info_buffer(dev: &RocmDevice, parts: &[&[usize]]) -> Result<ParamBuffer> {
    let mut data = Vec::with_capacity(parts.iter().map(|p| p.len()).sum());
    for part in parts {
        data.extend_from_slice(part);
    }
    params_from_vec(dev, data)
}

#[cfg(test)]
mod tests {
    use super::{params_from_vec, ParamBuffer, MAX_CACHED_PARAM_BYTES};
    use crate::rocm_backend::RocmDevice;
    use crate::Result;

    /// Every test resolves the device itself and returns early when the machine
    /// has no ROCm GPU, matching the rest of the backend's unit tests.
    macro_rules! device {
        () => {
            match RocmDevice::new(0) {
                Ok(dev) => dev,
                Err(_) => return Ok(()),
            }
        };
    }

    /// The whole point: the same `[dims, strides]` must not be re-uploaded.
    #[test]
    fn the_same_params_reuse_one_buffer() -> Result<()> {
        let dev = device!();
        let a = params_from_vec(&dev, vec![2, 3, 3, 1])?;
        let b = params_from_vec(&dev, vec![2, 3, 3, 1])?;
        assert_eq!(a.as_ptr(), b.as_ptr());
        assert!(!a.is_null());
        Ok(())
    }

    #[test]
    fn different_params_get_different_buffers() -> Result<()> {
        let dev = device!();
        let a = params_from_vec(&dev, vec![2, 3, 3, 1])?;
        let b = params_from_vec(&dev, vec![2, 4, 4, 1])?;
        assert_ne!(a.as_ptr(), b.as_ptr());
        Ok(())
    }

    /// A cache that never evicts must not accept unbounded entries. Above the
    /// cap the caller gets a private allocation that is freed with it.
    #[test]
    fn an_oversized_vector_is_not_cached() -> Result<()> {
        let dev = device!();
        let big: Vec<usize> = (0..MAX_CACHED_PARAM_BYTES).collect();
        let a = params_from_vec(&dev, big.clone())?;
        let b = params_from_vec(&dev, big)?;
        assert!(matches!(a, ParamBuffer::Owned(_)));
        assert_ne!(a.as_ptr(), b.as_ptr());
        Ok(())
    }

    /// Two devices must not share a buffer: the allocation belongs to one
    /// device's allocator and stream, and the cache is keyed per device rather
    /// than globally precisely so that cannot happen.
    #[test]
    fn each_device_caches_separately() -> Result<()> {
        let dev = device!();
        let other = device!();
        let a = params_from_vec(&dev, vec![5, 7, 7, 1])?;
        let b = params_from_vec(&other, vec![5, 7, 7, 1])?;
        assert_ne!(a.as_ptr(), b.as_ptr());
        Ok(())
    }
}
