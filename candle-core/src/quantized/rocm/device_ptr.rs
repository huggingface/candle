//! Handing the packed weights out as a raw device pointer.
//!
//! Downstream crates that own HIP kernels of their own need the address of the
//! quantized payload, the same way `quantized/cuda.rs` hands out a
//! `CUdeviceptr`. The address itself is trivial — [`SendSyncDeviceMemory`] is
//! plain `hipMalloc` memory that never moves — so everything interesting here
//! is about *when* it stops being valid.
//!
//! Two hazards, and one guard for each:
//!
//! * **Lifetime.** Dropping the storage returns its block to the allocator's
//!   free list, where the next allocation of that size picks it up (see
//!   `rocm_backend::alloc`). A pointer that outlives the storage therefore does
//!   not merely dangle, it aliases somebody else's tensor. [`QRocmPtrGuard`]
//!   borrows the storage, so the borrow checker rejects that.
//! * **Ordering.** The backend queues everything on the device's single stream,
//!   which is what makes the reuse above sound. A consumer that submits on a
//!   *different* stream steps outside that invariant: it may read before a
//!   queued write lands, and its reads may still be in flight when the block is
//!   recycled. So the guard drains the owning stream when it is taken and the
//!   consumer stream when it is dropped — the coarse equivalent of the
//!   wait-on-write / record-read event pair cudarc's `SyncOnDrop` performs.
//!
//! A consumer that submits on `RocmDevice::stream()` needs neither drain, and
//! the guard skips both.

use rocm_rs::hip::Stream;

use super::QRocmStorage;
use crate::rocm_backend::rocm_error;
use crate::Result;

/// Keeps a device pointer obtained from [`QRocmStorage::device_ptr_with_guard`]
/// valid: the storage cannot be dropped while it lives, and dropping it drains
/// the consumer stream when that stream is not the buffer's own.
pub struct QRocmPtrGuard<'a> {
    stream: Option<&'a Stream>,
}

impl Drop for QRocmPtrGuard<'_> {
    fn drop(&mut self) {
        if let Some(stream) = self.stream.take() {
            // A failed synchronise leaves HIP's error sticky, so the next call
            // on this device reports it; there is nothing this drop can do with
            // it that would not be worse.
            let _ = stream.synchronize();
        }
    }
}

impl QRocmStorage {
    /// Device address of the quantized payload.
    ///
    /// Valid for as long as `self` is alive and only for work queued on
    /// [`crate::rocm_backend::RocmDevice::stream`]; use
    /// [`Self::device_ptr_with_guard`] for anything else.
    pub fn device_ptr(&self) -> Result<*const u8> {
        Ok(self.data.as_ptr() as *const u8)
    }

    /// Device address of the quantized payload, ordered against `stream`.
    ///
    /// The returned guard borrows `self`, so the pointer cannot outlive the
    /// allocation. Drop it only once the work reading through the pointer has
    /// been submitted — see the module docs.
    pub fn device_ptr_with_guard<'a>(
        &'a self,
        stream: &'a Stream,
    ) -> Result<(*const u8, QRocmPtrGuard<'a>)> {
        let own = self.device.stream();
        let foreign = stream.as_raw() != own.as_raw();
        if foreign {
            own.synchronize()
                .map_err(|e| rocm_error(format!("failed to drain the device stream: {e}")))?;
        }
        let guard = QRocmPtrGuard {
            stream: foreign.then_some(stream),
        };
        Ok((self.data.as_ptr() as *const u8, guard))
    }
}
