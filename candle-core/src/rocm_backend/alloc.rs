//! Device allocation for the ROCm backend.
//!
//! `hipFree` blocks until *all* pending device work completes. Every dropped
//! temporary — the dims/strides buffer at the end of a launcher, the output of
//! every intermediate tensor — therefore used to drain the pipeline, so a chain
//! of twenty pointwise ops paid twenty full device synchronisations and the
//! backend had essentially no CPU/GPU overlap.
//!
//! [`RocmAllocator`] fixes that by not handing buffers back to the driver on
//! the hot path: freed blocks go on a per-size free list and are reused by the
//! next allocation of that size. Freeing becomes a host-side push and allocating
//! becomes a host-side pop. The cache is bounded — parked bytes above
//! [`default_cache_limit`] are evicted back to the driver, largest block first —
//! so a workload whose shapes drift every step (autoregressive decode above
//! all) cannot hoard the card's VRAM in blocks nothing will request again.
//!
//! # Why not `hipMallocAsync`
//!
//! HIP's own stream-ordered allocator is the textbook answer and was tried
//! first. On ROCm 7.2.4 it is marked Beta and behaves like it: it is not
//! thread-safe (two threads allocating and freeing against one pool hand back
//! overlapping buffers), a process that creates and destroys ~100 `RocmDevice`s
//! faults partway through when each gets its own `hipMemPoolCreate`, and using
//! the device's default pool instead faults sooner still under a mixed
//! pointwise/conv workload. Every one of those was reproduced on gfx1101; the
//! reuse below is ordinary `hipMalloc` memory with no driver bookkeeping to get
//! wrong.
//!
//! # The ordering invariant
//!
//! Reusing a block for a new tensor is safe because **every** device operation
//! this backend issues that touches allocator memory is queued on the owning
//! `RocmDevice`'s single stream: kernel launches, the D2D copies in
//! `copy_strided_src`/`copy2d`, `alloc_zeros`' memset, the host transfers in
//! `clone_htod`/`clone_dtoh`, rocBLAS, MIOpen and rocRAND. Work queued for the
//! new owner therefore runs strictly after the work of the old one. Anything
//! added later that issues on another stream — or on the null stream — must
//! either take the same stream or synchronise before the buffer can be recycled.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use rocm_rs::hip::bindings;
use rocm_rs::hip::error::Error as HipError;

use super::wrappers::SendSyncStream;

fn hip_check(status: bindings::hipError_t) -> Result<(), HipError> {
    if status == bindings::hipError_t_hipSuccess {
        Ok(())
    } else {
        Err(HipError::new(status))
    }
}

/// Bounds on the allocation granularity.
///
/// Rounding up means two tensors of *similar* size share a bucket instead of
/// each forcing a fresh `hipMalloc`. The granularity is *relative* — an eighth
/// of the request's power-of-two floor — because the workloads that defeat a
/// fixed granularity are the shape-drifting ones: autoregressive decode grows
/// its attention buffers by a few KB every step, and with a fixed step a
/// slightly-larger request lands in a fresh bucket while the previous step's
/// block is parked forever. An eighth keeps the overshoot at or below 12.5%
/// and puts every request between consecutive powers of two into one of eight
/// buckets, so a growing buffer reuses its block for many steps before
/// stepping up.
///
/// 512 B is the floor — the driver's own alignment, so small buffers waste
/// nothing that was not already padding. 32 MiB is the ceiling, so a
/// multi-GiB weight tensor overshoots by at most 32 MiB rather than by 12.5%.
const SMALL_GRANULARITY: usize = 512;
const MAX_GRANULARITY: usize = 32 << 20;

fn bucket_size(size: usize) -> usize {
    // `size` is never 0 here, but `max(1)` keeps `ilog2` total anyway.
    let granularity =
        ((1usize << size.max(1).ilog2()) / 8).clamp(SMALL_GRANULARITY, MAX_GRANULARITY);
    size.div_ceil(granularity) * granularity
}

/// A device pointer parked on the free list.
///
/// The raw pointer is not `Send`, but the memory it addresses belongs to the
/// GPU and is reachable from any host thread; the wrapper exists only to say so
/// once rather than at every use.
#[derive(Clone, Copy)]
struct Block(*mut std::ffi::c_void);

// SAFETY: a device address is process-wide and carries no thread affinity. The
// free list that holds these is behind a `Mutex`, so no two threads can observe
// the same block as available.
unsafe impl Send for Block {}

/// The free list plus the running total of the bytes parked on it, kept
/// together under one lock so the total can never drift from the map.
#[derive(Default)]
struct FreeLists {
    map: HashMap<usize, Vec<Block>>,
    cached_bytes: usize,
}

/// Caching device allocator for one [`super::RocmDevice`].
///
/// Holds an `Arc` of the device's stream so that the ordering invariant
/// documented at the top of this module is expressible: the allocator, and
/// every buffer it hands out, keeps that stream alive.
pub struct RocmAllocator {
    stream: Arc<SendSyncStream>,
    free: Mutex<FreeLists>,
    /// Ceiling on [`FreeLists::cached_bytes`]; parking a block above it evicts
    /// parked blocks back to the driver, largest first. `usize::MAX` disables
    /// the cap. Atomic only so tests can tighten it through the `Arc`.
    cache_limit: AtomicUsize,
}

// SAFETY: the state is a stream handle — a process-wide driver object with no
// thread affinity — and a `Mutex`-guarded map of device addresses.
unsafe impl Send for RocmAllocator {}
// SAFETY: see the `Send` impl above.
unsafe impl Sync for RocmAllocator {}

impl RocmAllocator {
    pub(crate) fn new(stream: Arc<SendSyncStream>) -> Self {
        Self {
            stream,
            free: Mutex::new(FreeLists::default()),
            cache_limit: AtomicUsize::new(default_cache_limit()),
        }
    }

    pub(crate) fn raw_stream(&self) -> bindings::hipStream_t {
        self.stream.0.as_raw()
    }

    /// Poisoning is ignored: the map is a pure cache, so the worst a panic
    /// mid-update can leave behind is a block that is never reused. Refusing to
    /// allocate afterwards would be strictly worse.
    fn lock_free(&self) -> MutexGuard<'_, FreeLists> {
        self.free.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A block of at least `size` bytes, from the free list if one is parked.
    fn alloc_bytes(&self, size: usize) -> Result<(*mut std::ffi::c_void, usize), HipError> {
        if size == 0 {
            return Ok((std::ptr::null_mut(), 0));
        }
        let bucket = bucket_size(size);
        {
            let mut free = self.lock_free();
            if let Some(block) = free.map.get_mut(&bucket).and_then(Vec::pop) {
                free.cached_bytes -= bucket;
                return Ok((block.0, bucket));
            }
        }

        match raw_malloc(bucket) {
            Ok(ptr) => Ok((ptr, bucket)),
            // Out of memory only means the *cache* is holding it. Hand
            // everything back and try once more before reporting failure.
            Err(e) => {
                self.release_all();
                raw_malloc(bucket).map(|ptr| (ptr, bucket)).map_err(|_| e)
            }
        }
    }

    /// Park a block for reuse; on the hot path this is a host-side push.
    ///
    /// If parking the block lifts the cache past [`Self::cache_limit`], parked
    /// blocks are returned to the driver, largest bucket first, until the
    /// cache fits again. The cap is what keeps a shape-drifting workload — a
    /// decode loop whose buffers grow every step, parking a slightly-too-small
    /// block each time — from hoarding the whole card: the driver, rocRAND,
    /// rocBLAS workspaces and every other process allocate outside this free
    /// list, so `alloc_bytes`' release-and-retry cannot save *them* from an
    /// OOM this cache caused. Largest-first eviction throws out exactly the
    /// outgrown blocks while the small, hot buckets survive.
    fn recycle(&self, ptr: *mut std::ffi::c_void, bucket: usize) {
        if ptr.is_null() {
            return;
        }
        let evicted = {
            let mut free = self.lock_free();
            free.map.entry(bucket).or_default().push(Block(ptr));
            free.cached_bytes += bucket;
            let mut evicted = Vec::new();
            let cache_limit = self.cache_limit.load(Ordering::Relaxed);
            // Every round removes a block or an empty bucket, so this ends.
            while free.cached_bytes > cache_limit {
                let Some(&largest) = free.map.keys().max() else {
                    break;
                };
                // `alloc_bytes` pops without pruning, so a bucket can be empty.
                match free.map.get_mut(&largest).and_then(Vec::pop) {
                    Some(block) => {
                        evicted.push(block);
                        free.cached_bytes -= largest;
                    }
                    None => {
                        free.map.remove(&largest);
                    }
                }
            }
            evicted
        };
        // `hipFree` synchronises the device, so it runs outside the lock; the
        // sync is also what makes freeing sound while queued work may still
        // reference the block.
        for block in evicted {
            // SAFETY: the block came from `hipMalloc` and is no longer
            // reachable from the map.
            unsafe {
                let _ = bindings::hipFree(block.0);
            }
        }
    }

    /// Bytes currently parked on the free list.
    #[cfg(test)]
    pub(crate) fn cached_bytes(&self) -> usize {
        self.lock_free().cached_bytes
    }

    #[cfg(test)]
    pub(crate) fn set_cache_limit(&self, bytes: usize) {
        self.cache_limit.store(bytes, Ordering::Relaxed);
    }

    /// Return every parked block to the driver.
    ///
    /// `hipFree` is documented to synchronise the device, which is exactly what
    /// is wanted here: the blocks may still be referenced by queued work, and
    /// this runs only when the device is being torn down or an allocation has
    /// already failed.
    fn release_all(&self) {
        let mut free = self.lock_free();
        free.cached_bytes = 0;
        for (_, blocks) in free.map.drain() {
            for block in blocks {
                // SAFETY: every block came from `hipMalloc` and is not
                // reachable from anywhere else once drained from the map.
                unsafe {
                    let _ = bindings::hipFree(block.0);
                }
            }
        }
    }
}

/// The cap on parked bytes: an eighth of the card's VRAM, overridable with
/// `CANDLE_ROCM_CACHE_LIMIT_MB` (`0` disables the cap).
///
/// An eighth is small enough that the cache never starves the driver or a
/// neighbouring library of a 16 GB card, and large enough that a shape-stable
/// workload — whose parked set is only the buffers currently between owners —
/// never reaches it and keeps the old always-cache behaviour.
fn default_cache_limit() -> usize {
    if let Some(mb) = std::env::var("CANDLE_ROCM_CACHE_LIMIT_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        return if mb == 0 { usize::MAX } else { mb << 20 };
    }
    let (mut free, mut total) = (0usize, 0usize);
    // SAFETY: both out-pointers are valid; the values are only read on success.
    let status = unsafe { bindings::hipMemGetInfo(&mut free, &mut total) };
    if status == bindings::hipError_t_hipSuccess && total > 0 {
        total / 8
    } else {
        1 << 30
    }
}

fn raw_malloc(size: usize) -> Result<*mut std::ffi::c_void, HipError> {
    let mut ptr = std::ptr::null_mut();
    // SAFETY: `ptr` is only read once the status has been checked.
    hip_check(unsafe { bindings::hipMalloc(&mut ptr, size) })?;
    Ok(ptr)
}

impl Drop for RocmAllocator {
    fn drop(&mut self) {
        self.release_all();
    }
}

/// A device buffer of `T`, returned to its allocator's free list when dropped.
///
/// The API surface deliberately matches the `rocm_rs::hip::DeviceMemory` this
/// replaces, so no call site had to change; what changed is that `Drop` no
/// longer drains the device.
pub struct SendSyncDeviceMemory<T> {
    ptr: *mut std::ffi::c_void,
    /// Requested size in bytes. The block behind it is [`Self::bucket`] bytes,
    /// which is at least this.
    size: usize,
    bucket: usize,
    alloc: Arc<RocmAllocator>,
    phantom: PhantomData<T>,
}

// SAFETY: the state is a device pointer plus a refcount on the allocator. The
// pointer addresses GPU memory, not host memory, so no host thread can race on
// it; ownership is unique and mutation goes through `&mut self`.
//
// `T` is unbounded on purpose. No value of `T` is ever stored here — it is a
// size and type tag for `PhantomData` and for the `&[T]` the host transfers
// borrow — so `T`'s own thread-safety is irrelevant. A `T: Send` bound would
// read as if it were load-bearing, which is exactly the confusion to avoid.
unsafe impl<T> Send for SendSyncDeviceMemory<T> {}
// SAFETY: see the `Send` impl above.
unsafe impl<T> Sync for SendSyncDeviceMemory<T> {}

impl<T> SendSyncDeviceMemory<T> {
    pub(crate) fn new(alloc: &Arc<RocmAllocator>, count: usize) -> Result<Self, HipError> {
        let size = count * std::mem::size_of::<T>();
        let (ptr, bucket) = alloc.alloc_bytes(size)?;
        Ok(Self {
            ptr,
            size,
            bucket,
            alloc: alloc.clone(),
            phantom: PhantomData,
        })
    }

    /// Base device pointer. Arithmetic on it advances *bytes*.
    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }

    /// Size of the allocation in bytes, as requested rather than as bucketed.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Number of elements the allocation holds.
    pub fn count(&self) -> usize {
        self.size / std::mem::size_of::<T>()
    }

    /// Device pointer to element `offset`.
    ///
    /// [`Self::as_ptr`] hands back a `*mut c_void`, so arithmetic on it
    /// advances *bytes*. Every offset candle deals in — `Layout::start_offset`
    /// above all — counts *elements*, so it has to be scaled by the element
    /// size. Doing this by hand at each call site silently mis-addresses every
    /// tensor whose dtype is wider than a byte.
    ///
    /// # Safety
    /// `offset` must be within the allocation.
    pub unsafe fn ptr_at(&self, offset: usize) -> *mut std::ffi::c_void {
        self.ptr.add(offset * std::mem::size_of::<T>())
    }

    /// Host-to-device copy, clamped to the shorter of the two buffers.
    ///
    /// Drains the stream first, then copies synchronously. The drain is what
    /// orders this against the queued work of whoever owned the block before —
    /// the module-level invariant, made explicit rather than left to the
    /// blocking stream's legacy synchronisation with the null stream. The copy
    /// itself stays synchronous because `data` is a borrowed host slice the
    /// caller may drop the moment this returns, and because the driver's
    /// pageable-memory path is measurably faster than an async copy that is
    /// immediately waited on.
    pub fn copy_from_host(&mut self, data: &[T]) -> Result<(), HipError> {
        if self.ptr.is_null() || data.is_empty() {
            return Ok(());
        }
        let bytes = self.size.min(std::mem::size_of_val(data));
        self.synchronize()?;
        // SAFETY: both pointers are valid for `bytes`, which is clamped to the
        // smaller of the two.
        hip_check(unsafe {
            bindings::hipMemcpy(
                self.ptr,
                data.as_ptr() as *const std::ffi::c_void,
                bytes,
                bindings::hipMemcpyKind_hipMemcpyHostToDevice,
            )
        })
    }

    /// Device-to-host copy, clamped to the shorter of the two buffers.
    ///
    /// Same shape as [`Self::copy_from_host`]: drain, then copy. The drain is
    /// also what makes the data this reads the data the caller expects.
    pub fn copy_to_host(&self, data: &mut [T]) -> Result<(), HipError> {
        if self.ptr.is_null() || data.is_empty() {
            return Ok(());
        }
        let bytes = self.size.min(std::mem::size_of_val(data));
        self.synchronize()?;
        // SAFETY: as above.
        hip_check(unsafe {
            bindings::hipMemcpy(
                data.as_mut_ptr() as *mut std::ffi::c_void,
                self.ptr,
                bytes,
                bindings::hipMemcpyKind_hipMemcpyDeviceToHost,
            )
        })
    }

    /// Stream-ordered device-to-device copy, clamped to the shorter buffer.
    ///
    /// Both buffers belong to the same device — the only case candle produces —
    /// so one stream orders the copy against whatever produced `src` and
    /// whatever consumes `self`, with no host synchronisation.
    pub fn copy_from_device(&mut self, src: &SendSyncDeviceMemory<T>) -> Result<(), HipError> {
        if self.ptr.is_null() || src.ptr.is_null() {
            return Ok(());
        }
        let bytes = self.size.min(src.size);
        // SAFETY: both pointers are valid for `bytes`.
        hip_check(unsafe {
            bindings::hipMemcpyAsync(
                self.ptr,
                src.ptr,
                bytes,
                bindings::hipMemcpyKind_hipMemcpyDeviceToDevice,
                self.alloc.raw_stream(),
            )
        })
    }

    /// Stream-ordered `memset` over the requested extent.
    pub fn memset(&mut self, value: i32) -> Result<(), HipError> {
        if self.ptr.is_null() {
            return Ok(());
        }
        // SAFETY: `self.ptr` is valid for `self.size` bytes.
        hip_check(unsafe {
            bindings::hipMemsetAsync(self.ptr, value, self.size, self.alloc.raw_stream())
        })
    }

    fn synchronize(&self) -> Result<(), HipError> {
        // SAFETY: the allocator keeps this stream alive.
        hip_check(unsafe { bindings::hipStreamSynchronize(self.alloc.raw_stream()) })
    }
}

impl<T> Drop for SendSyncDeviceMemory<T> {
    fn drop(&mut self) {
        self.alloc.recycle(self.ptr, self.bucket);
        self.ptr = std::ptr::null_mut();
    }
}

#[cfg(test)]
mod tests {
    use super::{bucket_size, MAX_GRANULARITY, SMALL_GRANULARITY};

    #[test]
    fn buckets_round_up_to_the_granularity() {
        assert_eq!(bucket_size(1), SMALL_GRANULARITY);
        assert_eq!(bucket_size(SMALL_GRANULARITY), SMALL_GRANULARITY);
        assert_eq!(bucket_size(SMALL_GRANULARITY + 1), 2 * SMALL_GRANULARITY);
        // At 1 MiB the relative granularity is 128 KiB, so one byte past an
        // exact bucket steps up by 128 KiB rather than by 512 B.
        assert_eq!(bucket_size(1 << 20), 1 << 20);
        assert_eq!(bucket_size((1 << 20) + 1), (1 << 20) + (128 << 10));
    }

    /// Two tensors of the same shape must land in the same bucket, or the free
    /// list never hits during an inference loop.
    #[test]
    fn equal_sizes_share_a_bucket() {
        assert_eq!(bucket_size(4096 * 4), bucket_size(4096 * 4));
        assert_eq!(bucket_size(64 << 20), bucket_size(64 << 20));
    }

    /// A bucket must hold the request, and the granularity choice bounds the
    /// overshoot: 12.5% relative in the mid range, 32 MiB absolute on giants.
    #[test]
    fn overshoot_is_bounded() {
        for size in [1usize, 700, 4 << 10, 1 << 20, 70 << 20, 3 << 30] {
            let bucket = bucket_size(size);
            assert!(bucket >= size);
            assert!(bucket - size <= (size / 8).clamp(SMALL_GRANULARITY, MAX_GRANULARITY));
        }
    }

    /// Growing a buffer across a whole octave visits only a handful of
    /// buckets, which is what stops a decode loop — whose attention buffers
    /// grow by a few KB per generated token — from parking a fresh
    /// never-reused block every step.
    #[test]
    fn an_octave_of_growth_visits_few_buckets() {
        let mut buckets = std::collections::BTreeSet::new();
        let mut size = 1usize << 20;
        while size <= 2 << 20 {
            buckets.insert(bucket_size(size));
            size += 4 << 10;
        }
        assert!(buckets.len() <= 9, "got {} buckets", buckets.len());
    }
}
