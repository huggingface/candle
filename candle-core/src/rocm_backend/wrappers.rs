//! `Send + Sync` wrappers over `rocm-rs` handles.
//!
//! Every type here holds a raw handle, which is the only reason `rocm-rs`
//! declines to derive `Send`/`Sync` itself. A HIP/rocBLAS/MIOpen/rocRAND handle
//! is a process-wide driver object with no thread affinity, so *moving* one
//! across threads is always sound; what differs between them is whether two
//! threads may *use* one at the same time, and that is what each `SAFETY` note
//! below has to pin down. Device memory lives in [`super::alloc`], not here: it
//! needs an allocator, not just a wrapper.

use std::ops::{Deref, DerefMut};

use rocm_rs::hip::Stream;
use rocm_rs::rocrand::PseudoRng;

pub struct SendSyncStream(pub Stream);

// SAFETY: a `hipStream_t` is a driver object owned by the process, not by the
// thread that created it. HIP's own API is thread-safe for concurrent
// submission to one stream — which is exactly what this backend does, since
// every device operation is queued on the owning device's single stream from
// whichever thread runs it (see the ordering invariant in `super::alloc`).
unsafe impl Send for SendSyncStream {}
// SAFETY: see the `Send` impl above.
unsafe impl Sync for SendSyncStream {}

impl Deref for SendSyncStream {
    type Target = Stream;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct SendSyncRocblasHandle(pub rocm_rs::rocblas::Handle);

// SAFETY: rocBLAS is documented thread-safe for concurrent calls against one
// handle; the exception is the handle's *own* mutable state, of which this
// backend touches only the stream — set once in `RocmDevice::new` while the
// handle is still local to the constructing thread, and never changed again.
// A future `set_stream`/`set_math_mode` on a shared handle would need a lock
// and would invalidate this note.
unsafe impl Send for SendSyncRocblasHandle {}
// SAFETY: see the `Send` impl above.
unsafe impl Sync for SendSyncRocblasHandle {}

impl SendSyncRocblasHandle {
    pub fn new() -> Result<Self, rocm_rs::rocblas::error::Error> {
        Ok(Self(rocm_rs::rocblas::Handle::new()?))
    }
}

impl Deref for SendSyncRocblasHandle {
    type Target = rocm_rs::rocblas::Handle;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct SendSyncPseudoRng(pub PseudoRng);

// SAFETY: unlike the other handles here, a rocRAND generator genuinely is *not*
// thread-safe — it carries mutable offset state that two concurrent
// `rocrand_generate_*` calls would corrupt. What makes `Sync` sound is external:
// the generator is reachable only through `RocmDevice::rocrand()`, which hands
// back a `MutexGuard`, so every use is serialised by that mutex. Removing the
// mutex, or adding a second path to the generator, breaks this impl — the
// compiler will not catch it.
unsafe impl Send for SendSyncPseudoRng {}
// SAFETY: see the `Send` impl above — the mutex is load-bearing.
unsafe impl Sync for SendSyncPseudoRng {}

impl SendSyncPseudoRng {
    pub fn new(rng_type: u32) -> Result<Self, rocm_rs::rocrand::error::Error> {
        Ok(Self(PseudoRng::new(rng_type)?))
    }
}

impl Deref for SendSyncPseudoRng {
    type Target = PseudoRng;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for SendSyncPseudoRng {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(feature = "miopen")]
pub struct SendSyncMIOpenHandle(pub rocm_rs::miopen::Handle);

// SAFETY: a `miopenHandle_t` is a process-wide driver object, and this one's
// stream is bound once at construction and never rebound, so moving it between
// threads is sound. `Sync` is the weaker claim: MIOpen serialises internally per
// handle, and this backend's convolution calls are the only users — they read
// the handle, allocate their own workspace and submit to the handle's stream,
// mutating no handle state. The `find` results they share go through
// `miopen::algo_cache`'s own mutex.
#[cfg(feature = "miopen")]
unsafe impl Send for SendSyncMIOpenHandle {}
// SAFETY: see the `Send` impl above.
#[cfg(feature = "miopen")]
unsafe impl Sync for SendSyncMIOpenHandle {}

#[cfg(feature = "miopen")]
impl SendSyncMIOpenHandle {
    pub fn new(stream: &Stream) -> Result<Self, rocm_rs::miopen::error::Error> {
        let handle = rocm_rs::miopen::Handle::with_stream(stream)?;
        Ok(Self(handle))
    }
}

#[cfg(feature = "miopen")]
impl Deref for SendSyncMIOpenHandle {
    type Target = rocm_rs::miopen::Handle;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
