//! `Send + Sync` wrappers over `rocm-rs` handles.
//!
//! Each of these asserts a property of the handle it holds — process-wide,
//! no thread affinity — that `rocm-rs` declines to assert itself because the
//! type contains a raw pointer. Device memory lives in [`super::alloc`], not
//! here: it needs an allocator, not just a wrapper.

use std::ops::{Deref, DerefMut};

use rocm_rs::hip::Stream;
use rocm_rs::rocrand::PseudoRng;

pub struct SendSyncStream(pub Stream);

unsafe impl Send for SendSyncStream {}
unsafe impl Sync for SendSyncStream {}

impl Deref for SendSyncStream {
    type Target = Stream;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct SendSyncRocblasHandle(pub rocm_rs::rocblas::Handle);

unsafe impl Send for SendSyncRocblasHandle {}
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

unsafe impl Send for SendSyncPseudoRng {}
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

#[cfg(feature = "miopen")]
unsafe impl Send for SendSyncMIOpenHandle {}
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
