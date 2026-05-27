use std::ops::{Deref, DerefMut};

use rocm_rs::hip::{DeviceMemory, Stream};
use rocm_rs::miopen::Handle;
use rocm_rs::rocrand::PseudoRng;

pub struct SendSyncDeviceMemory<T>(pub DeviceMemory<T>);

unsafe impl<T: Send> Send for SendSyncDeviceMemory<T> {}
unsafe impl<T: Sync> Sync for SendSyncDeviceMemory<T> {}

impl<T> SendSyncDeviceMemory<T> {
    pub fn new(len: usize) -> Result<Self, rocm_rs::hip::error::Error> {
        Ok(Self(DeviceMemory::new(len)?))
    }
}

impl<T> Deref for SendSyncDeviceMemory<T> {
    type Target = DeviceMemory<T>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for SendSyncDeviceMemory<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

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

pub struct SendSyncMIOpenHandle(pub Handle);

unsafe impl Send for SendSyncMIOpenHandle {}
unsafe impl Sync for SendSyncMIOpenHandle {}

impl SendSyncMIOpenHandle {
    pub fn new(stream: &Stream) -> Result<Self, rocm_rs::miopen::error::Error> {
        let handle = Handle::with_stream(stream)?;
        Ok(Self(handle))
    }
}

impl Deref for SendSyncMIOpenHandle {
    type Target = Handle;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
