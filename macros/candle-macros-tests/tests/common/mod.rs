//! Common test utilities and mocks for candle-macros integration tests

#![allow(dead_code)]

// Mock candle-core types
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Msg(String),
}

impl From<String> for Error {
    fn from(s: String) -> Self {
        Error::Msg(s)
    }
}

// Mock CudaDevice at crate root (where macro expects it)
// This needs to be re-exported at the crate root of each test file
#[cfg(feature = "cuda")]
pub struct CudaDevice;

#[cfg(feature = "cuda")]
impl candle_macros_types::CudaStorageDevice for CudaDevice {
    fn alloc_zeros<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
        &self,
        _len: usize,
    ) -> std::result::Result<cudarc::driver::CudaSlice<T>, Box<dyn std::error::Error + Send + Sync>>
    {
        Err("Not implemented in test".into())
    }
}
