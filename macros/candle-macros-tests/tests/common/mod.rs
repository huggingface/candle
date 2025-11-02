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
impl CudaDevice {
    pub fn alloc_zeros<T>(
        &self,
        _len: usize,
    ) -> std::result::Result<cudarc::driver::CudaSlice<T>, Error> {
        Err(Error::Msg("Not implemented in test".to_string()))
    }
}
