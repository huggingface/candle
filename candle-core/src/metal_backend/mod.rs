use crate::{DType, Layout};
use candle_metal_kernels::BufferOffset;
use metal::Buffer;
use std::sync::TryLockError;

pub mod device;
pub mod storage;
pub use device::{DeviceId, MetalDevice};
pub use storage::MetalStorage;

fn buffer_o<'a>(buffer: &'a Buffer, l: &Layout, dtype: DType) -> BufferOffset<'a> {
    BufferOffset {
        buffer,
        offset_in_bytes: l.start_offset() * dtype.size_in_bytes(),
    }
}
/// Simple way to catch lock error without
/// depending on T
#[derive(thiserror::Error, Debug)]
pub enum LockError {
    #[error("{0}")]
    Poisoned(String),
    #[error("Would block")]
    WouldBlock,
}

impl<T> From<TryLockError<T>> for MetalError {
    fn from(value: TryLockError<T>) -> Self {
        match value {
            TryLockError::Poisoned(p) => MetalError::LockError(LockError::Poisoned(p.to_string())),
            TryLockError::WouldBlock => MetalError::LockError(LockError::WouldBlock),
        }
    }
}

/// Metal related errors
#[derive(thiserror::Error, Debug)]
pub enum MetalError {
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    KernelError(#[from] candle_metal_kernels::MetalKernelError),
    #[error("{0:?}")]
    LockError(LockError),
    #[error("{msg}, expected: {expected:?}, got: {got:?}")]
    UnexpectedDType {
        msg: &'static str,
        expected: DType,
        got: DType,
    },
}

impl From<String> for MetalError {
    fn from(e: String) -> Self {
        MetalError::Message(e)
    }
}

fn read_to_vec<T: Clone>(buffer: &Buffer, n: usize) -> Vec<T> {
    let ptr = buffer.contents() as *const T;
    assert!(!ptr.is_null());
    let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
    slice.to_vec()
}
