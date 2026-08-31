//! cuTile kernel authoring and CUDA stream interop.
//!
//! ```no_run
//! use candle_core::cutile;
//!
//! #[cutile::module]
//! mod kernels {
//!     use candle_core::cutile;
//!     use cutile::core::*;
//!     use cutile::cutile_compiler;
//!
//!     #[cutile::entry]
//!     unsafe fn noop() {}
//! }
//! ```

use super::cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, DeviceRepr, SyncOnDrop};
use super::{CudaDType, CudaDevice, CudaStorage, WrapErr};
use crate::{Error, Layout, Result};
use ::core::ffi::{c_int, c_void};
use std::sync::Arc;

pub use ::cutile::*;

/// A cuTile launch context backed by Candle's current CUDA stream.
pub struct CutileContext {
    candle_stream: Arc<super::cudarc::driver::CudaStream>,
    cutile_stream: Arc<::cutile::cuda_core::Stream>,
}

impl CutileContext {
    /// Borrows Candle's CUDA context and stream without transferring ownership.
    pub fn new(device: &CudaDevice) -> Result<Self> {
        let candle_stream = device.cuda_stream();
        let candle_context = candle_stream.context();
        candle_context.bind_to_thread().w()?;
        let cutile_device = unsafe {
            ::cutile::cuda_core::Device::borrow_with_owner(
                candle_context.cu_ctx() as *mut c_void,
                candle_context.cu_device() as c_int,
                candle_context.ordinal(),
                candle_context.clone(),
            )
        };
        let cutile_stream = unsafe {
            ::cutile::cuda_core::Stream::borrow_with_owner(
                candle_stream.cu_stream() as *mut c_void,
                &cutile_device,
                candle_stream.clone(),
            )
        };
        Ok(Self {
            candle_stream,
            cutile_stream,
        })
    }

    /// The borrowed cuTile stream used by `compile_on` and `async_on`.
    pub fn stream(&self) -> &Arc<::cutile::cuda_core::Stream> {
        &self.cutile_stream
    }

    /// Borrows a typed Candle allocation for a cuTile kernel read.
    pub fn read<'a, T: DeviceRepr>(
        &'a self,
        slice: &'a CudaSlice<T>,
        offset: usize,
    ) -> Result<CutileRead<'a, T>> {
        if offset > slice.len() {
            return Err(Error::msg(format!(
                "cuTile read offset {offset} exceeds allocation length {}",
                slice.len()
            )));
        }
        let (pointer, guard) = slice.device_ptr(&self.candle_stream);
        let byte_offset = offset
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| Error::msg("cuTile read offset overflow"))?;
        let pointer = pointer
            .checked_add(byte_offset as u64)
            .ok_or_else(|| Error::msg("cuTile read pointer overflow"))?;
        Ok(CutileRead {
            pointer: unsafe {
                ::cutile::cuda_async::device_buffer::DevicePointer::from_cu_deviceptr(pointer)
            },
            _guard: guard,
        })
    }

    /// Borrows a typed Candle allocation for a cuTile kernel write.
    pub fn write<'a, T: DeviceRepr>(
        &'a self,
        slice: &'a mut CudaSlice<T>,
        offset: usize,
    ) -> Result<CutileWrite<'a, T>> {
        if offset > slice.len() {
            return Err(Error::msg(format!(
                "cuTile write offset {offset} exceeds allocation length {}",
                slice.len()
            )));
        }
        let (pointer, guard) = slice.device_ptr_mut(&self.candle_stream);
        let byte_offset = offset
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| Error::msg("cuTile write offset overflow"))?;
        let pointer = pointer
            .checked_add(byte_offset as u64)
            .ok_or_else(|| Error::msg("cuTile write pointer overflow"))?;
        Ok(CutileWrite {
            pointer: unsafe {
                ::cutile::cuda_async::device_buffer::DevicePointer::from_cu_deviceptr(pointer)
            },
            _guard: guard,
        })
    }

    /// Borrows a typed Candle storage and applies its layout offset.
    pub fn read_storage<'a, T: CudaDType + DeviceRepr + 'a>(
        &'a self,
        storage: &'a CudaStorage,
        layout: &Layout,
    ) -> Result<CutileRead<'a, T>> {
        self.read(storage.as_cuda_slice::<T>()?, layout.start_offset())
    }

    /// Mutably borrows a typed Candle storage and applies its layout offset.
    pub fn write_storage<'a, T: CudaDType + DeviceRepr + 'a>(
        &'a self,
        storage: &'a mut CudaStorage,
        layout: &Layout,
    ) -> Result<CutileWrite<'a, T>> {
        self.write(storage.as_cuda_slice_mut::<T>()?, layout.start_offset())
    }
}

/// A typed cuTile pointer that records a Candle read after it is dropped.
#[must_use = "keep the pointer guard alive until the cuTile launch is enqueued"]
pub struct CutileRead<'a, T> {
    pointer: ::cutile::cuda_async::device_buffer::DevicePointer<T>,
    _guard: SyncOnDrop<'a>,
}

impl<T> CutileRead<'_, T> {
    /// Returns the pointer accepted by raw-pointer cuTile entry functions.
    pub fn device_pointer(&self) -> ::cutile::cuda_async::device_buffer::DevicePointer<T> {
        unsafe {
            ::cutile::cuda_async::device_buffer::DevicePointer::from_cu_deviceptr(
                self.pointer.cu_deviceptr(),
            )
        }
    }
}

/// A typed cuTile pointer that records a Candle write after it is dropped.
#[must_use = "keep the pointer guard alive until the cuTile launch is enqueued"]
pub struct CutileWrite<'a, T> {
    pointer: ::cutile::cuda_async::device_buffer::DevicePointer<T>,
    _guard: SyncOnDrop<'a>,
}

impl<T> CutileWrite<'_, T> {
    /// Returns the pointer accepted by raw-pointer cuTile entry functions.
    pub fn device_pointer(&self) -> ::cutile::cuda_async::device_buffer::DevicePointer<T> {
        unsafe {
            ::cutile::cuda_async::device_buffer::DevicePointer::from_cu_deviceptr(
                self.pointer.cu_deviceptr(),
            )
        }
    }
}

/// Converts a cuTile error or panic into a Candle error.
pub fn kernel<T, E: std::fmt::Debug>(
    operation: &str,
    f: impl FnOnce() -> std::result::Result<T, E>,
) -> Result<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(error)) => Err(Error::msg(format!("cuTile {operation}: {error:?}"))),
        Err(payload) => {
            let message = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&'static str>().copied())
                .unwrap_or("non-string panic");
            Err(Error::msg(format!(
                "cuTile {operation} panicked: {message}"
            )))
        }
    }
}
