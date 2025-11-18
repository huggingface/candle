#![cfg(feature = "metal")]

use crate::{storage::Storage, DType, Tensor};
use candle_metal_kernels::metal::Buffer;
use std::fmt;
use std::sync::RwLockReadGuard;

/// Borrowed view into a Metal-backed tensor buffer.
pub struct MetalBufferView<'a> {
    // Hold the storage guard to keep the Metal buffer alive for the duration of the view.
    _storage_guard: RwLockReadGuard<'a, Storage>,
    pub buffer: &'a Buffer,
    pub byte_offset: u64,
    pub byte_len: u64,
    pub dtype: DType,
}

impl<'a> fmt::Debug for MetalBufferView<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetalBufferView")
            .field("buffer", &"...")
            .field("byte_offset", &self.byte_offset)
            .field("byte_len", &self.byte_len)
            .field("dtype", &self.dtype)
            .finish()
    }
}

impl Tensor {
    /// Returns a borrowed Metal buffer view if this tensor is on Metal and contiguous.
    pub fn as_metal_buffer_view(&self) -> Option<MetalBufferView<'_>> {
        if !matches!(self.device(), crate::Device::Metal(_)) {
            return None;
        }
        if !self.is_contiguous() {
            return None;
        }

        let dtype = self.dtype();
        let bpe = dtype.size_in_bytes() as u64;
        let byte_len = (self.elem_count() as u64) * bpe;
        let byte_offset = (self.layout().start_offset() as u64) * bpe;

        let storage_guard = self.storage();
        let buffer_ptr: *const Buffer = {
            let storage_ref: &Storage = &*storage_guard;
            match storage_ref {
                Storage::Metal(storage) => storage.buffer() as *const Buffer,
                _ => return None,
            }
        };
        // SAFETY: buffer_ptr was obtained while holding the guard and remains valid as long as
        // `_storage_guard` is kept alive inside `MetalBufferView`.
        let buffer = unsafe { &*buffer_ptr };

        Some(MetalBufferView {
            _storage_guard: storage_guard,
            buffer,
            byte_offset,
            byte_len,
            dtype,
        })
    }
}
