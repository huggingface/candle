use crate::wgpu_functions;

use super::{
    cache::BufferReferenceId,
    device::WgpuDevice
};

#[derive(Debug)]
pub struct WgpuStorage {
    buffer: BufferReferenceId,
    size: u64,
    wgpu_device: WgpuDevice,
    dtype: crate::DType,
    is_original: bool, //We may have a temporary representation of a buffer. Nothing happens on Drop if this is not the original object.
}


impl WgpuStorage {
    pub fn buffer(&self) -> BufferReferenceId {
        self.buffer
    }

    pub fn device(&self) -> &WgpuDevice {
        &self.wgpu_device
    }

    pub fn wgpu_dtype(&self) -> crate::DType {
        self.dtype
    }

    pub fn new(
        buffer: BufferReferenceId,
        wgpu_device: WgpuDevice,
        dtype: crate::DType,
        size: u64,
    ) -> Self {
        Self {
            buffer,
            wgpu_device,
            dtype,
            size,
            is_original: true,
        }
    }



    /// Returns a temporary clone of this [`WgpuStorage`].
    ///
    /// # Safety
    /// when a WgpuStorage is dropped it internally marks the underlying wgpu buffer to be freed.
    /// this function creates a WgpuStorage pointing to the same buffer, but the ownership of the buffer will still be the original one.
    /// So this temporary cloned WgpuStorage will not keep the buffer alive, nor will droping this cloned WgpuStorage free the underlying buffer.
    /// 
    /// This clone may be usefull to allow static analysers to detect correct behavior with MutexGuard in async functions.
    pub unsafe fn temporary_clone(&self) -> Self {
        Self {
            buffer: self.buffer,
            size: self.size,
            wgpu_device: self.wgpu_device.clone(),
            dtype: self.dtype,
            is_original: false,
        }
    }

    pub fn size_in_bytes(&self) -> usize {
        self.size as usize
    }

    pub async fn read_from_buffer_reference_async<T: bytemuck::Pod>(&self) -> crate::Result<Vec<T>> {
       return wgpu_functions::read_from_buffer_reference_async(self.device(), self.buffer()).await;
    }

}

impl Drop for WgpuStorage {
    fn drop(&mut self) {
        if self.is_original {
            let mut cache = self.device().cache.lock().expect("could not lock cache");
            cache.buffer_reference.queue_for_deletion(&self.buffer);
        }
    }
}
