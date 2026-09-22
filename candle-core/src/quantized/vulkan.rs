//! Vulkan quantized storage: raw GGUF quantized bytes in VRAM.
//!
//! `QVulkanStorage` holds the raw quantized bytes (as read from the GGUF
//! file) in a `Subbuffer<[u8]>`. The decode (m == 1) matmul runs the Q4_K
//! GEMV kernel directly on the raw bytes; the prefill (m > 1) path
//! dequantizes to f32 in VRAM and runs the f32 GEMM. The other dtypes
//! (Q6_K, Q8_0, Q5_0, Q5_K) dequantize to f32 on the fly and run the f32
//! kernels.
use crate::quantized::{GgmlDType, GgmlType};
use crate::{Layout, Result, Shape};
use crate::{VulkanDevice, VulkanStorage};
use vulkano::buffer::Subbuffer;
pub struct QVulkanStorage {
    bytes: Subbuffer<[u8]>,
    device: VulkanDevice,
    dtype: GgmlDType,
}
impl std::fmt::Debug for QVulkanStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QVulkanStorage")
            .field("storage_size_in_bytes", &self.storage_size_in_bytes())
            .field("dtype", &self.dtype)
            .finish()
    }
}
impl QVulkanStorage {
    /// Allocate a zero-filled quantized buffer for `elem_count` elements.
    pub fn zeros(device: &VulkanDevice, elem_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size = elem_count.div_ceil(dtype.block_size()) * dtype.type_size();
        let bytes = device.upload_u8(&vec![0u8; size])?;
        Ok(Self {
            bytes,
            device: device.clone(),
            dtype,
        })
    }

    /// Vulkan kernels take buffer handles; raw device pointers are not exposed.
    pub fn device_ptr(&self) -> Result<*const u8> {
        crate::bail!("vulkan: raw device pointers are not exposed by this backend")
    }

    pub fn device(&self) -> &VulkanDevice {
        &self.device
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.bytes.len() as usize
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        self.device.download_u8(&self.bytes)
    }

    pub fn dequantize(&self, _elem_count: usize) -> Result<VulkanStorage> {
        todo!()
    }

    pub fn embedding(
        &self,
        _rows: usize,
        _hidden: usize,
        _ids_storage: &VulkanStorage,
        _layout: &Layout,
    ) -> Result<VulkanStorage> {
        todo!()
    }

    pub fn fwd(
        &self,
        _shape: &Shape,
        _storage: &VulkanStorage,
        _layout: &Layout,
    ) -> Result<(VulkanStorage, Shape)> {
        todo!()
    }
}

/// Uploads raw quantized bytes to the Vulkan device.
pub fn load_quantized<T: GgmlType + Send + Sync + 'static>(
    _device: &VulkanDevice,
    _data: &[T],
) -> Result<super::QStorage> {
    todo!()
}
