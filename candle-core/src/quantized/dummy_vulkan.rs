#![allow(unused)]
use super::GgmlDType;
use crate::{quantized::GgmlType, Error, Result, VulkanDevice, VulkanStorage};

pub struct QVulkanStorage {
    dtype: GgmlDType,
    device: VulkanDevice,
}

impl QVulkanStorage {
    pub fn zeros(_device: &VulkanDevice, _elem_count: usize, _dtype: GgmlDType) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &VulkanDevice {
        &self.device
    }

    pub fn dequantize(&self, _elem_count: usize) -> Result<VulkanStorage> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    pub fn quantize(&mut self, _src: &VulkanStorage) -> Result<()> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    pub fn quantize_imatrix(
        &mut self,
        _src: &VulkanStorage,
        _imatrix_weights: &[f32],
        _n_per_row: usize,
    ) -> Result<()> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        _src: &crate::CpuStorage,
        _imatrix_weights: &[f32],
        _n_per_row: usize,
    ) -> Result<()> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    pub fn quantize_onto(&mut self, _src: &crate::CpuStorage) -> Result<()> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        0
    }

    pub fn embedding(
        &self,
        _rows: usize,
        _hidden: usize,
        _ids: &VulkanStorage,
        _ids_l: &crate::Layout,
    ) -> Result<VulkanStorage> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    pub fn fwd(
        &self,
        _shape: &crate::Shape,
        _storage: &VulkanStorage,
        _layout: &crate::Layout,
    ) -> Result<(VulkanStorage, crate::Shape)> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    pub fn indexed_moe_forward(
        &self,
        _: &crate::Shape,
        _: &VulkanStorage,
        _: &crate::Layout,
        _: &VulkanStorage,
        _: &crate::Layout,
    ) -> Result<(VulkanStorage, crate::Shape)> {
        Err(Error::NotCompiledWithVulkanSupport)
    }
}

pub fn load_quantized<T: GgmlType + Send + Sync + 'static>(
    _device: &VulkanDevice,
    _data: &[T],
) -> Result<super::QStorage> {
    Err(Error::NotCompiledWithVulkanSupport)
}
