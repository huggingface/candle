#![allow(unused)]
use super::GgmlDType;
use crate::{Error, Result, VulkanDevice, VulkanStorage};

pub struct QVulkanStorage {
    dtype: GgmlDType,
    device: VulkanDevice,
}

impl QVulkanStorage {
    pub fn new(_device: &VulkanDevice, _dtype: GgmlDType, _data: &[u8]) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    pub fn zeros(_device: &VulkanDevice, _elem_count: usize, _dtype: GgmlDType) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &VulkanDevice {
        &self.device
    }

    pub fn size_in_bytes(&self) -> usize {
        0
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    pub fn dequantize_f32(&self, _elem_count: usize) -> Result<VulkanStorage> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
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
}

pub fn load_quantized(
    _device: &VulkanDevice,
    _data: &[u8],
    _dtype: GgmlDType,
) -> Result<super::QStorage> {
    Err(Error::NotCompiledWithVulkanSupport)
}
