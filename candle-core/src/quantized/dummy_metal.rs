#![allow(unused)]
use super::GgmlDType;
use crate::{
    quantized::{GgmlType, QuantizedBackend, QuantizedDevice},
    Error, MetalDevice, MetalStorage, Result,
};

#[derive(Debug)]
pub struct QMetalStorage {
    dtype: GgmlDType,
    device: MetalDevice,
}

impl QuantizedDevice<QMetalStorage> for MetalDevice {
    type Storage = MetalStorage;

    fn qzeros(&self, elem_count: usize, dtype: GgmlDType) -> Result<QMetalStorage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn load_quantized<T: GgmlType + Send + Sync + 'static>(
        self: &Self,
        data: &[T],
    ) -> Result<QMetalStorage> {
        Err(Error::NotCompiledWithCudaSupport)
    }
}

impl QuantizedBackend for QMetalStorage {
    type Storage = MetalStorage;
    type Device = MetalDevice;

    fn block_size(&self) -> usize {
        0
    }

    fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    fn storage_size_in_bytes(&self) -> usize {
        0
    }

    fn quantize(&mut self, src: &Self::Storage) -> Result<()> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn dequantize(&self, elem_count: usize) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn data(&self) -> Result<std::borrow::Cow<'_, [u8]>> {
        crate::bail!("not implemented");
    }

    fn device(&self) -> impl AsRef<Self::Device> {
        &self.device
    }
}

impl QMetalStorage {
    pub fn fwd(
        &self,
        _self_shape: &crate::Shape,
        _storage: &MetalStorage,
        _layout: &crate::Layout,
    ) -> Result<(MetalStorage, crate::Shape)> {
        Err(Error::NotCompiledWithMetalSupport)
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    _device: &MetalDevice,
    _data: &[T],
) -> Result<super::QStorage> {
    Err(Error::NotCompiledWithMetalSupport)
}
