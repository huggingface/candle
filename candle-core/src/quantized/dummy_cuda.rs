#![allow(unused)]
use super::GgmlDType;
use crate::{
    quantized::{GgmlType, QuantizedBackend, QuantizedDevice},
    CudaDevice, CudaStorage, Error, Result,
};

#[derive(Debug, Clone)]
pub struct QCudaStorage {
    dtype: GgmlDType,
    device: CudaDevice,
}

impl QuantizedDevice for CudaDevice {
    type Storage = QCudaStorage;

    fn qzeros(&self, elem_count: usize, dtype: GgmlDType) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn load_quantized<T: GgmlType + Send + Sync + 'static>(
        self: &Self,
        data: &[T],
    ) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithCudaSupport)
    }
}

impl QuantizedBackend for QCudaStorage {
    type Storage = CudaStorage;
    type Device = CudaDevice;

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
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn dequantize(&self, elem_count: usize) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn data(&self) -> Result<std::borrow::Cow<'_, [u8]>> {
        crate::bail!("not implemented");
    }

    fn device(&self) -> impl AsRef<Self::Device> {
        &self.device
    }
}

impl QCudaStorage {
    pub fn fwd(
        &self,
        _self_shape: &crate::Shape,
        _storage: &CudaStorage,
        _layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn dequantize_f16(&self, elem_count: usize) -> Result<CudaStorage> {
        Err(Error::NotCompiledWithCudaSupport)
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    _device: &CudaDevice,
    _data: &[T],
) -> Result<super::QStorage> {
    Err(Error::NotCompiledWithCudaSupport)
}
