#![allow(unused)]
use super::GgmlDType;
use crate::{CudaDevice, CudaStorage, Error, Result};

pub struct QCudaStorage {
    dtype: GgmlDType,
    device: CudaDevice,
}

impl QCudaStorage {
    pub fn zeros(_: &CudaDevice, _: usize, _: GgmlDType) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    pub fn dequantize(&self, _elem_count: usize) -> Result<CudaStorage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn dequantize_f16(&self, _elem_count: usize) -> Result<CudaStorage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn quantize(&mut self, _src: &CudaStorage) -> Result<()> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        0
    }

    pub fn fwd(
        &self,
        _self_shape: &crate::Shape,
        _storage: &CudaStorage,
        _layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        Err(Error::NotCompiledWithCudaSupport)
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    _device: &CudaDevice,
    _data: &[T],
) -> Result<super::QStorage> {
    Err(Error::NotCompiledWithCudaSupport)
}
