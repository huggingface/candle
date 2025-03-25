#![allow(unused)]
use super::GgmlDType;
use crate::{WgpuDevice, WgpuStorage, Error, Result};

pub struct QWgpuStorage {
    dtype: GgmlDType,
    device: WgpuDevice,
}

impl QWgpuStorage {
    pub fn zeros(_: &WgpuDevice, _: usize, _: GgmlDType) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &WgpuDevice {
        &self.device
    }

    pub fn dequantize(&self, _elem_count: usize) -> Result<WgpuStorage> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    pub fn dequantize_f16(&self, _elem_count: usize) -> Result<WgpuStorage> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    pub fn quantize(&mut self, _src: &WgpuStorage) -> Result<()> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        0
    }

    pub fn fwd(
        &self,
        _self_shape: &crate::Shape,
        _storage: &WgpuStorage,
        _layout: &crate::Layout,
    ) -> Result<(WgpuStorage, crate::Shape)> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    _device: &WgpuDevice,
    _data: &[T],
) -> Result<super::QStorage> {
    Err(Error::NotCompiledWithWgpuSupport)
}
