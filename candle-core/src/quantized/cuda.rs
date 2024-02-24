use super::{GgmlDType, QStorage};
use crate::cuda_backend::WrapErr;
use crate::{CudaDevice, CudaStorage, Error, Result};

use cudarc::driver::{CudaSlice, DeviceSlice};

pub struct QCudaStorage {
    data: CudaSlice<u8>,
    dtype: GgmlDType,
    device: CudaDevice,
}

impl QCudaStorage {
    pub fn zeros(device: &CudaDevice, el_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size_in_bytes = el_count * dtype.type_size() / dtype.block_size();
        let data = device.alloc_zeros::<u8>(size_in_bytes).w()?;
        Ok(QCudaStorage {
            data,
            device: device.clone(),
            dtype,
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    pub fn dequantize(&self, _elem_count: usize) -> Result<CudaStorage> {
        Err(Error::NotCompiledWithCudaSupport.bt())
    }

    pub fn quantize(&mut self, _src: &CudaStorage) -> Result<()> {
        Err(Error::NotCompiledWithCudaSupport.bt())
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.data.len()
    }

    pub fn fwd(
        &self,
        _self_shape: &crate::Shape,
        _storage: &CudaStorage,
        _layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        Err(Error::NotCompiledWithCudaSupport.bt())
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &CudaDevice,
    data: &[T],
) -> Result<super::QStorage> {
    let size_in_bytes = core::mem::size_of_val(data);
    let data = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, size_in_bytes) };
    let data = device.htod_sync_copy(data).w()?;
    Ok(QStorage::Cuda(QCudaStorage {
        data,
        device: device.clone(),
        dtype: T::DTYPE,
    }))
}
