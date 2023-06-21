use crate::{CpuStorage, DType, Result, Shape};
use cudarc::driver::CudaSlice;

pub(crate) type Error = cudarc::driver::DriverError;

#[derive(Debug, Clone)]
pub struct CudaDevice(std::sync::Arc<cudarc::driver::CudaDevice>);

impl CudaDevice {
    pub(crate) fn new(ordinal: usize) -> Result<Self> {
        let device = cudarc::driver::CudaDevice::new(ordinal)?;
        Ok(Self(device))
    }

    pub(crate) fn ordinal(&self) -> usize {
        self.0.ordinal()
    }

    pub(crate) fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        match dtype {
            DType::F32 => {
                let data = self.0.alloc_zeros::<f32>(elem_count)?;
                Ok(CudaStorage::F32(data))
            }
            DType::F64 => {
                let data = self.0.alloc_zeros::<f64>(elem_count)?;
                Ok(CudaStorage::F64(data))
            }
        }
    }

    pub(crate) fn cuda_from_cpu_storage(&self, storage: &CpuStorage) -> Result<CudaStorage> {
        match storage {
            CpuStorage::F32(storage) => {
                let data = self.0.htod_sync_copy(storage)?;
                Ok(CudaStorage::F32(data))
            }
            CpuStorage::F64(storage) => {
                let data = self.0.htod_sync_copy(storage)?;
                Ok(CudaStorage::F64(data))
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum CudaStorage {
    F32(CudaSlice<f32>),
    F64(CudaSlice<f64>),
}

impl CudaStorage {
    pub fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
        }
    }

    pub fn device(&self) -> CudaDevice {
        match self {
            Self::F32(slice) => CudaDevice(slice.device()),
            Self::F64(slice) => CudaDevice(slice.device()),
        }
    }

    pub(crate) fn to_cpu_storage(&self) -> Result<CpuStorage> {
        match self {
            Self::F32(slice) => {
                let dev = slice.device();
                let cpu_storage = dev.dtoh_sync_copy(slice)?;
                Ok(CpuStorage::F32(cpu_storage))
            }
            Self::F64(slice) => {
                let dev = slice.device();
                let cpu_storage = dev.dtoh_sync_copy(slice)?;
                Ok(CpuStorage::F64(cpu_storage))
            }
        }
    }
}
