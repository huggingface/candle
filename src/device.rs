use crate::{CpuStorage, DType, Result, Shape, Storage};

/// A `DeviceLocation` represents a physical device whereas multiple `Device`
/// can live on the same location (typically for cuda devices).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DeviceLocation {
    Cpu,
    Cuda { gpu_id: usize },
}

#[derive(Debug, Clone)]
pub enum Device {
    Cpu,
    Cuda(std::sync::Arc<cudarc::driver::CudaDevice>),
}

// TODO: Should we back the cpu implementation using the NdArray crate or similar?
pub trait NdArray {
    fn shape(&self) -> Result<Shape>;

    fn to_cpu_storage(&self) -> CpuStorage;
}

impl<S: crate::WithDType> NdArray for S {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(()))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        S::to_cpu_storage(&[*self])
    }
}

impl<S: crate::WithDType, const N: usize> NdArray for &[S; N] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        S::to_cpu_storage(self.as_slice())
    }
}

impl<S: crate::WithDType> NdArray for &[S] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        S::to_cpu_storage(self)
    }
}

impl<S: crate::WithDType, const N: usize, const M: usize> NdArray for &[[S; N]; M] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((M, N)))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        S::to_cpu_storage_owned(self.concat())
    }
}

impl Device {
    pub fn new_cuda(ordinal: usize) -> Result<Self> {
        let device = cudarc::driver::CudaDevice::new(ordinal)?;
        Ok(Self::Cuda(device))
    }

    pub fn location(&self) -> DeviceLocation {
        match self {
            Self::Cpu => DeviceLocation::Cpu,
            Self::Cuda(device) => DeviceLocation::Cuda {
                gpu_id: device.ordinal(),
            },
        }
    }

    pub(crate) fn ones(&self, shape: &Shape, dtype: DType) -> Result<Storage> {
        match self {
            Device::Cpu => {
                let storage = CpuStorage::ones_impl(shape, dtype);
                Ok(Storage::Cpu(storage))
            }
            Device::Cuda(device) => {
                // TODO: Instead of allocating memory on the host and transfering it,
                // allocate some zeros on the device and use a shader to set them to 1.
                let storage = device.htod_copy(vec![1f32; shape.elem_count()])?;
                Ok(Storage::Cuda(storage))
            }
        }
    }

    pub(crate) fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Storage> {
        match self {
            Device::Cpu => {
                let storage = CpuStorage::zeros_impl(shape, dtype);
                Ok(Storage::Cpu(storage))
            }
            Device::Cuda(device) => {
                let storage = device.alloc_zeros::<f32>(shape.elem_count())?;
                Ok(Storage::Cuda(storage))
            }
        }
    }

    pub(crate) fn tensor<A: NdArray>(&self, array: A) -> Result<Storage> {
        match self {
            Device::Cpu => Ok(Storage::Cpu(array.to_cpu_storage())),
            Device::Cuda(device) => {
                // TODO: Avoid making a copy through the cpu.
                match array.to_cpu_storage() {
                    CpuStorage::F64(_) => {
                        todo!()
                    }
                    CpuStorage::F32(data) => {
                        let storage = device.htod_copy(data)?;
                        Ok(Storage::Cuda(storage))
                    }
                }
            }
        }
    }
}
