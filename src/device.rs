use crate::{CpuStorage, DType, Result, Shape, Storage};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Cuda { gpu_id: usize },
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
    pub(crate) fn ones(&self, shape: &Shape, dtype: DType) -> Result<Storage> {
        match self {
            Device::Cpu => {
                let storage = Storage::Cpu(CpuStorage::ones_impl(shape, dtype));
                Ok(storage)
            }
            Device::Cuda { gpu_id: _ } => {
                todo!()
            }
        }
    }

    pub(crate) fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Storage> {
        match self {
            Device::Cpu => {
                let storage = Storage::Cpu(CpuStorage::zeros_impl(shape, dtype));
                Ok(storage)
            }
            Device::Cuda { gpu_id: _ } => {
                todo!()
            }
        }
    }

    pub(crate) fn tensor<A: NdArray>(&self, array: A) -> Result<Storage> {
        match self {
            Device::Cpu => {
                let storage = Storage::Cpu(array.to_cpu_storage());
                Ok(storage)
            }
            Device::Cuda { gpu_id: _ } => {
                todo!()
            }
        }
    }
}
