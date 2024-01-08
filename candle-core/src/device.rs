use crate::backend::{BackendDevice, BackendStorage};
use crate::cpu_backend::CpuDevice;
use crate::custom_backend::CustomLocation;
use crate::{CpuStorage, DType, Result, Shape, Storage, WithDType};
use std::any::{Any, TypeId};

/// A `DeviceLocation` represents a physical device whereas multiple `Device`
/// can live on the same location (typically for cuda devices).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DeviceLocation {
    Cpu,
    Cuda {
        gpu_id: usize,
    },
    Metal {
        gpu_id: usize,
    },
    Custom {
        type_id: TypeId,
        custom_location: CustomLocation,
    },
}

#[derive(Debug, Clone)]
pub enum Device {
    Cpu,
    Cuda(crate::CudaDevice),
    Metal(crate::MetalDevice),
    Custom(crate::CustomDevice),
}

pub trait NdArray {
    fn shape(&self) -> Result<Shape>;

    fn to_cpu_storage(&self) -> CpuStorage;
}

impl<S: WithDType> NdArray for S {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(()))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        S::to_cpu_storage(&[*self])
    }
}

impl<S: WithDType, const N: usize> NdArray for &[S; N] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        S::to_cpu_storage(self.as_slice())
    }
}

impl<S: WithDType> NdArray for &[S] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        S::to_cpu_storage(self)
    }
}

impl<S: WithDType, const N: usize, const M: usize> NdArray for &[[S; N]; M] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((M, N)))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        S::to_cpu_storage_owned(self.concat())
    }
}

impl<S: WithDType, const N1: usize, const N2: usize, const N3: usize> NdArray
    for &[[[S; N3]; N2]; N1]
{
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((N1, N2, N3)))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        let mut vec = Vec::with_capacity(N1 * N2 * N3);
        for i1 in 0..N1 {
            for i2 in 0..N2 {
                vec.extend(self[i1][i2])
            }
        }
        S::to_cpu_storage_owned(vec)
    }
}

impl<S: WithDType, const N1: usize, const N2: usize, const N3: usize, const N4: usize> NdArray
    for &[[[[S; N4]; N3]; N2]; N1]
{
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((N1, N2, N3, N4)))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        let mut vec = Vec::with_capacity(N1 * N2 * N3 * N4);
        for i1 in 0..N1 {
            for i2 in 0..N2 {
                for i3 in 0..N3 {
                    vec.extend(self[i1][i2][i3])
                }
            }
        }
        S::to_cpu_storage_owned(vec)
    }
}

impl<S: NdArray> NdArray for Vec<S> {
    fn shape(&self) -> Result<Shape> {
        if self.is_empty() {
            crate::bail!("empty array")
        }
        let shape0 = self[0].shape()?;
        let n = self.len();
        for v in self.iter() {
            let shape = v.shape()?;
            if shape != shape0 {
                crate::bail!("two elements have different shapes {shape:?} {shape0:?}")
            }
        }
        Ok(Shape::from([[n].as_slice(), shape0.dims()].concat()))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        // This allocates intermediary memory and shouldn't be necessary.
        let storages = self.iter().map(|v| v.to_cpu_storage()).collect::<Vec<_>>();
        CpuStorage::concat(storages.as_slice()).unwrap()
    }
}

impl Device {
    pub fn new_cuda(ordinal: usize) -> Result<Self> {
        Ok(Self::Cuda(crate::CudaDevice::new(ordinal)?))
    }

    pub fn new_metal(ordinal: usize) -> Result<Self> {
        Ok(Self::Metal(crate::MetalDevice::new(ordinal)?))
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu)
    }

    pub fn is_cuda(&self) -> bool {
        matches!(self, Self::Cuda(_))
    }

    pub fn is_metal(&self) -> bool {
        matches!(self, Self::Metal(_))
    }

    pub fn cuda_if_available(ordinal: usize) -> Result<Self> {
        if crate::utils::cuda_is_available() {
            Self::new_cuda(ordinal)
        } else {
            Ok(Self::Cpu)
        }
    }
}

impl BackendDevice for Device {
    type Storage = Storage;
    type Location = DeviceLocation;

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        Ok(match self {
            Self::Cpu => Storage::Cpu(storage.clone()),
            Self::Cuda(device) => Storage::Cuda(device.storage_from_cpu_storage(storage)?),
            Self::Metal(device) => Storage::Metal(device.storage_from_cpu_storage(storage)?),
            Self::Custom(device) => Storage::Custom(device.storage_from_cpu_storage(storage)?),
        })
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        match self {
            Self::Cpu => CpuDevice.set_seed(seed),
            Self::Cuda(c) => c.set_seed(seed),
            Self::Metal(m) => m.set_seed(seed),
            Self::Custom(c) => c.set_seed(seed),
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (Self::Cpu, Self::Cpu) => true,
            (Self::Cuda(lhs), Self::Cuda(rhs)) => lhs.same_device(rhs),
            (Self::Metal(lhs), Self::Metal(rhs)) => lhs.same_device(rhs),
            (Self::Custom(lhs), Self::Custom(rhs)) => lhs.same_device(rhs),
            _ => false,
        }
    }

    fn location(&self) -> DeviceLocation {
        match self {
            Self::Cpu => DeviceLocation::Cpu,
            Self::Cuda(device) => DeviceLocation::Cuda {
                gpu_id: device.location(),
            },
            Self::Metal(device) => DeviceLocation::Metal {
                gpu_id: device.location(),
            },
            Self::Custom(device) => DeviceLocation::Custom {
                type_id: device.type_id(),
                custom_location: device.location(),
            },
        }
    }

    fn rand_uniform_f64(&self, shape: &Shape, dtype: DType, lo: f64, up: f64) -> Result<Storage> {
        match self {
            Device::Cpu => {
                let storage = CpuDevice.rand_uniform_f64(shape, dtype, lo, up)?;
                Ok(Storage::Cpu(storage))
            }
            Device::Cuda(device) => {
                // TODO: Remove the special case if we start supporting generating f16/bf16 directly.
                if dtype == DType::F16 || dtype == DType::BF16 {
                    let storage = device.rand_uniform_f64(shape, DType::F32, lo, up)?;
                    Storage::Cuda(storage).to_dtype(&crate::Layout::contiguous(shape), dtype)
                } else {
                    let storage = device.rand_uniform_f64(shape, dtype, lo, up)?;
                    Ok(Storage::Cuda(storage))
                }
            }
            Device::Metal(device) => {
                let storage = device.rand_uniform_f64(shape, dtype, lo, up)?;
                Ok(Storage::Metal(storage))
            }
            Device::Custom(device) => {
                let storage = device.rand_uniform_f64(shape, dtype, lo, up)?;
                Ok(Storage::Custom(storage))
            }
        }
    }

    fn rand_normal_f64(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<Storage> {
        match self {
            Device::Cpu => {
                let storage = CpuDevice.rand_normal_f64(shape, dtype, mean, std)?;
                Ok(Storage::Cpu(storage))
            }
            Device::Cuda(device) => {
                // TODO: Remove the special case if we start supporting generating f16/bf16 directly.
                if dtype == DType::F16 || dtype == DType::BF16 {
                    let storage = device.rand_normal_f64(shape, DType::F32, mean, std)?;
                    Storage::Cuda(storage).to_dtype(&crate::Layout::contiguous(shape), dtype)
                } else {
                    let storage = device.rand_normal_f64(shape, dtype, mean, std)?;
                    Ok(Storage::Cuda(storage))
                }
            }
            Device::Metal(device) => {
                let storage = device.rand_normal_f64(shape, dtype, mean, std)?;
                Ok(Storage::Metal(storage))
            }
            Device::Custom(device) => {
                let storage = device.rand_normal_f64(shape, dtype, mean, std)?;
                Ok(Storage::Custom(storage))
            }
        }
    }

    fn ones_impl(&self, shape: &Shape, dtype: DType) -> Result<Storage> {
        match self {
            Device::Cpu => {
                let storage = CpuDevice.ones_impl(shape, dtype)?;
                Ok(Storage::Cpu(storage))
            }
            Device::Cuda(device) => {
                let storage = device.ones_impl(shape, dtype)?;
                Ok(Storage::Cuda(storage))
            }
            Device::Metal(device) => {
                let storage = device.ones_impl(shape, dtype)?;
                Ok(Storage::Metal(storage))
            }
            Device::Custom(device) => {
                let storage = device.ones_impl(shape, dtype)?;
                Ok(Storage::Custom(storage))
            }
        }
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Storage> {
        match self {
            Device::Cpu => {
                let storage = CpuDevice.zeros_impl(shape, dtype)?;
                Ok(Storage::Cpu(storage))
            }
            Device::Cuda(device) => {
                let storage = device.zeros_impl(shape, dtype)?;
                Ok(Storage::Cuda(storage))
            }
            Device::Metal(device) => {
                let storage = device.zeros_impl(shape, dtype)?;
                Ok(Storage::Metal(storage))
            }
            Device::Custom(device) => {
                let storage = device.zeros_impl(shape, dtype)?;
                Ok(Storage::Custom(storage))
            }
        }
    }
}

impl Device {
    pub(crate) fn ones(&self, shape: &Shape, dtype: DType) -> Result<Storage> {
        self.ones_impl(shape, dtype)
    }

    pub(crate) fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Storage> {
        self.zeros_impl(shape, dtype)
    }

    pub(crate) fn rand_uniform<T: crate::FloatDType>(
        &self,
        lo: T,
        up: T,
        shape: &Shape,
    ) -> Result<Storage> {
        self.rand_uniform_f64(shape, T::DTYPE, lo.to_f64(), up.to_f64())
    }

    pub(crate) fn rand_normal<T: crate::FloatDType>(
        &self,
        mean: T,
        std: T,
        shape: &Shape,
    ) -> Result<Storage> {
        self.rand_normal_f64(shape, T::DTYPE, mean.to_f64(), std.to_f64())
    }

    pub(crate) fn storage<A: NdArray>(&self, array: A) -> Result<Storage> {
        match self {
            Device::Cpu => Ok(Storage::Cpu(array.to_cpu_storage())),
            Device::Cuda(device) => {
                let storage = array.to_cpu_storage();
                let storage = device.storage_from_cpu_storage(&storage)?;
                Ok(Storage::Cuda(storage))
            }
            Device::Metal(device) => {
                let storage = array.to_cpu_storage();
                let storage = device.storage_from_cpu_storage(&storage)?;
                Ok(Storage::Metal(storage))
            }
            Device::Custom(device) => {
                let storage = array.to_cpu_storage();
                let storage = device.storage_from_cpu_storage(&storage)?;
                Ok(Storage::Custom(storage))
            }
        }
    }

    pub(crate) fn storage_owned<S: WithDType>(&self, data: Vec<S>) -> Result<Storage> {
        match self {
            Device::Cpu => Ok(Storage::Cpu(S::to_cpu_storage_owned(data))),
            Device::Cuda(device) => {
                let storage = S::to_cpu_storage_owned(data);
                let storage = device.storage_from_cpu_storage(&storage)?;
                Ok(Storage::Cuda(storage))
            }
            Device::Metal(device) => {
                let storage = S::to_cpu_storage_owned(data);
                let storage = device.storage_from_cpu_storage(&storage)?;
                Ok(Storage::Metal(storage))
            }
            Device::Custom(device) => {
                let storage = S::to_cpu_storage_owned(data);
                let storage = device.storage_from_cpu_storage(&storage)?;
                Ok(Storage::Custom(storage))
            }
        }
    }
}
