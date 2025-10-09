use crate::backend::{BackendDevice, BackendStorage};
use crate::cpu_backend::CpuDevice;
use crate::{CpuStorage, DType, Result, Shape, Storage, WithDType};
use std::convert::Into;

/// A `DeviceLocation` represents a physical device whereas multiple `Device`
/// can live on the same location (typically for cuda devices).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DeviceLocation {
    Cpu,
    Cuda { gpu_id: usize },
    Metal { gpu_id: usize },
}

/// Cpu, Cuda, or Metal
#[derive(Debug, Clone)]
pub enum Device {
    Cpu,
    Cuda(crate::CudaDevice),
    Metal(crate::MetalDevice),
}

impl AsRef<Device> for Device {
    fn as_ref(&self) -> &Device {
        self
    }
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

impl<S: WithDType> NdArray for Vec<S> {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        S::to_cpu_storage(self.as_slice())
    }
}

impl<S: WithDType> NdArray for Vec<&[S]> {
    fn shape(&self) -> Result<Shape> {
        if self.is_empty() {
            crate::bail!("empty array")
        }
        let n = self.len();
        let m = self[0].len();
        for v in self.iter() {
            if v.len() != m {
                crate::bail!("two elements have different len {m} {}", v.len())
            }
        }
        Ok(Shape::from((n, m)))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        let data = self.iter().copied().flatten().copied().collect::<Vec<_>>();
        S::to_cpu_storage_owned(data)
    }
}

impl<S: WithDType> NdArray for Vec<Vec<S>> {
    fn shape(&self) -> Result<Shape> {
        if self.is_empty() {
            crate::bail!("empty array")
        }
        let n = self.len();
        let m = self[0].len();
        for v in self.iter() {
            if v.len() != m {
                crate::bail!("two elements have different len {m} {}", v.len())
            }
        }
        Ok(Shape::from((n, m)))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        let len: usize = self.iter().map(|v| v.len()).sum();
        let mut dst = Vec::with_capacity(len);
        for v in self.iter() {
            dst.extend(v.iter().copied());
        }
        S::to_cpu_storage_owned(dst)
    }
}

impl<S: WithDType> NdArray for Vec<Vec<Vec<S>>> {
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
        if self.is_empty() {
            return S::to_cpu_storage_owned(vec![]);
        }
        let len: usize = self
            .iter()
            .map(|v| v.iter().map(|v| v.len()).sum::<usize>())
            .sum();
        let mut dst = Vec::with_capacity(len);
        for v1 in self.iter() {
            for v2 in v1.iter() {
                dst.extend(v2.iter().copied());
            }
        }
        S::to_cpu_storage_owned(dst)
    }
}

impl<S: WithDType> NdArray for Vec<Vec<Vec<Vec<S>>>> {
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
        let len: usize = self
            .iter()
            .map(|v| {
                v.iter()
                    .map(|v| v.iter().map(|v| v.len()).sum::<usize>())
                    .sum::<usize>()
            })
            .sum();
        let mut dst = Vec::with_capacity(len);
        for v1 in self.iter() {
            for v2 in v1.iter() {
                for v3 in v2.iter() {
                    dst.extend(v3.iter().copied());
                }
            }
        }
        S::to_cpu_storage_owned(dst)
    }
}

impl Device {
    pub fn new_cuda(ordinal: usize) -> Result<Self> {
        Ok(Self::Cuda(crate::CudaDevice::new(ordinal)?))
    }

    pub fn as_cuda_device(&self) -> Result<&crate::CudaDevice> {
        match self {
            Self::Cuda(d) => Ok(d),
            Self::Cpu => crate::bail!("expected a cuda device, got cpu"),
            Self::Metal(_) => crate::bail!("expected a cuda device, got Metal"),
        }
    }

    pub fn as_metal_device(&self) -> Result<&crate::MetalDevice> {
        match self {
            Self::Cuda(_) => crate::bail!("expected a metal device, got cuda"),
            Self::Cpu => crate::bail!("expected a metal device, got cpu"),
            Self::Metal(d) => Ok(d),
        }
    }

    pub fn new_cuda_with_stream(ordinal: usize) -> Result<Self> {
        Ok(Self::Cuda(crate::CudaDevice::new_with_stream(ordinal)?))
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

    pub fn supports_bf16(&self) -> bool {
        match self {
            Self::Cuda(_) | Self::Metal(_) => true,
            Self::Cpu => false,
        }
    }

    /// Return `BF16` for devices that support it, otherwise default to `F32`.
    pub fn bf16_default_to_f32(&self) -> DType {
        if self.supports_bf16() {
            DType::BF16
        } else {
            DType::F32
        }
    }

    pub fn cuda_if_available(ordinal: usize) -> Result<Self> {
        if crate::utils::cuda_is_available() {
            Self::new_cuda(ordinal)
        } else {
            Ok(Self::Cpu)
        }
    }

    pub fn metal_if_available(ordinal: usize) -> Result<Self> {
        if crate::utils::metal_is_available() {
            Self::new_metal(ordinal)
        } else {
            Ok(Self::Cpu)
        }
    }
}

impl BackendDevice<Storage> for Device {
    fn new(ordinal: usize) -> Result<Self> {
        if cfg!(feature = "metal") {
            Device::new_metal(ordinal)
        } else if cfg!(feature = "cuda") {
            Device::new_cuda(ordinal)
        } else {
            Ok(Device::Cpu)
        }
    }

    fn location(&self) -> DeviceLocation {
        match self {
            Self::Cpu => DeviceLocation::Cpu,
            Self::Cuda(device) => device.location(),
            Device::Metal(device) => device.location(),
        }
    }

    fn is_cpu(&self) -> bool {
        match self {
            Self::Cpu => true,
            Self::Cuda(_) => false,
            Self::Metal(_) => false,
        }
    }

    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Storage> {
        match self {
            Device::Cpu => {
                let storage = CpuDevice.zeros(shape, dtype)?;
                Ok(Storage::Cpu(storage))
            }
            Device::Cuda(device) => {
                let storage = device.zeros(shape, dtype)?;
                Ok(Storage::Cuda(storage))
            }
            Device::Metal(device) => {
                let storage = device.zeros(shape, dtype)?;
                Ok(Storage::Metal(storage))
            }
        }
    }

    /// # Safety
    /// This function is unsafe as it doesn't initialize the underlying data store.
    /// The caller should ensure that the data is properly initialized as early as possible
    /// after this call.
    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Storage> {
        match self {
            Self::Cpu => {
                let storage = CpuDevice.alloc_uninit(shape, dtype)?;
                Ok(Storage::Cpu(storage))
            }
            Self::Cuda(device) => {
                let storage = device.alloc_uninit(shape, dtype)?;
                Ok(Storage::Cuda(storage))
            }
            Self::Metal(device) => {
                let storage = device.alloc_uninit(shape, dtype)?;
                Ok(Storage::Metal(storage))
            }
        }
    }

    fn storage_from_slice<T: crate::WithDType>(&self, data: &[T]) -> Result<Storage> {
        match self {
            Device::Cpu => Ok(Storage::Cpu(data.to_cpu_storage())),
            Device::Cuda(device) => {
                let storage = device.storage_from_slice(data)?;
                Ok(Storage::Cuda(storage))
            }
            Device::Metal(device) => {
                let storage = device.storage_from_slice(data)?;
                Ok(Storage::Metal(storage))
            }
        }
    }

    fn storage_from_cpu_storage(&self, cpu: &CpuStorage) -> Result<Storage> {
        match self {
            Device::Cpu => Ok(Storage::Cpu(cpu.clone())),
            Device::Cuda(device) => {
                let storage = device.storage_from_cpu_storage(cpu)?;
                Ok(Storage::Cuda(storage))
            }
            Device::Metal(device) => {
                let storage = device.storage_from_cpu_storage(cpu)?;
                Ok(Storage::Metal(storage))
            }
        }
    }

    fn storage_from_cpu_storage_owned(&self, cpu: CpuStorage) -> Result<Storage> {
        match self {
            Device::Cpu => Ok(Storage::Cpu(cpu)),
            Device::Cuda(device) => {
                let storage = device.storage_from_cpu_storage_owned(cpu)?;
                Ok(Storage::Cuda(storage))
            }
            Device::Metal(device) => {
                let storage = device.storage_from_cpu_storage_owned(cpu)?;
                Ok(Storage::Metal(storage))
            }
        }
    }

    fn storage<A: NdArray>(&self, array: A) -> Result<Storage> {
        match self {
            Device::Cpu => Ok(Storage::Cpu(array.to_cpu_storage())),
            Device::Cuda(device) => {
                let storage = array.to_cpu_storage();
                let storage = device.storage_from_cpu_storage_owned(storage)?;
                Ok(Storage::Cuda(storage))
            }
            Device::Metal(device) => {
                let storage = array.to_cpu_storage();
                let storage = device.storage_from_cpu_storage_owned(storage)?;
                Ok(Storage::Metal(storage))
            }
        }
    }

    fn storage_owned<S: WithDType>(&self, data: Vec<S>) -> Result<Storage> {
        match self {
            Device::Cpu => Ok(Storage::Cpu(S::to_cpu_storage_owned(data))),
            Device::Cuda(device) => {
                let storage = S::to_cpu_storage_owned(data);
                let storage = device.storage_from_cpu_storage_owned(storage)?;
                Ok(Storage::Cuda(storage))
            }
            Device::Metal(device) => {
                let storage = S::to_cpu_storage_owned(data);
                let storage = device.storage_from_cpu_storage_owned(storage)?;
                Ok(Storage::Metal(storage))
            }
        }
    }

    fn rand_uniform<T: crate::FloatDType>(
        &self,
        shape: &Shape,
        dtype: DType,
        lo: T,
        up: T,
    ) -> Result<Storage> {
        match self {
            Device::Cpu => {
                let storage = CpuDevice.rand_uniform(shape, dtype, lo, up)?;
                Ok(Storage::Cpu(storage))
            }
            Device::Cuda(device) => {
                // TODO: Remove the special case if we start supporting generating f16/bf16 directly.
                if dtype == DType::F16 || dtype == DType::BF16 {
                    let storage = device.rand_uniform(shape, DType::F32, lo, up)?;
                    Storage::Cuda(storage).to_dtype(&crate::Layout::contiguous(shape), dtype)
                } else {
                    let storage = device.rand_uniform(shape, dtype, lo, up)?;
                    Ok(Storage::Cuda(storage))
                }
            }
            Device::Metal(device) => {
                if dtype == DType::F64 {
                    let storage = device.rand_uniform(shape, DType::F32, lo, up)?;
                    Ok(Storage::Metal(storage))
                } else {
                    let storage = device.rand_uniform(shape, dtype, lo, up)?;
                    Ok(Storage::Metal(storage))
                }
            }
        }
    }

    fn rand_normal<T: crate::FloatDType>(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: T,
        std: T,
    ) -> Result<Storage> {
        match self {
            Device::Cpu => {
                let storage = CpuDevice.rand_normal(shape, dtype, mean, std)?;
                Ok(Storage::Cpu(storage))
            }
            Device::Cuda(device) => {
                // TODO: Remove the special case if we start supporting generating f16/bf16 directly.
                if dtype == DType::F16 || dtype == DType::BF16 {
                    let storage = device.rand_normal(shape, DType::F32, mean, std)?;
                    Storage::Cuda(storage).to_dtype(&crate::Layout::contiguous(shape), dtype)
                } else {
                    let storage = device.rand_normal(shape, dtype, mean, std)?;
                    Ok(Storage::Cuda(storage))
                }
            }
            Device::Metal(device) => {
                let storage = device.rand_normal(shape, dtype, mean, std)?;
                Ok(Storage::Metal(storage))
            }
        }
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        match self {
            Self::Cpu => CpuDevice.set_seed(seed),
            Self::Cuda(c) => c.set_seed(seed),
            Self::Metal(m) => m.set_seed(seed),
        }
    }

    fn synchronize(&self) -> Result<()> {
        match self {
            Self::Cpu => Ok(()),
            Self::Cuda(d) => d.synchronize(),
            Self::Metal(d) => d.synchronize(),
        }
    }
}
