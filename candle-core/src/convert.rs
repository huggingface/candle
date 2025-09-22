//! Implement conversion traits for tensors
use crate::{
    BackendDevice, BackendStorage, CudaStorage, DType, Error, MetalStorage, Tensor, WithDType,
};
use crate::{CpuDevice, CpuStorage};
use float8::F8E4M3;
use half::{bf16, f16, slice::HalfFloatSliceExt};
use std::convert::TryFrom;

impl<B: BackendStorage, T: WithDType> TryFrom<&Tensor<B>> for Vec<T> {
    type Error = Error;
    fn try_from(tensor: &Tensor<B>) -> Result<Self, Self::Error> {
        tensor.to_vec1::<T>()
    }
}

impl<B: BackendStorage, T: WithDType> TryFrom<&Tensor<B>> for Vec<Vec<T>> {
    type Error = Error;
    fn try_from(tensor: &Tensor<B>) -> Result<Self, Self::Error> {
        tensor.to_vec2::<T>()
    }
}

impl<B: BackendStorage, T: WithDType> TryFrom<&Tensor<B>> for Vec<Vec<Vec<T>>> {
    type Error = Error;
    fn try_from(tensor: &Tensor<B>) -> Result<Self, Self::Error> {
        tensor.to_vec3::<T>()
    }
}

impl<B: BackendStorage, T: WithDType> TryFrom<Tensor<B>> for Vec<T> {
    type Error = Error;
    fn try_from(tensor: Tensor<B>) -> Result<Self, Self::Error> {
        Vec::<T>::try_from(&tensor)
    }
}

impl<B: BackendStorage, T: WithDType> TryFrom<Tensor<B>> for Vec<Vec<T>> {
    type Error = Error;
    fn try_from(tensor: Tensor<B>) -> Result<Self, Self::Error> {
        Vec::<Vec<T>>::try_from(&tensor)
    }
}

impl<B: BackendStorage, T: WithDType> TryFrom<Tensor<B>> for Vec<Vec<Vec<T>>> {
    type Error = Error;
    fn try_from(tensor: Tensor<B>) -> Result<Self, Self::Error> {
        Vec::<Vec<Vec<T>>>::try_from(&tensor)
    }
}

impl<T: WithDType> TryFrom<&[T]> for Tensor<CpuStorage> {
    type Error = Error;
    fn try_from(v: &[T]) -> Result<Self, Self::Error> {
        Tensor::from_slice(v, v.len(), &CpuDevice {})
    }
}

impl<T: WithDType> TryFrom<Vec<T>> for Tensor<CpuStorage> {
    type Error = Error;
    fn try_from(v: Vec<T>) -> Result<Self, Self::Error> {
        let len = v.len();
        Tensor::from_vec(v, len, &CpuDevice {})
    }
}

macro_rules! from_tensor {
    ($typ:ident) => {
        impl<B: BackendStorage> TryFrom<&Tensor<B>> for $typ {
            type Error = Error;

            fn try_from(tensor: &Tensor<B>) -> Result<Self, Self::Error> {
                tensor.to_scalar::<$typ>()
            }
        }

        impl<B: BackendStorage> TryFrom<Tensor<B>> for $typ {
            type Error = Error;

            fn try_from(tensor: Tensor<B>) -> Result<Self, Self::Error> {
                $typ::try_from(&tensor)
            }
        }

        impl TryFrom<$typ> for Tensor<CpuStorage> {
            type Error = Error;

            fn try_from(v: $typ) -> Result<Self, Self::Error> {
                Tensor::new(v, &CpuDevice {})
            }
        }
    };
}

from_tensor!(f64);
from_tensor!(f32);
from_tensor!(f16);
from_tensor!(bf16);
from_tensor!(i64);
from_tensor!(u32);
from_tensor!(u8);

impl<B: BackendStorage> Tensor<B> {
    pub fn write_bytes<W: std::io::Write>(&self, f: &mut W) -> crate::Result<()> {
        use byteorder::{LittleEndian, WriteBytesExt};

        let vs = self.flatten_all()?;
        match self.dtype() {
            DType::BF16 => {
                let vs = vs.to_vec1::<bf16>()?;
                for &v in vs.reinterpret_cast() {
                    f.write_u16::<LittleEndian>(v)?
                }
            }
            DType::F16 => {
                let vs = vs.to_vec1::<f16>()?;
                for &v in vs.reinterpret_cast() {
                    f.write_u16::<LittleEndian>(v)?
                }
            }
            DType::F32 => {
                // TODO: Avoid using a buffer when data is already on the CPU.
                for v in vs.to_vec1::<f32>()? {
                    f.write_f32::<LittleEndian>(v)?
                }
            }
            DType::F64 => {
                for v in vs.to_vec1::<f64>()? {
                    f.write_f64::<LittleEndian>(v)?
                }
            }
            DType::U32 => {
                for v in vs.to_vec1::<u32>()? {
                    f.write_u32::<LittleEndian>(v)?
                }
            }
            DType::I64 => {
                for v in vs.to_vec1::<i64>()? {
                    f.write_i64::<LittleEndian>(v)?
                }
            }
            DType::U8 => {
                let vs = vs.to_vec1::<u8>()?;
                f.write_all(&vs)?;
            }
            DType::F8E4M3 => {
                for v in vs.to_vec1::<F8E4M3>()? {
                    f.write_u8(v.to_bits())?
                }
            }
        }
        Ok(())
    }
}

pub trait TryConvertStorage<T: BackendStorage>: BackendStorage {
    /// Performs the conversion.
    fn convert(storage: T, device: &Self::Device) -> Result<Self, Error>;
}

/*
impl<B: BackendStorage> TryConvertStorage<B, B> for B::Device {
    fn convert(&self, storage: B) -> Result<B, Error> {
        Ok(storage)
    }
}
 */
impl TryConvertStorage<CpuStorage> for CpuStorage {
    fn convert(storage: CpuStorage, _: &Self::Device) -> Result<Self, Error> {
        Ok(storage)
    }
}

impl TryConvertStorage<CudaStorage> for CpuStorage {
    fn convert(storage: CudaStorage, _: &Self::Device) -> Result<Self, Error> {
        storage.to_cpu_storage()
    }
}

impl TryConvertStorage<MetalStorage> for CpuStorage {
    fn convert(storage: MetalStorage, _: &Self::Device) -> Result<Self, Error> {
        storage.to_cpu_storage()
    }
}

/*
pub trait ToDevice<T, U>: Sized
where
    Self: ConvertStorage<T, U>,
{
    /// Performs the conversion.
    fn try_to_device(&self, device: &Self) -> Result<U, Error>;
}

pub trait TryConvertStorage<T, U>: BackendDevice<U>
where
    T: BackendStorage,
    U: BackendStorage<Device = Self>,
{
    /// Performs the conversion.
    fn try_convert(&self, storage: T) -> Result<U, Error>;
}

impl<B: BackendStorage> TryConvertStorage<B, B> for B::Device
where
    IsSame<B, B>: Condition<Value = True>,
{
    fn try_convert(&self, storage: B) -> Result<B, Error> {
        Ok(storage)
    }
}

impl<B: BackendStorage> TryConvertStorage<B, CpuStorage> for CpuDevice
where
    IsSame<B, CpuStorage>: Condition<Value = False>,
{
    fn try_convert(&self, storage: B) -> Result<CpuStorage, Error> {
        storage.to_cpu_storage()
    }
}

impl TryConvertStorage<CpuStorage, CudaStorage> for CudaDevice {
    fn try_convert(&self, storage: CpuStorage) -> Result<CudaStorage, Error> {
        self.storage_from_cpu_storage(&storage)
    }
}

impl TryConvertStorage<CpuStorage, MetalStorage> for MetalDevice {
    fn try_convert(&self, storage: CpuStorage) -> Result<MetalStorage, Error> {
        self.storage_from_cpu_storage(&storage)
    }
}

pub trait TryToDevice<T, U>: Sized
where
    T: BackendStorage,
    U: BackendStorage,
{
    /// The type returned in the event of a conversion error.
    type Error;

    /// Performs the conversion.
    fn try_to_device(&self, device: &U::Device) -> Result<Tensor<U>, Error>;
}
 */
