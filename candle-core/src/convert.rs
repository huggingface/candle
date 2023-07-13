//! Implement conversion traits for tensors
use crate::{Device, Error, Tensor, WithDType};
use half::{bf16, f16};
use std::convert::TryFrom;

impl<T: WithDType> TryFrom<&Tensor> for Vec<T> {
    type Error = Error;
    fn try_from(tensor: &Tensor) -> Result<Self, Self::Error> {
        tensor.to_vec1::<T>()
    }
}

impl<T: WithDType> TryFrom<&Tensor> for Vec<Vec<T>> {
    type Error = Error;
    fn try_from(tensor: &Tensor) -> Result<Self, Self::Error> {
        tensor.to_vec2::<T>()
    }
}

impl<T: WithDType> TryFrom<&Tensor> for Vec<Vec<Vec<T>>> {
    type Error = Error;
    fn try_from(tensor: &Tensor) -> Result<Self, Self::Error> {
        tensor.to_vec3::<T>()
    }
}

impl<T: WithDType> TryFrom<Tensor> for Vec<T> {
    type Error = Error;
    fn try_from(tensor: Tensor) -> Result<Self, Self::Error> {
        Vec::<T>::try_from(&tensor)
    }
}

impl<T: WithDType> TryFrom<Tensor> for Vec<Vec<T>> {
    type Error = Error;
    fn try_from(tensor: Tensor) -> Result<Self, Self::Error> {
        Vec::<Vec<T>>::try_from(&tensor)
    }
}

impl<T: WithDType> TryFrom<Tensor> for Vec<Vec<Vec<T>>> {
    type Error = Error;
    fn try_from(tensor: Tensor) -> Result<Self, Self::Error> {
        Vec::<Vec<Vec<T>>>::try_from(&tensor)
    }
}

impl<T: WithDType> TryFrom<&[T]> for Tensor {
    type Error = Error;
    fn try_from(v: &[T]) -> Result<Self, Self::Error> {
        Tensor::from_slice(v, v.len(), &Device::Cpu)
    }
}

impl<T: WithDType> TryFrom<Vec<T>> for Tensor {
    type Error = Error;
    fn try_from(v: Vec<T>) -> Result<Self, Self::Error> {
        let len = v.len();
        Tensor::from_vec(v, len, &Device::Cpu)
    }
}

macro_rules! from_tensor {
    ($typ:ident) => {
        impl TryFrom<&Tensor> for $typ {
            type Error = Error;

            fn try_from(tensor: &Tensor) -> Result<Self, Self::Error> {
                tensor.to_scalar::<$typ>()
            }
        }

        impl TryFrom<Tensor> for $typ {
            type Error = Error;

            fn try_from(tensor: Tensor) -> Result<Self, Self::Error> {
                $typ::try_from(&tensor)
            }
        }

        impl TryFrom<$typ> for Tensor {
            type Error = Error;

            fn try_from(v: $typ) -> Result<Self, Self::Error> {
                Tensor::new(v, &Device::Cpu)
            }
        }
    };
}

from_tensor!(f64);
from_tensor!(f32);
from_tensor!(f16);
from_tensor!(bf16);
from_tensor!(u32);
from_tensor!(u8);
