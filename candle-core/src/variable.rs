// Variables are wrappers around tensors that can be modified, they are typically used for holding
// weights and being modified by gradient descent.
// They are not cloneable by default to avoid having too many potential writers on the data.
// We also do not expose a public way to create variables as this would break the invariant that
// the tensor within a variable is actually with `is_variable` set to `true`.
use crate::{DType, Device, Result, Shape, Tensor};

/// A variable is a wrapper around a tensor, however variables can have their content modified
/// whereas tensors are immutable.
#[derive(Debug)]
pub struct Var(Tensor);

impl std::ops::Deref for Var {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl Var {
    pub fn zeros<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        let inner = Tensor::zeros_impl(shape, dtype, device, true)?;
        Ok(Self(inner))
    }

    pub fn ones<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        let inner = Tensor::ones_impl(shape, dtype, device, true)?;
        Ok(Self(inner))
    }

    pub fn rand<S: Into<Shape>>(
        s: S,
        dtype: DType,
        device: &Device,
        lo: f64,
        up: f64,
    ) -> Result<Self> {
        let inner = Tensor::rand_impl(s, dtype, device, lo, up, true)?;
        Ok(Self(inner))
    }

    pub fn randn<S: Into<Shape>>(
        s: S,
        dtype: DType,
        device: &Device,
        mean: f64,
        std: f64,
    ) -> Result<Self> {
        let inner = Tensor::randn_impl(s, dtype, device, mean, std, true)?;
        Ok(Self(inner))
    }

    /// Creates a new tensor on the specified device using the content and shape of the input.
    /// This is similar to `new` but the resulting tensor is a variable.
    pub fn new<A: crate::device::NdArray>(array: A, device: &Device) -> Result<Self> {
        let shape = array.shape()?;
        let inner = Tensor::new_impl(array, shape, device, true)?;
        Ok(Self(inner))
    }

    pub fn from_vec<S: Into<Shape>, D: crate::WithDType>(
        data: Vec<D>,
        shape: S,
        device: &Device,
    ) -> Result<Self> {
        let inner = Tensor::from_vec_impl(data, shape, device, true)?;
        Ok(Self(inner))
    }

    pub fn from_slice<S: Into<Shape>, D: crate::WithDType>(
        array: &[D],
        shape: S,
        device: &Device,
    ) -> Result<Self> {
        let inner = Tensor::new_impl(array, shape.into(), device, true)?;
        Ok(Self(inner))
    }

    pub fn as_tensor(&self) -> &Tensor {
        &self.0
    }

    /// Consumes this `Var` and return the underlying tensor.
    pub fn into_inner(self) -> Tensor {
        self.0
    }
}
