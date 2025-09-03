#![allow(dead_code)]
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Error, Layout, Result, Shape};

#[derive(Debug, Clone)]
pub struct MetalDevice;

impl AsRef<MetalDevice> for MetalDevice {
    fn as_ref(&self) -> &MetalDevice {
        self
    }
}

#[derive(Debug, Clone)]
pub struct MetalStorage;

#[derive(thiserror::Error, Debug)]
pub enum MetalError {
    #[error("{0}")]
    Message(String),
}

impl From<String> for MetalError {
    fn from(e: String) -> Self {
        MetalError::Message(e)
    }
}

macro_rules! fail {
    () => {
        unimplemented!("metal support has not been enabled, add `metal` feature to enable.")
    };
}

impl crate::backend::BackendStorage for MetalStorage {
    type Device = MetalDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn dtype(&self) -> DType {
        fail!()
    }

    fn device(&self) -> impl AsRef<Self::Device> {
        fail!();
        #[allow(unreachable_code)]
        MetalDevice
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn affine(&self, _: &Layout, _: f64, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn powf(&self, _: &Layout, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn elu(&self, _: &Layout, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn reduce_op(&self, _: ReduceOp, _: &Layout, _: &[usize]) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn to_dtype(&self, _: &Layout, _: DType) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn unary_impl<B: UnaryOpT>(&self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn binary_impl<B: BinaryOpT>(&self, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn where_cond(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn conv1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn conv_transpose1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn conv2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }
    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn scatter_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<()> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn scatter_add_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<()> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn index_select(&self, _: &Self, _: &Layout, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn index_add(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn matmul(
        &self,
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &Layout,
        _: &Layout,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &Layout) -> Result<()> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn copy2d(
        &self,
        _: &mut Self,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
    ) -> Result<()> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn const_set(&mut self, _: crate::scalar::Scalar, _: &Layout) -> Result<()> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn apply_op1(&self, _l: &Layout, _c: &dyn crate::CustomOp1<Self>) -> Result<(Self, Shape)> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn apply_op2(
        &self,
        _l1: &Layout,
        _t2: &Self,
        _l2: &Layout,
        _c: &dyn crate::CustomOp2<Self>,
    ) -> Result<(Self, Shape)> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn apply_op3(
        &self,
        _l1: &Layout,
        _t2: &Self,
        _l2: &Layout,
        _t3: &Self,
        _l3: &Layout,
        _c: &dyn crate::CustomOp3<Self>,
    ) -> Result<(Self, Shape)> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn inplace_op1(&mut self, _: &Layout, _: &dyn crate::InplaceOp1) -> Result<()> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn inplace_op2(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &dyn crate::InplaceOp2,
    ) -> Result<()> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn inplace_op3(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &dyn crate::InplaceOp3,
    ) -> Result<()> {
        Err(Error::NotCompiledWithMetalSupport)
    }
}

impl crate::backend::BackendDevice<MetalStorage> for MetalDevice {
    fn new(_: usize) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn location(&self) -> crate::DeviceLocation {
        fail!()
    }

    fn same_device(&self, _: &MetalDevice) -> bool {
        fail!()
    }

    fn zeros(&self, _shape: &Shape, _dtype: DType) -> Result<MetalStorage> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    unsafe fn alloc_uninit(&self, _shape: &Shape, _dtype: DType) -> Result<MetalStorage> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn storage_from_slice<T: crate::WithDType>(&self, _: &[T]) -> Result<MetalStorage> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn storage_from_cpu_storage(&self, _: &CpuStorage) -> Result<MetalStorage> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn storage_from_cpu_storage_owned(&self, _: CpuStorage) -> Result<MetalStorage> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn storage<A: crate::NdArray>(&self, _: A) -> Result<MetalStorage> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn storage_owned<S: crate::WithDType>(&self, _: Vec<S>) -> Result<MetalStorage> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn rand_uniform<T: crate::FloatDType>(
        &self,
        _: &Shape,
        _: DType,
        _: T,
        _: T,
    ) -> Result<MetalStorage> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn rand_normal<T: crate::FloatDType>(
        &self,
        _: &Shape,
        _: DType,
        _: T,
        _: T,
    ) -> Result<MetalStorage> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn set_seed(&self, _: u64) -> Result<()> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}
