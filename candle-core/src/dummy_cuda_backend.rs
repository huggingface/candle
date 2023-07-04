#![allow(dead_code)]
use crate::{CpuStorage, DType, Error, Layout, Result, Shape};

#[derive(thiserror::Error, Debug)]
pub enum DummyError {}
pub type CudaError = DummyError;

#[derive(Debug, Clone)]
pub struct CudaDevice;

macro_rules! fail {
    () => {
        unimplemented!("cuda support has not been enabled")
    };
}

impl CudaDevice {
    pub(crate) fn new(_: usize) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub(crate) fn same_id(&self, _: &Self) -> bool {
        true
    }

    pub(crate) fn ordinal(&self) -> usize {
        fail!()
    }

    pub(crate) fn zeros_impl(&self, _shape: &Shape, _dtype: DType) -> Result<CudaStorage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub(crate) fn ones_impl(&self, _shape: &Shape, _dtype: DType) -> Result<CudaStorage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub(crate) fn cuda_from_cpu_storage(&self, _: &CpuStorage) -> Result<CudaStorage> {
        Err(Error::NotCompiledWithCudaSupport)
    }
}

#[derive(Debug)]
pub struct CudaStorage;

impl CudaStorage {
    pub fn try_clone(&self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn dtype(&self) -> DType {
        fail!()
    }

    pub fn device(&self) -> &CudaDevice {
        fail!()
    }

    pub(crate) fn to_cpu_storage(&self) -> Result<CpuStorage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub(crate) fn affine(&self, _: &Layout, _: f64, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub(crate) fn sum(&self, _: &Layout, _: &[usize]) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub(crate) fn divide_by_sum_over_dim(&mut self, _: &Shape, _: usize) -> Result<()> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub(crate) fn to_dtype(&self, _: &Layout, _: DType) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub(crate) fn unary_impl<B: crate::op::UnaryOp>(&self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub(crate) fn binary_impl<B: crate::op::BinaryOp>(
        &self,
        _: &Self,
        _: &Layout,
        _: &Layout,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub(crate) fn where_cond(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub(crate) fn conv1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub(crate) fn embedding(&self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub(crate) fn matmul(
        &self,
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &Layout,
        _: &Layout,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub(crate) fn copy_strided_src(&self, _: &mut Self, _: usize, _: &Layout) -> Result<()> {
        Err(Error::NotCompiledWithCudaSupport)
    }
}
