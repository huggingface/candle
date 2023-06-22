#![allow(dead_code)]
use crate::{CpuStorage, DType, Error, Result, Shape};

pub type CudaError = std::io::Error;

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
    pub fn try_clone(&self) -> Result<Self> {
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

    pub(crate) fn affine_impl(&self, _: &Shape, _: &[usize], _: f64, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub(crate) fn unary_impl<B: crate::op::UnaryOp>(&self, _: &Shape, _: &[usize]) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub(crate) fn binary_impl<B: crate::op::BinaryOp>(
        &self,
        _: &Self,
        _: &Shape,
        _: &[usize],
        _: &[usize],
    ) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }
}
