#![allow(dead_code)]
use crate::{CpuStorage, DType, Result, Shape};

pub type CudaError = std::io::Error;

#[derive(Debug, Clone)]
pub struct CudaDevice;

impl CudaDevice {
    pub(crate) fn new(_: usize) -> Result<Self> {
        unimplemented!("cuda support hasn't been enabled")
    }

    pub(crate) fn ordinal(&self) -> usize {
        unimplemented!("cuda support hasn't been enabled")
    }

    pub(crate) fn zeros_impl(&self, _shape: &Shape, _dtype: DType) -> Result<CudaStorage> {
        unimplemented!("cuda support hasn't been enabled")
    }

    pub(crate) fn cuda_from_cpu_storage(&self, _: &CpuStorage) -> Result<CudaStorage> {
        unimplemented!("cuda support hasn't been enabled")
    }
}

#[derive(Debug, Clone)]
pub struct CudaStorage;

impl CudaStorage {
    pub fn dtype(&self) -> DType {
        unimplemented!()
    }

    pub fn device(&self) -> CudaDevice {
        unimplemented!()
    }

    pub(crate) fn to_cpu_storage(&self) -> Result<CpuStorage> {
        unimplemented!()
    }

    pub(crate) fn affine_impl(&self, _: &Shape, _: &[usize], _: f64, _: f64) -> Result<Self> {
        unimplemented!()
    }
}
