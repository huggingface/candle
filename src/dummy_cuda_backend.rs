#![allow(dead_code)]
use crate::{CpuStorage, DType, Result, Shape};

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
        fail!()
    }

    pub(crate) fn ordinal(&self) -> usize {
        fail!()
    }

    pub(crate) fn zeros_impl(&self, _shape: &Shape, _dtype: DType) -> Result<CudaStorage> {
        fail!()
    }

    pub(crate) fn cuda_from_cpu_storage(&self, _: &CpuStorage) -> Result<CudaStorage> {
        fail!()
    }
}

#[derive(Debug, Clone)]
pub struct CudaStorage;

impl CudaStorage {
    pub fn dtype(&self) -> DType {
        fail!()
    }

    pub fn device(&self) -> CudaDevice {
        fail!()
    }

    pub(crate) fn to_cpu_storage(&self) -> Result<CpuStorage> {
        fail!()
    }

    pub(crate) fn affine_impl(&self, _: &Shape, _: &[usize], _: f64, _: f64) -> Result<Self> {
        fail!()
    }
}
