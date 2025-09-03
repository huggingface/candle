//! Implementation of the Cuda backend when Cuda support has not been compiled in.
//!
#![allow(dead_code)]
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Error, Layout, Result, Shape};

#[derive(Debug, Clone)]
pub struct CudaDevice;

impl AsRef<CudaDevice> for CudaDevice {
    fn as_ref(&self) -> &CudaDevice {
        self
    }
}

#[derive(Debug, Clone)]
pub struct CudaStorage;

macro_rules! fail {
    () => {
        unimplemented!("cuda support has not been enabled, add `cuda` feature to enable.")
    };
}

impl CudaDevice {
    pub fn new_with_stream(_: usize) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }
}

impl crate::backend::BackendStorage for CudaStorage {
    type Device = CudaDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn dtype(&self) -> DType {
        fail!()
    }

    fn device(&self) -> impl AsRef<Self::Device> {
        fail!();
        #[allow(unreachable_code)]
        CudaDevice
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn affine(&self, _: &Layout, _: f64, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn powf(&self, _: &Layout, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn elu(&self, _: &Layout, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn reduce_op(&self, _: ReduceOp, _: &Layout, _: &[usize]) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn to_dtype(&self, _: &Layout, _: DType) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn unary_impl<B: UnaryOpT>(&self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn binary_impl<B: BinaryOpT>(&self, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn where_cond(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn conv1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn conv_transpose1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn conv2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }
    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
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
        Err(Error::NotCompiledWithCudaSupport)
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
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn index_select(&self, _: &Self, _: &Layout, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
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
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn matmul(
        &self,
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &Layout,
        _: &Layout,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &Layout) -> Result<()> {
        Err(Error::NotCompiledWithCudaSupport)
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
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn const_set(&mut self, _: crate::scalar::Scalar, _: &Layout) -> Result<()> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn apply_op1(&self, _l: &Layout, _c: &dyn crate::CustomOp1<Self>) -> Result<(Self, Shape)> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn apply_op2(
        &self,
        _l1: &Layout,
        _t2: &Self,
        _l2: &Layout,
        _c: &dyn crate::CustomOp2<Self>,
    ) -> Result<(Self, Shape)> {
        Err(Error::NotCompiledWithCudaSupport)
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
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn inplace_op1(&mut self, _l: &Layout, _c: &dyn crate::InplaceOp1) -> Result<()> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn inplace_op2(
        &mut self,
        _l1: &Layout,
        _t2: &Self,
        _l2: &Layout,
        _c: &dyn crate::InplaceOp2,
    ) -> Result<()> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn inplace_op3(
        &mut self,
        _l1: &Layout,
        _t2: &Self,
        _l2: &Layout,
        _t3: &Self,
        _l3: &Layout,
        _c: &dyn crate::InplaceOp3,
    ) -> Result<()> {
        Err(Error::NotCompiledWithCudaSupport)
    }
}

impl crate::backend::BackendDevice<CudaStorage> for CudaDevice {
    fn new(_: usize) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn location(&self) -> crate::DeviceLocation {
        fail!()
    }

    fn same_device(&self, _: &CudaDevice) -> bool {
        fail!()
    }

    fn zeros(&self, _shape: &Shape, _dtype: DType) -> Result<CudaStorage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    unsafe fn alloc_uninit(&self, _shape: &Shape, _dtype: DType) -> Result<CudaStorage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn storage_from_slice<T: crate::WithDType>(&self, _: &[T]) -> Result<CudaStorage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn storage_from_cpu_storage(&self, _: &CpuStorage) -> Result<CudaStorage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn storage_from_cpu_storage_owned(&self, _: CpuStorage) -> Result<CudaStorage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn storage<A: crate::NdArray>(&self, _: A) -> Result<CudaStorage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn storage_owned<S: crate::WithDType>(&self, _: Vec<S>) -> Result<CudaStorage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn rand_uniform<T: crate::FloatDType>(
        &self,
        _: &Shape,
        _: DType,
        _: T,
        _: T,
    ) -> Result<CudaStorage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn rand_normal<T: crate::FloatDType>(
        &self,
        _: &Shape,
        _: DType,
        _: T,
        _: T,
    ) -> Result<CudaStorage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn set_seed(&self, _: u64) -> Result<()> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}

/// This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
/// allowed with f16 GEMMs.
pub fn gemm_reduced_precision_f16() -> bool {
    true
}

/// This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
/// allowed with f16 GEMMs.
pub fn set_gemm_reduced_precision_f16(_: bool) {}

/// This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
/// allowed with bf16 GEMMs.
pub fn gemm_reduced_precision_bf16() -> bool {
    true
}

/// This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
/// allowed with bf16 GEMMs.
pub fn set_gemm_reduced_precision_bf16(_: bool) {}

/// This bool controls whether reduced precision reductions (e.g., with tf32 accumulation type) are
/// allowed with f32 GEMMs.
pub fn gemm_reduced_precision_f32() -> bool {
    true
}

/// This bool controls whether reduced precision reductions (e.g., with tf32 accumulation type) are
/// allowed with f32 GEMMs.
pub fn set_gemm_reduced_precision_f32(_b: bool) {}
