//! Traits to Define Backend Behavior
//!
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape};
use std::fmt::Debug;

#[cfg(feature = "cuda")]
use crate::CudaDevice;
#[cfg(feature = "metal")]
use crate::{MetalDevice, MetalError};

pub trait BackendStorage: Sized + Clone + Send + Sync + Debug {
    type Device: BackendDevice<Self>;

    fn try_clone(&self, _: &Layout) -> Result<Self>;

    fn dtype(&self) -> DType;

    fn device(&self) -> Self::Device;

    // Maybe this should return a Cow instead so that no copy is done on the cpu case.
    fn to_cpu_storage(&self) -> Result<CpuStorage>;

    fn affine(&self, layout: &Layout, a: f64, b: f64) -> Result<Self>;

    fn powf(&self, layout: &Layout, e: f64) -> Result<Self>;

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self>;

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, reduce_dims: &[usize]) -> Result<Self>;

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self>;

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self>;

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self>;

    fn binary_impl<B: BinaryOpT>(&self, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout)
        -> Result<Self>;

    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self>;

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self>;

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self>;

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self>;

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self>;

    fn avg_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self>;

    fn max_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self>;

    fn upsample_nearest1d(&self, layout: &Layout, sz: usize) -> Result<Self>;

    fn upsample_nearest2d(&self, layout: &Layout, h: usize, w: usize) -> Result<Self>;

    fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self>;

    fn scatter_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()>;

    fn scatter_add_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()>;

    fn index_select(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self>;
    fn index_add(
        &self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self>;

    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self>;

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()>;

    #[allow(clippy::too_many_arguments)]
    // Similar to cudaMemcpy2D, though values are in elements and not in bytes.
    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_s: usize,
        dst_s: usize,
        src_o: usize,
        dst_o: usize,
    ) -> Result<()>;

    fn const_set(&mut self, s: crate::scalar::Scalar, l: &Layout) -> Result<()>;

    fn apply_op1(&self, l: &Layout, c: &dyn crate::CustomOp1<Self>) -> Result<(Self, Shape)>;

    fn apply_op2(
        &self,
        l1: &Layout,
        t2: &Self,
        l2: &Layout,
        c: &dyn crate::CustomOp2<Self>,
    ) -> Result<(Self, Shape)>;

    fn apply_op3(
        &self,
        l1: &Layout,
        t2: &Self,
        l2: &Layout,
        t3: &Self,
        l3: &Layout,
        c: &dyn crate::CustomOp3<Self>,
    ) -> Result<(Self, Shape)>;

    fn inplace_op1(&mut self, l: &Layout, c: &dyn crate::InplaceOp1) -> Result<()>;

    fn inplace_op2(
        &mut self,
        l1: &Layout,
        t2: &Self,
        l2: &Layout,
        c: &dyn crate::InplaceOp2,
    ) -> Result<()>;

    fn inplace_op3(
        &mut self,
        l1: &Layout,
        t2: &Self,
        l2: &Layout,
        t3: &Self,
        l3: &Layout,
        c: &dyn crate::InplaceOp3,
    ) -> Result<()>;
}

pub trait BackendDevice<B: BackendStorage>: Sized + std::fmt::Debug + Clone + Send + Sync {
    const SUPPORTS_BF16: bool = false;

    // TODO: Make the usize generic and part of a generic DeviceLocation.
    fn new(_: usize) -> Result<Self>;

    fn location(&self) -> crate::DeviceLocation;

    fn same_device(&self, device: &Self) -> bool {
        self.location() == device.location()
    }

    fn is_cpu(&self) -> bool;

    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<B>;

    /// # Safety
    /// This function is unsafe as it doesn't initialize the underlying data store.
    /// The caller should ensure that the data is properly initialized as early as possible
    /// after this call.
    unsafe fn alloc_uninit(&self, _hape: &Shape, dtype: DType) -> Result<B>;

    fn storage_from_slice<T: crate::WithDType>(&self, data: &[T]) -> Result<B>;

    fn storage_from_cpu_storage(&self, cpu_storage: &CpuStorage) -> Result<B>;

    fn storage_from_cpu_storage_owned(&self, cpu_storage: CpuStorage) -> Result<B>;

    fn storage<A: crate::NdArray>(&self, array: A) -> Result<B>;

    fn storage_owned<S: crate::WithDType>(&self, data: Vec<S>) -> Result<B>;

    fn rand_uniform<T: crate::FloatDType>(
        &self,
        shape: &Shape,
        dtype: DType,
        low: T,
        high: T,
    ) -> Result<B>;

    fn rand_normal<T: crate::FloatDType>(
        &self,
        shape: &Shape,
        dtype: DType,
        low: T,
        high: T,
    ) -> Result<B>;

    fn set_seed(&self, seed: u64) -> Result<()>;

    /// Synchronize should block until all the operations on the device are completed.
    fn synchronize(&self) -> Result<()>;
}

pub trait UgDevice {
    type UgFunction;

    fn compile(
        &self,
        _func_name: &'static str,
        _kernel: ug::lang::ssa::Kernel,
    ) -> Result<Self::UgFunction>;
}

#[cfg(all(feature = "cuda", not(target_arch = "wasm32")))]
impl UgDevice for CudaDevice {
    type UgFunction = candle_core::cudarc::driver::CudaFunction;
    fn compile(
        &self,
        func_name: &'static str,
        kernel: ug::lang::ssa::Kernel,
    ) -> Result<Self::UgFunction> {
        let mut buf = vec![];
        ug_cuda::code_gen::gen(&mut buf, func_name, &kernel)?;
        let cuda_code = String::from_utf8(buf)?;
        let opts = cudarc::nvrtc::CompileOptions {
            use_fast_math: Some(true),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::safe::compile_ptx_with_opts(cuda_code, opts).w()?;
        let module = self.context().load_module(ptx).w()?;
        let func = module.load_function(func_name).w()?;
        Ok(CudaFunc {
            func,
            stream: self.stream().clone(),
        })
    }
}

#[cfg(all(feature = "metal", not(target_arch = "wasm32")))]
impl UgDevice for MetalDevice {
    type UgFunction = candle_metal_kernels::metal::ComputePipeline;

    fn compile(
        &self,
        func_name: &'static str,
        kernel: ug::lang::ssa::Kernel,
    ) -> Result<Self::UgFunction> {
        let mut buf = vec![];
        ug_metal::code_gen::gen(&mut buf, func_name, &kernel)?;
        let metal_code = String::from_utf8(buf)?;
        let lib = self
            .device
            .new_library_with_source(&metal_code, None)
            .map_err(MetalError::from)?;
        let func = lib
            .get_function(func_name, None)
            .map_err(MetalError::from)?;
        let pl = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(MetalError::from)?;
        Ok(pl)
    }
}
