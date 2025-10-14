//! Traits to Define Backend Behavior
//!
use crate::op::{BinaryOpT, CmpOp, Op, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape, Tensor};
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

#[cfg(feature = "cuda")]
use crate::CudaDevice;
#[cfg(feature = "metal")]
use crate::{MetalDevice, MetalError};

pub trait BackendStorage: Sized + Clone + Send + Sync + Debug {
    type Device: BackendDevice<Self>;
    type Storage: BackendStorage;

    fn backprop_op(&self) -> Option<Op<Self>> {
        None
    }

    fn is_variable(&self) -> bool {
        false
    }

    fn to_tensor(
        storage: Arc<RwLock<Self>>,
        layout: Layout,
        _op: crate::op::BackpropOp<Self>,
        is_variable: bool,
    ) -> Tensor<Self> {
        Tensor::create(storage, layout, is_variable)
    }

    fn try_clone(&self, _: &Layout) -> Result<Self>;

    fn dtype(&self) -> DType;

    fn device(&self) -> Self::Device;

    // Maybe this should return a Cow instead so that no copy is done on the cpu case.
    fn to_cpu_storage(&self) -> Result<CpuStorage>;

    fn affine(&self, _: &Layout, _: f64, _: f64) -> Result<Self>;

    fn powf(&self, _: &Layout, _: f64) -> Result<Self>;

    fn elu(&self, _: &Layout, _: f64) -> Result<Self>;

    fn reduce_op(&self, _: ReduceOp, _: &Layout, _: &[usize]) -> Result<Self>;

    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self>;

    fn to_dtype(&self, _: &Layout, _: DType) -> Result<Self>;

    fn unary_impl<B: UnaryOpT>(&self, _: &Layout) -> Result<Self>;

    fn binary_impl<B: BinaryOpT>(&self, _: &Self, _: &Layout, _: &Layout) -> Result<Self>;

    fn where_cond(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout) -> Result<Self>;

    fn conv1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConv1D,
    ) -> Result<Self>;

    fn conv_transpose1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self>;

    fn conv2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConv2D,
    ) -> Result<Self>;

    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self>;

    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self>;
    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self>;
    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self>;
    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self>;

    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self>;

    fn scatter_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<()>;

    fn scatter_add_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<()>;

    fn index_select(&self, _: &Self, _: &Layout, _: &Layout, _: usize) -> Result<Self>;
    fn index_add(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<Self>;

    fn matmul(
        &self,
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &Layout,
        _: &Layout,
    ) -> Result<Self>;

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &Layout) -> Result<()>;

    #[allow(clippy::too_many_arguments)]
    // Similar to cudaMemcpy2D, though values are in elements and not in bytes.
    fn copy2d(
        &self,
        _: &mut Self,
        _d1: usize,
        _d2: usize,
        _src_stride1: usize,
        _dst_stride1: usize,
        _src_offset: usize,
        _dst_offset: usize,
    ) -> Result<()>;

    fn const_set(&mut self, _: crate::scalar::Scalar, _: &Layout) -> Result<()>;

    fn apply_op1(
        &self,
        _l: &Layout,
        _c: &dyn crate::CustomOp1<Self::Storage>,
    ) -> Result<(Self, Shape)>;

    fn apply_op2(
        &self,
        _l1: &Layout,
        _t2: &Self,
        _l2: &Layout,
        _c: &dyn crate::CustomOp2<Self::Storage>,
    ) -> Result<(Self, Shape)>;

    fn apply_op3(
        &self,
        _l1: &Layout,
        _t2: &Self,
        _l2: &Layout,
        _t3: &Self,
        _l3: &Layout,
        _c: &dyn crate::CustomOp3<Self::Storage>,
    ) -> Result<(Self, Shape)>;

    fn inplace_op1(&mut self, _l: &Layout, _c: &dyn crate::InplaceOp1) -> Result<()>;

    fn inplace_op2(
        &mut self,
        _l1: &Layout,
        _t2: &Self,
        _l2: &Layout,
        _c: &dyn crate::InplaceOp2,
    ) -> Result<()>;

    fn inplace_op3(
        &mut self,
        _l1: &Layout,
        _t2: &Self,
        _l2: &Layout,
        _t3: &Self,
        _l3: &Layout,
        _c: &dyn crate::InplaceOp3,
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

    fn zeros(&self, _shape: &Shape, _dtype: DType) -> Result<B>;

    /// # Safety
    /// This function is unsafe as it doesn't initialize the underlying data store.
    /// The caller should ensure that the data is properly initialized as early as possible
    /// after this call.
    unsafe fn alloc_uninit(&self, _shape: &Shape, _dtype: DType) -> Result<B>;

    fn storage_from_slice<T: crate::WithDType>(&self, _: &[T]) -> Result<B>;

    fn storage_from_cpu_storage(&self, _: &CpuStorage) -> Result<B>;

    fn storage_from_cpu_storage_owned(&self, _: CpuStorage) -> Result<B>;

    fn storage<A: crate::NdArray>(&self, array: A) -> Result<B>;

    fn storage_owned<S: crate::WithDType>(&self, data: Vec<S>) -> Result<B>;

    fn rand_uniform<T: crate::FloatDType>(&self, _: &Shape, _: DType, _: T, _: T) -> Result<B>;

    fn rand_normal<T: crate::FloatDType>(&self, _: &Shape, _: DType, _: T, _: T) -> Result<B>;

    fn set_seed(&self, _: u64) -> Result<()>;

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
