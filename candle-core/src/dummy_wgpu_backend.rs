#![allow(dead_code)]
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Error, Layout, Result, Shape};

#[derive(Debug, Clone)]
pub struct WgpuDevice;

#[derive(Debug)]
pub struct WgpuStorage;


pub enum Backend {
    /// Dummy backend, used for testing.
    Empty = 0,
    /// Vulkan API (Windows, Linux, Android, MacOS via `vulkan-portability`/MoltenVK)
    Vulkan = 1,
    /// Metal API (Apple platforms)
    Metal = 2,
    /// Direct3D-12 (Windows)
    Dx12 = 3,
    /// OpenGL 3.3+ (Windows), OpenGL ES 3.0+ (Linux, Android, MacOS via Angle), and WebGL2
    Gl = 4,
    /// WebGPU in the browser
    BrowserWebGpu = 5,
}

#[derive(Debug, Clone, std::marker::Copy)]
pub struct WgpuBackends(u32);

impl WgpuBackends{
    pub fn vulkan() -> Self{
        WgpuBackends(1 << Backend::Vulkan as u32)
    }

    pub fn gl() -> Self{
        WgpuBackends(1 << Backend::Gl as u32)
    }

    pub fn metal() -> Self{
        WgpuBackends(1 << Backend::Metal as u32)
    }

    pub fn dx12() -> Self{
        WgpuBackends(1 << Backend::Dx12 as u32)
    }

    pub fn browser_webgpu() -> Self{
        WgpuBackends(1 << Backend::BrowserWebGpu as u32)
    }

    pub fn primary() -> Self{
        Self::vulkan() | Self::metal() | Self::dx12() | Self::browser_webgpu()
    }

    pub fn secondary() -> Self{
        Self::gl()
    }
}

impl Default for WgpuBackends {
    fn default() -> Self {
        WgpuBackends::primary() | WgpuBackends::secondary()
    }
}

impl std::ops::BitOr for WgpuBackends{
    type Output = WgpuBackends;

    fn bitor(self, rhs: Self) -> Self::Output {
        return WgpuBackends(self.0 | rhs.0);
    }
}

impl std::ops::BitAnd for WgpuBackends{
    type Output = bool;

    fn bitand(self, rhs: Self) -> Self::Output {
        return (self.0 & rhs.0) > 0;
    }
}


#[derive(Debug)]
pub struct WgpuDeviceConfig{
    pub meta_buffer_size : u32, //the size of the buffer used for storing meta information (e.g. input layouts)
    pub max_workload_size : u64, //specifys how much max floating point operations will be queued in one single command. (e.g. a matrix multiplication of 1000x1000 * 1000x1000 would be about 1gb operations, so only 2 of theses may be queued in one command buffer) 
    pub buffer_cached_max_allowed_size : u64,//maximum size for cached wgpu::buffers. When this size is reached, free buffers will be deleted until only 75% of this max size is used. 
                                             //if this value is to low for the desired model, the performance may drop significatly(e.g. model needs at least 2gb of data, if this value would be e.g. only 100mb all free buffers would be deleted after each command)
    pub use_cache : bool, 
    pub flush_gpu_before_buffer_init : bool, //when data is copied from cpu to the wgpu device, all previous commands may be flushed, to allow other buffers to be freed and reused. 
                                            //But on webGpu this may not be optimal, as we can not wait for commands to finish (as this functin is not asyny) 
    pub buffer_mapping_size : u32,

    pub backend : WgpuBackends
}

impl Default for WgpuDeviceConfig {
    fn default() -> WgpuDeviceConfig {
        WgpuDeviceConfig {
            meta_buffer_size : 10*1024*1024,
            max_workload_size :  1024u64*1024*1024*2, 
            buffer_cached_max_allowed_size : 1024*1024*1024*8,                                        
            use_cache : true,
            flush_gpu_before_buffer_init : true,
            buffer_mapping_size : 3,
            backend: WgpuBackends::metal() | WgpuBackends::vulkan(), //directx shader compilation is much slower than vulkan. (like 300secs vs 5s there is a faster copmiler, but this would need additional .dlls, and with this compilations needs 30s as well)
        }
    }
}



#[derive(thiserror::Error, Debug)]
pub enum WebGpuError {
    #[error("{0}")]
    Message(String),
}

impl From<String> for WebGpuError {
    fn from(e: String) -> Self {
        WebGpuError::Message(e)
    }
}

macro_rules! fail {
    () => {
        unimplemented!("wgpu support has not been enabled, add `wgpu` feature to enable.")
    };
}

impl WgpuStorage{
    pub async fn to_cpu_storage_async(&self) -> crate::Result<crate::CpuStorage> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
}

impl crate::backend::BackendStorage for WgpuStorage {
    type Device = WgpuDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    fn dtype(&self) -> DType {
        fail!()
    }

    fn device(&self) -> &Self::Device {
        fail!()
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    fn affine(&self, _: &Layout, _: f64, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    fn powf(&self, _: &Layout, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    fn elu(&self, _: &Layout, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    fn reduce_op(&self, _: ReduceOp, _: &Layout, _: &[usize]) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    fn to_dtype(&self, _: &Layout, _: DType) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    fn unary_impl<B: UnaryOpT>(&self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    fn binary_impl<B: BinaryOpT>(&self, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    fn where_cond(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    fn conv1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    fn conv_transpose1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    fn conv2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    fn index_select(&self, _: &Self, _: &Layout, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    fn scatter_add(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
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
        Err(Error::NotCompiledWithWgpuSupport)
    }

    fn matmul(
        &self,
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &Layout,
        _: &Layout,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &Layout) -> Result<()> {
        Err(Error::NotCompiledWithWgpuSupport)
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
        Err(Error::NotCompiledWithWgpuSupport)
    }

    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
}

impl WgpuDevice{
    pub (crate) async fn create(_: usize, _ : WgpuDeviceConfig) -> crate::Result<Self>{
        Err(Error::NotCompiledWithWgpuSupport)
    }

    pub (crate) async fn synchronize_async(&self) -> crate::Result<()> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
}

impl crate::backend::BackendDevice for WgpuDevice {
    type Storage = WgpuStorage;
    fn new(_: usize) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    fn set_seed(&self, _: u64) -> Result<()> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    fn location(&self) -> crate::DeviceLocation {
        fail!()
    }

    fn same_device(&self, _: &Self) -> bool {
        fail!()
    }

    fn zeros_impl(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    fn ones_impl(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    unsafe fn alloc_uninit(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    fn storage_from_slice<T: crate::WithDType>(&self, _: &[T]) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    fn storage_from_cpu_storage(&self, _: &CpuStorage) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    fn storage_from_cpu_storage_owned(&self, _: CpuStorage) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    fn rand_uniform(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    fn rand_normal(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithWgpuSupport)
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
