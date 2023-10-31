use crate::backend::{BackendDevice, BackendStorage};
use crate::conv::{ParamsConv1D, ParamsConv2D, ParamsConvTranspose2D};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape};
pub use candle_metal;
use metal;

/// Metal related errors
#[derive(thiserror::Error, Debug)]
pub enum MetalError {
    #[error("metal error")]
    Metal,
}

#[derive(Clone)]
pub struct MetalDevice {
    device: metal::Device,
}

impl std::fmt::Debug for MetalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetalDevice({:?})", self.device.registry_id())
    }
}

impl std::ops::Deref for MetalDevice {
    type Target = metal::DeviceRef;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl MetalDevice {
    pub fn metal_device(&self) -> &metal::DeviceRef {
        self.device.as_ref()
    }

    pub fn id(&self) -> u64 {
        self.registry_id()
    }
}

#[derive(Debug, Clone)]
pub struct MetalStorage {
    pub buffer: metal::Buffer,
    pub device: metal::Device,
}

impl BackendStorage for MetalStorage {
    type Device = MetalDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        Ok(self.clone())
    }

    fn dtype(&self) -> DType {
        todo!()
    }

    fn device(&self) -> &Self::Device {
        todo!()
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        todo!()
    }

    fn affine(&self, _: &Layout, _: f64, _: f64) -> Result<Self> {
        todo!()
    }

    fn powf(&self, _: &Layout, _: f64) -> Result<Self> {
        todo!()
    }

    fn elu(&self, _: &Layout, _: f64) -> Result<Self> {
        todo!()
    }

    fn reduce_op(&self, _: ReduceOp, _: &Layout, _: &[usize]) -> Result<Self> {
        todo!()
    }

    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        todo!()
    }

    fn to_dtype(&self, _: &Layout, _: DType) -> Result<Self> {
        todo!()
    }

    fn unary_impl<B: UnaryOpT>(&self, _: &Layout) -> Result<Self> {
        todo!()
    }

    fn binary_impl<B: BinaryOpT>(&self, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        todo!()
    }

    fn where_cond(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
        todo!()
    }

    fn conv1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &ParamsConv1D,
    ) -> Result<Self> {
        todo!()
    }

    fn conv2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &ParamsConv2D,
    ) -> Result<Self> {
        todo!()
    }

    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &ParamsConvTranspose2D,
    ) -> Result<Self> {
        todo!()
    }

    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        todo!()
    }

    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        todo!()
    }

    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        todo!()
    }

    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
        todo!()
    }

    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self> {
        todo!()
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
        todo!()
    }

    fn index_select(&self, _: &Self, _: &Layout, _: &Layout, _: usize) -> Result<Self> {
        todo!()
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
        todo!()
    }

    fn matmul(
        &self,
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &Layout,
        _: &Layout,
    ) -> Result<Self> {
        todo!()
    }

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &Layout) -> Result<()> {
        todo!()
    }
}

impl BackendDevice for MetalDevice {
    type Storage = MetalStorage;

    fn new(_ordinal: usize) -> Result<Self> {
        todo!()
    }

    fn set_seed(&self, _seed: u64) -> Result<()> {
        todo!()
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Metal
    }

    fn same_device(&self, _rhs: &Self) -> bool {
        todo!()
    }

    fn zeros_impl(&self, _shape: &Shape, _dtype: DType) -> Result<MetalStorage> {
        todo!()
    }

    fn ones_impl(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
        todo!()
    }

    fn storage_from_cpu_storage(&self, _: &CpuStorage) -> Result<Self::Storage> {
        todo!()
    }

    fn rand_uniform(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        todo!()
    }

    fn rand_normal(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        todo!()
    }
}
