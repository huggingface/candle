//! Implementation of the Vulkan backend when Vulkan support has not been compiled in.
//!
#![allow(dead_code)]
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Error, Layout, Result, Shape};

#[derive(Debug, Clone)]
pub struct VulkanDevice;

#[derive(Debug)]
pub struct VulkanStorage;

/// Placeholder for the Vulkan backend's pipelined id-readback handle
/// (see the real implementation when the `vulkan` feature is enabled).
pub struct PipedId;

impl PipedId {
    pub fn wait(self) -> Result<u32> {
        Err(Error::NotCompiledWithVulkanSupport)
    }
}

impl VulkanStorage {
    pub fn transfer_to_device(&self, _dst: &VulkanDevice) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fused_decode_attention(
        &self,
        _l: &Layout,
        _k: &VulkanStorage,
        _k_l: &Layout,
        _v: &VulkanStorage,
        _v_l: &Layout,
        _scale: f32,
        _num_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _kv_len: usize,
        _kv_stride: usize,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    pub fn fused_add_rms_norm(
        &self,
        _l: &Layout,
        _h: &VulkanStorage,
        _h_l: &Layout,
        _alpha: &VulkanStorage,
        _alpha_l: &Layout,
        _eps: f64,
    ) -> Result<(Self, Self)> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    pub fn fused_post_norm_add(
        &self,
        _l: &Layout,
        _h: &VulkanStorage,
        _h_l: &Layout,
        _alpha: &VulkanStorage,
        _alpha_l: &Layout,
        _eps: f64,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn kv_cache_append_rows(
        &self,
        _l: &Layout,
        _h: &VulkanStorage,
        _h_l: &Layout,
        _dst_k: &VulkanStorage,
        _dst_k_l: &Layout,
        _dst_v: &VulkanStorage,
        _dst_v_l: &Layout,
        _row: usize,
    ) -> Result<()> {
        Err(Error::NotCompiledWithVulkanSupport)
    }
}

#[derive(thiserror::Error, Debug)]
pub enum VulkanError {
    #[error("{0}")]
    Message(String),
}

impl From<String> for VulkanError {
    fn from(e: String) -> Self {
        VulkanError::Message(e)
    }
}

macro_rules! fail {
    () => {
        unimplemented!("vulkan support has not been enabled, add `vulkan` feature to enable.")
    };
}

impl crate::backend::BackendStorage for VulkanStorage {
    type Device = VulkanDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn dtype(&self) -> DType {
        fail!()
    }

    fn device(&self) -> &Self::Device {
        fail!()
    }

    fn const_set(&mut self, _: crate::scalar::Scalar, _: &Layout) -> Result<()> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn affine(&self, _: &Layout, _: f64, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn powf(&self, _: &Layout, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn elu(&self, _: &Layout, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn reduce_op(&self, _: ReduceOp, _: &Layout, _: &[usize]) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn to_dtype(&self, _: &Layout, _: DType) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn unary_impl<B: UnaryOpT>(&self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn binary_impl<B: BinaryOpT>(&self, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn where_cond(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn conv1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn conv_transpose1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn conv2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn index_select(&self, _: &Self, _: &Layout, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }
    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
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
        Err(Error::NotCompiledWithVulkanSupport)
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
        Err(Error::NotCompiledWithVulkanSupport)
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
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn matmul(
        &self,
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &Layout,
        _: &Layout,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &Layout) -> Result<()> {
        Err(Error::NotCompiledWithVulkanSupport)
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
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn upsample_bilinear2d(
        &self,
        _: &Layout,
        _: usize,
        _: usize,
        _: bool,
        _: Option<f64>,
        _: Option<f64>,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }
}

impl VulkanDevice {
    /// Records a 4-byte id readback and submits the batch without waiting
    /// (Vulkan only; the dummy backend has no queue).
    pub fn submit_id_readback(&self, _id: &VulkanStorage) -> Result<PipedId> {
        Err(Error::NotCompiledWithVulkanSupport)
    }
}

impl crate::backend::BackendDevice for VulkanDevice {
    type Storage = VulkanStorage;
    fn new(_: usize) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn set_seed(&self, _: u64) -> Result<()> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn get_current_seed(&self) -> Result<u64> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn location(&self) -> crate::DeviceLocation {
        fail!()
    }

    fn same_device(&self, _: &Self) -> bool {
        fail!()
    }

    fn zeros_impl(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    unsafe fn alloc_uninit(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn storage_from_slice<T: crate::WithDType>(&self, _: &[T]) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn storage_from_cpu_storage(&self, _: &CpuStorage) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn storage_from_cpu_storage_owned(&self, _: CpuStorage) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn rand_uniform(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn rand_normal(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}
