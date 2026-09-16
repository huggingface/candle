//! The Vulkan backend: storage + device trait implementations.
//!
//! Scaffold: `affine` (f32) and `reduce` (Sum, f32) are implemented through
//! the `candle_vulkan_kernels` compute shaders; every other operation returns
//! `Err(Error::Msg("vulkan: <op> not implemented (scaffold)"))`.

use std::sync::atomic::Ordering;

use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};

use candle_vulkan_kernels::{call_affine_f32, call_reduce_sum_f32};

pub use crate::vulkan_backend::device::{VulkanDevice, VulkanStorage};

use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Error, Layout, Result, Shape};

mod device;

/// Errors from the Vulkan backend.
#[derive(thiserror::Error, Debug)]
pub enum VulkanError {
    #[error("kernel error: {0}")]
    KernelError(#[from] candle_vulkan_kernels::VulkanKernelError),

    #[error("{0}")]
    Message(String),
}

impl From<String> for VulkanError {
    fn from(e: String) -> Self {
        VulkanError::Message(e)
    }
}

fn not_impl(op: &str) -> Error {
    Error::Msg(format!("vulkan: {op} not implemented (scaffold)"))
}

impl BackendStorage for VulkanStorage {
    type Device = VulkanDevice;

    fn try_clone(&self, _layout: &Layout) -> Result<Self> {
        Ok(self.clone())
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn const_set(&mut self, _: crate::scalar::Scalar, _: &Layout) -> Result<()> {
        Err(not_impl("const_set"))
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        self.device.synchronize()?;
        if self.dtype != DType::F32 {
            return Err(not_impl("to_cpu_storage (dtype)"));
        }
        let data = self.device.download_f32(&self.buffer)?;
        Ok(CpuStorage::F32(data))
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(not_impl("affine (dtype)"));
        }
        let size = layout.dim().map_err(|_| not_impl("affine (layout)"))?;
        let out = self.device.new_f32_buffer(size)?;
        let input = self.buffer.clone();
        let device = self.device.clone();
        let kernels = device.kernels();
        self.device
            .execute(move |cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>| {
                call_affine_f32(cbb, kernels, &input, &out, mul as f32, add as f32)
                    .map_err(|e| e.to_string())
            })?;
        Ok(Self::new(out, &self.device, size, DType::F32))
    }

    fn reduce_op(
        &self,
        ro: ReduceOp,
        layout: &Layout,
        axes: &[usize],
    ) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(not_impl("reduce_op (dtype)"));
        }
        let ReduceOp::Sum = ro else {
            return Err(not_impl("reduce_op (op)"));
        };
        let ndim = layout.ndim();
        let last_axis = ndim - 1;
        if axes.len() != 1 || axes[0] != last_axis {
            return Err(not_impl("reduce_op (axes)"));
        }
        let cols = layout
            .contiguous_last_dim()
            .map_err(|_| not_impl("reduce_op (layout)"))?;
        let total = layout
            .dim()
            .map_err(|_| not_impl("reduce_op (layout)"))?;
        let rows = total / cols;
        let out = self.device.new_f32_buffer(rows)?;
        let input = self.buffer.clone();
        let kernels = self.device.kernels();
        self.device.execute(move |
            cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        | {
            call_reduce_sum_f32(cbb, kernels, &input, &out, rows, cols)
                .map_err(|e| e.to_string())
        })?;
        Ok(Self::new(out, &self.device, rows, DType::F32))
    }

    fn powf(&self, _: &Layout, _: f64) -> Result<Self> {
        Err(not_impl("powf"))
    }

    fn elu(&self, _: &Layout, _: f64) -> Result<Self> {
        Err(not_impl("elu"))
    }

    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        Err(not_impl("cmp"))
    }

    fn to_dtype(&self, _: &Layout, _: DType) -> Result<Self> {
        Err(not_impl("to_dtype"))
    }

    fn unary_impl<B: UnaryOpT>(&self, _: &Layout) -> Result<Self> {
        Err(not_impl("unary_impl"))
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        _: &Self,
        _: &Layout,
        _: &Layout,
    ) -> Result<Self> {
        Err(not_impl("binary_impl"))
    }

    fn where_cond(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
    ) -> Result<Self> {
        Err(not_impl("where_cond"))
    }

    fn conv1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        Err(not_impl("conv1d"))
    }

    fn conv_transpose1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        Err(not_impl("conv_transpose1d"))
    }

    fn conv2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        Err(not_impl("conv2d"))
    }

    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        Err(not_impl("conv_transpose2d"))
    }

    fn index_select(&self, _: &Self, _: &Layout, _: &Layout, _: usize) -> Result<Self> {
        Err(not_impl("index_select"))
    }

    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self> {
        Err(not_impl("gather"))
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
        Err(not_impl("scatter_set"))
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
        Err(not_impl("scatter_add_set"))
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
        Err(not_impl("index_add"))
    }

    fn matmul(
        &self,
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &Layout,
        _: &Layout,
    ) -> Result<Self> {
        Err(not_impl("matmul"))
    }

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &Layout) -> Result<()> {
        Err(not_impl("copy_strided_src"))
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
        Err(not_impl("copy2d"))
    }

    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Err(not_impl("avg_pool2d"))
    }

    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Err(not_impl("max_pool2d"))
    }

    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        Err(not_impl("upsample_nearest1d"))
    }

    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
        Err(not_impl("upsample_nearest2d"))
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
        Err(not_impl("upsample_bilinear2d"))
    }
}

impl BackendDevice for VulkanDevice {
    type Storage = VulkanStorage;

    fn new(gpu_id: usize) -> Result<Self> {
        Self::new(gpu_id)
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        self.seed_atomic().store(seed, Ordering::Relaxed);
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        Ok(self.seed_atomic().load(Ordering::Relaxed))
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Vulkan {
            gpu_id: self.gpu_id(),
        }
    }

    fn same_device(&self, other: &Self) -> bool {
        self.gpu_id() == other.gpu_id()
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let dim = shape.elem_count()?;
        let buffer = self.new_f32_buffer(dim)?;
        Ok(VulkanStorage::new(buffer, self, dim, dtype))
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let dim = shape.elem_count()?;
        let buffer = self.new_f32_buffer(dim)?;
        Ok(VulkanStorage::new(buffer, self, dim, dtype))
    }

    fn storage_from_slice<T: crate::WithDType>(&self, slice: &[T]) -> Result<Self::Storage> {
        let dtype = T::dtype();
        let dim = slice.len();
        if dtype != DType::F32 {
            return Err(not_impl("storage_from_slice (dtype)"));
        }
        let data: Vec<f32> =
            slice
                .iter()
                .map(|t| t.to_dtype::<f32>().expect("f32 dtype"))
                .collect();
        let buffer = self.upload_f32(&data)?;
        Ok(VulkanStorage::new(buffer, self, dim, dtype))
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        let (data, dtype) = match storage {
            CpuStorage::F32(d) => (Some(d.clone()), DType::F32),
            _ => (None, storage.dtype()),
        };
        let Some(data) = data else {
            return Err(not_impl("storage_from_cpu_storage (dtype)"));
        };
        let buffer = self.upload_f32(&data)?;
        Ok(VulkanStorage::new(buffer, self, data.len(), dtype))
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<Self::Storage> {
        self.storage_from_cpu_storage(&storage)
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, low: f64, upper: f64) -> Result<Self::Storage> {
        use rand::Rng;
        let dim = shape.elem_count()?;
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..dim)
            .map(|_| rng.gen_range(low as f32..upper as f32))
            .collect();
        let buffer = self.upload_f32(&data)?;
        Ok(VulkanStorage::new(buffer, self, dim, dtype))
    }

    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<Self::Storage> {
        use rand::Rng;
        use rand_distr::{Distribution, Normal};
        let dim = shape.elem_count()?;
        let mut rng = rand::thread_rng();
        let normal = Normal::new(mean as f64, std as f64).expect("std > 0");
        let data: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rng) as f32).collect();
        let buffer = self.upload_f32(&data)?;
        Ok(VulkanStorage::new(buffer, self, dim, dtype))
    }

    fn synchronize(&self) -> Result<()> {
        self.synchronize()
    }
}
