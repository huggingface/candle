//! The Vulkan backend: storage + device trait implementations.
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo};

use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::sync::GpuFuture;

pub use crate::vulkan_backend::device::{VBuf, VulkanDevice};

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

impl BackendDevice for VulkanDevice {
    type Storage = VulkanStorage;

    fn new(gpu_id: usize) -> Result<Self> {
        Self::new(gpu_id)
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
        VulkanStorage::new(self, shape.elem_count(), dtype)
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        self.zeros_impl(shape, dtype)
    }

    fn storage_from_slice<T: crate::WithDType>(&self, _: &[T]) -> Result<Self::Storage> {
        todo!()
    }

    fn storage_from_cpu_storage(&self, _: &CpuStorage) -> Result<Self::Storage> {
        todo!()
    }

    fn storage_from_cpu_storage_owned(&self, _: CpuStorage) -> Result<Self::Storage> {
        todo!()
    }

    fn rand_uniform(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        todo!()
    }

    fn rand_normal(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        todo!()
    }

    fn set_seed(&self, _: u64) -> Result<()> {
        todo!()
    }

    fn get_current_seed(&self) -> Result<u64> {
        todo!()
    }

    fn synchronize(&self) -> Result<()> {
        todo!()
    }
}

#[derive(Debug)]
pub enum VulkanStorageBuffer {
    U8(Subbuffer<[u8]>),
    U32(Subbuffer<[u32]>),
    I16(Subbuffer<[i16]>),
    I32(Subbuffer<[i32]>),
    I64(Subbuffer<[i64]>),
    BF16(Subbuffer<[half::bf16]>),
    F16(Subbuffer<[half::f16]>),
    F32(Subbuffer<[f32]>),
    F64(Subbuffer<[f64]>),
    F8E4M3(Subbuffer<[float8::F8E4M3]>),
    // Dummy types that store raw bytes
    F6E2M3(Subbuffer<[u8]>),
    F6E3M2(Subbuffer<[u8]>),
    F4(Subbuffer<[u8]>),
    F8E8M0(Subbuffer<[u8]>),
}

#[derive(Debug)]
pub struct VulkanStorage {
    buffer: VulkanStorageBuffer,
}

impl VulkanStorage {
    pub fn new(device: &VulkanDevice, size: usize, dtype: DType) -> Result<Self> {
        let slice = match dtype {
            DType::U8 => VulkanStorageBuffer::U8(Self::create_vram_buffer::<u8>(device, size)?),
            DType::U32 => VulkanStorageBuffer::U32(Self::create_vram_buffer::<u32>(device, size)?),
            DType::I16 => VulkanStorageBuffer::I16(Self::create_vram_buffer::<i16>(device, size)?),
            DType::I32 => VulkanStorageBuffer::I32(Self::create_vram_buffer::<i32>(device, size)?),
            DType::I64 => VulkanStorageBuffer::I64(Self::create_vram_buffer::<i64>(device, size)?),
            DType::BF16 => {
                VulkanStorageBuffer::BF16(Self::create_vram_buffer::<half::bf16>(device, size)?)
            }
            DType::F16 => {
                VulkanStorageBuffer::F16(Self::create_vram_buffer::<half::f16>(device, size)?)
            }
            DType::F32 => VulkanStorageBuffer::F32(Self::create_vram_buffer::<f32>(device, size)?),
            DType::F64 => VulkanStorageBuffer::F64(Self::create_vram_buffer::<f64>(device, size)?),
            DType::F8E4M3 => VulkanStorageBuffer::F8E4M3(
                Self::create_vram_buffer::<float8::F8E4M3>(device, size)?,
            ),
            DType::F6E2M3 => VulkanStorageBuffer::U8(Self::create_vram_buffer::<u8>(device, size)?),
            DType::F6E3M2 => VulkanStorageBuffer::U8(Self::create_vram_buffer::<u8>(device, size)?),
            DType::F4 => VulkanStorageBuffer::U8(Self::create_vram_buffer::<u8>(device, size)?),
            DType::F8E8M0 => VulkanStorageBuffer::U8(Self::create_vram_buffer::<u8>(device, size)?),
        };
        Ok(Self { buffer: slice })
    }

    pub fn buffer(&self) -> &VulkanStorageBuffer {
        &self.buffer
    }

    fn create_vram_buffer<T: Default + Clone>(
        device: &VulkanDevice,
        size: usize,
    ) -> Result<Subbuffer<[T]>>
    where
        T: BufferContents,
    {
        let data = vec![T::default(); size];
        // Create the staging buffer on the host with the data that needs to be copied to VRAM.
        let source_buffer = Buffer::from_iter(
            device.mem_alloc().clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            data,
        )
        .map_err(|error| {
            Error::Vulkan(format!("Unable to create source buffer: {error:?}").into())
        })?;

        // Create the VRAM buffer.
        let vram_buffer = Buffer::new_slice::<T>(
            device.mem_alloc().clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST
                    | BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            size as u64,
        )
        .map_err(|error| {
            Error::Vulkan(format!("Unable to create vram buffer: {error:?}").into())
        })?;

        // Build the copy command
        let mut builder = AutoCommandBufferBuilder::primary(
            device.cbb_alloc().clone(),
            device.queue().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(|error| {
            Error::Vulkan(format!("Unable to create command buffer builder: {error:?}").into())
        })?;
        builder
            .copy_buffer(CopyBufferInfo::buffers(
                source_buffer.clone(),
                vram_buffer.clone(),
            ))
            .map_err(|error| {
                Error::Vulkan(format!("Unable to set copy_buffer command: {error:?}").into())
            })?;
        let command_buffer = builder.build().map_err(|error| {
            Error::Vulkan(format!("Unable to create command buffer: {error:?}").into())
        })?;

        // Execute and flush the command buffer
        let future = vulkano::sync::now(device.device().clone())
            .then_execute(device.queue().clone(), command_buffer)
            .map_err(|error| {
                Error::Vulkan(format!("Unable to execute command buffer: {error:?}").into())
            })?
            .then_signal_fence_and_flush()
            .map_err(|error| {
                Error::Vulkan(
                    format!("Unable to signal fence and flush command buffer: {error:?}").into(),
                )
            })?;

        // Block the CPU thread until the GPU finishes copying the data
        future.wait(None).map_err(|error| {
            Error::Vulkan(
                format!("Failure occurred waiting for data to be uploaded: {error:?}").into(),
            )
        })?;
        Ok(vram_buffer)
    }
}

impl BackendStorage for VulkanStorage {
    type Device = VulkanDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        todo!()
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
        _params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        todo!()
    }

    fn conv_transpose1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        todo!()
    }

    fn conv2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        todo!()
    }

    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
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

    fn upsample_bilinear2d(
        &self,
        _: &Layout,
        _: usize,
        _: usize,
        _: bool,
        _: Option<f64>,
        _: Option<f64>,
    ) -> Result<Self> {
        todo!()
    }

    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self> {
        todo!()
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
        todo!()
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

    fn copy2d(
        &self,
        _: &mut Self,
        _d1: usize,
        _d2: usize,
        _src_stride1: usize,
        _dst_stride1: usize,
        _src_offset: usize,
        _dst_offset: usize,
    ) -> Result<()> {
        todo!()
    }

    fn const_set(&mut self, _: crate::scalar::Scalar, _: &Layout) -> Result<()> {
        todo!()
    }
}
