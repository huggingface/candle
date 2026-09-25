//! The Vulkan backend: storage + device trait implementations.
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryAutoCommandBuffer,
};

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
/// Uploads `bytes` (`n` elements of `dtype`) to a new device storage via a
/// host-visible staging buffer and a deferred device copy.
fn upload_bytes(device: &VulkanDevice, bytes: &[u8], n: usize, dtype: DType) -> Result<VulkanStorage> {
    let staging = Buffer::from_iter(
        device.mem_alloc(),
        &BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        &AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        bytes.iter().copied(),
    )
    .map_err(|e| Error::Vulkan(format!("upload staging: {e:?}").into()))?;
    let storage = VulkanStorage::new(device, n, dtype)?;
    let dst = storage.buffer().as_u8();
    device.execute(move |cbb| {
        cbb.copy_buffer(CopyBufferInfo::new(staging, dst))
            .map_err(|e| e.to_string())?;
        Ok(())
    })?;
    Ok(storage)
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

    fn storage_from_slice<T: crate::WithDType>(&self, data: &[T]) -> Result<Self::Storage> {
        let n = data.len();
        // Upload through a byte view so every element type works with one
        // code path.
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, n * std::mem::size_of::<T>())
        };
        upload_bytes(self, bytes, n, T::DTYPE)
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        match storage {
            CpuStorage::U8(d) => self.storage_from_slice(d),
            CpuStorage::U32(d) => self.storage_from_slice(d),
            CpuStorage::I16(d) => self.storage_from_slice(d),
            CpuStorage::I32(d) => self.storage_from_slice(d),
            CpuStorage::I64(d) => self.storage_from_slice(d),
            CpuStorage::BF16(d) => self.storage_from_slice(d),
            CpuStorage::F16(d) => self.storage_from_slice(d),
            CpuStorage::F32(d) => self.storage_from_slice(d),
            CpuStorage::F64(d) => self.storage_from_slice(d),
            CpuStorage::F8E4M3(d) => self.storage_from_slice(d),
            // Byte-width dummy types have no `WithDType` impl: upload the
            // raw bytes and tag the storage with the target dtype.
            CpuStorage::F6E2M3(d) => {
                let bytes: Vec<u8> = d.iter().map(|v| v.to_bits()).collect();
                upload_bytes(self, &bytes, d.len(), DType::F6E2M3)
            }
            CpuStorage::F6E3M2(d) => {
                let bytes: Vec<u8> = d.iter().map(|v| v.to_bits()).collect();
                upload_bytes(self, &bytes, d.len(), DType::F6E3M2)
            }
            CpuStorage::F4(d) => {
                let bytes: Vec<u8> = d.iter().map(|v| v.to_bits()).collect();
                upload_bytes(self, &bytes, d.len(), DType::F4)
            }
            CpuStorage::F8E8M0(d) => {
                let bytes: Vec<u8> = d.iter().map(|v| v.to_bits()).collect();
                upload_bytes(self, &bytes, d.len(), DType::F8E8M0)
            }
        }
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<Self::Storage> {
        match storage {
            CpuStorage::U8(d) => self.storage_from_slice(&d),
            CpuStorage::U32(d) => self.storage_from_slice(&d),
            CpuStorage::I16(d) => self.storage_from_slice(&d),
            CpuStorage::I32(d) => self.storage_from_slice(&d),
            CpuStorage::I64(d) => self.storage_from_slice(&d),
            CpuStorage::BF16(d) => self.storage_from_slice(&d),
            CpuStorage::F16(d) => self.storage_from_slice(&d),
            CpuStorage::F32(d) => self.storage_from_slice(&d),
            CpuStorage::F64(d) => self.storage_from_slice(&d),
            CpuStorage::F8E4M3(d) => self.storage_from_slice(&d),
            CpuStorage::F6E2M3(d) => {
                let bytes: Vec<u8> = d.into_iter().map(|v| v.to_bits()).collect();
                upload_bytes(self, &bytes, bytes.len(), DType::F6E2M3)
            }
            CpuStorage::F6E3M2(d) => {
                let bytes: Vec<u8> = d.into_iter().map(|v| v.to_bits()).collect();
                upload_bytes(self, &bytes, bytes.len(), DType::F6E3M2)
            }
            CpuStorage::F4(d) => {
                let bytes: Vec<u8> = d.into_iter().map(|v| v.to_bits()).collect();
                upload_bytes(self, &bytes, bytes.len(), DType::F4)
            }
            CpuStorage::F8E8M0(d) => {
                let bytes: Vec<u8> = d.into_iter().map(|v| v.to_bits()).collect();
                upload_bytes(self, &bytes, bytes.len(), DType::F8E8M0)
            }
        }
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, min: f64, max: f64) -> Result<Self::Storage> {
        use rand::prelude::*;
        let elem_count = shape.elem_count();
        // Host-side generation seeded from the device seed, then uploaded:
        // keeps the distribution semantics of the CPU backend while the
        // device stays free of RNG plumbing.
        let seed = self.seed_atomic().load(std::sync::atomic::Ordering::Relaxed);
        let mut rng = StdRng::seed_from_u64(seed);
        let cpu = match dtype {
            DType::U8
            | DType::U32
            | DType::I16
            | DType::I32
            | DType::I64
            | DType::F6E2M3
            | DType::F6E3M2
            | DType::F4
            | DType::F8E8M0 => {
                return Err(Error::UnsupportedDTypeForOp(dtype, "rand_uniform").bt())
            }
            DType::BF16 => {
                let uniform = rand::distr::Uniform::new(
                    half::bf16::from_f64(min),
                    half::bf16::from_f64(max),
                )
                .map_err(Error::wrap)?;
                let data: Vec<half::bf16> =
                    (0..elem_count).map(|_| rng.sample(uniform.clone())).collect();
                CpuStorage::BF16(data)
            }
            DType::F16 => {
                let uniform = rand::distr::Uniform::new(
                    half::f16::from_f64(min),
                    half::f16::from_f64(max),
                )
                .map_err(Error::wrap)?;
                let data: Vec<half::f16> =
                    (0..elem_count).map(|_| rng.sample(uniform.clone())).collect();
                CpuStorage::F16(data)
            }
            DType::F32 => {
                let uniform =
                    rand::distr::Uniform::new(min as f32, max as f32).map_err(Error::wrap)?;
                let data: Vec<f32> =
                    (0..elem_count).map(|_| rng.sample(uniform.clone())).collect();
                CpuStorage::F32(data)
            }
            DType::F64 => {
                let uniform =
                    rand::distr::Uniform::new(min, max).map_err(Error::wrap)?;
                let data: Vec<f64> =
                    (0..elem_count).map(|_| rng.sample(uniform.clone())).collect();
                CpuStorage::F64(data)
            }
            DType::F8E4M3 => {
                let uniform = rand::distr::Uniform::new(
                    microfloat::f8e4m3::from_f64(min),
                    microfloat::f8e4m3::from_f64(max),
                )
                .map_err(Error::wrap)?;
                let data: Vec<microfloat::f8e4m3> =
                    (0..elem_count).map(|_| rng.sample(uniform.clone())).collect();
                CpuStorage::F8E4M3(data)
            }
        };
        self.storage_from_cpu_storage_owned(cpu)
    }

    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<Self::Storage> {
        use rand::prelude::*;
        let elem_count = shape.elem_count();
        // Host-side generation seeded from the device seed, then uploaded.
        let seed = self.seed_atomic().load(std::sync::atomic::Ordering::Relaxed);
        let mut rng = StdRng::seed_from_u64(seed);
        let cpu = match dtype {
            DType::U8
            | DType::U32
            | DType::I16
            | DType::I32
            | DType::I64
            | DType::F6E2M3
            | DType::F6E3M2
            | DType::F4
            | DType::F8E8M0 => {
                return Err(Error::UnsupportedDTypeForOp(dtype, "rand_normal").bt())
            }
            DType::BF16 => {
                let normal = rand_distr::Normal::new(
                    half::bf16::from_f64(mean),
                    half::bf16::from_f64(std),
                )
                .map_err(Error::wrap)?;
                let data: Vec<half::bf16> = (0..elem_count).map(|_| normal.sample(&mut rng)).collect();
                CpuStorage::BF16(data)
            }
            DType::F16 => {
                let normal = rand_distr::Normal::new(
                    half::f16::from_f64(mean),
                    half::f16::from_f64(std),
                )
                .map_err(Error::wrap)?;
                let data: Vec<half::f16> = (0..elem_count).map(|_| normal.sample(&mut rng)).collect();
                CpuStorage::F16(data)
            }
            DType::F32 => {
                let normal = rand_distr::Normal::new(mean as f32, std as f32).map_err(Error::wrap)?;
                let data: Vec<f32> = (0..elem_count).map(|_| normal.sample(&mut rng)).collect();
                CpuStorage::F32(data)
            }
            DType::F64 => {
                let normal = rand_distr::Normal::new(mean, std).map_err(Error::wrap)?;
                let data: Vec<f64> = (0..elem_count).map(|_| normal.sample(&mut rng)).collect();
                CpuStorage::F64(data)
            }
            DType::F8E4M3 => {
                // f8e4m3 has no Float impl: draw in f64 and convert.
                let normal = rand_distr::Normal::new(mean, std).map_err(Error::wrap)?;
                let data: Vec<microfloat::f8e4m3> = (0..elem_count)
                    .map(|_| microfloat::f8e4m3::from_f64(normal.sample(&mut rng)))
                    .collect();
                CpuStorage::F8E4M3(data)
            }
        };
        self.storage_from_cpu_storage_owned(cpu)
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        self.seed_atomic().store(seed, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        Ok(self.seed_atomic().load(std::sync::atomic::Ordering::Relaxed))
    }

    fn synchronize(&self) -> Result<()> {
        self.synchronize()
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
    F8E4M3(Subbuffer<[microfloat::f8e4m3]>),
    // Dummy types that store raw bytes
    F6E2M3(Subbuffer<[microfloat::f6e2m3fn]>),
    F6E3M2(Subbuffer<[microfloat::f6e3m2fn]>),
    F4(Subbuffer<[microfloat::f4e2m1fn]>),
    F8E8M0(Subbuffer<[microfloat::f8e8m0fnu]>),
}

impl VulkanStorageBuffer {
    /// The underlying device buffer (regardless of element type).
    pub fn buffer(&self) -> std::sync::Arc<vulkano::buffer::Buffer> {
        match self {
            Self::U8(b) => b.buffer().clone(),
            Self::U32(b) => b.buffer().clone(),
            Self::I16(b) => b.buffer().clone(),
            Self::I32(b) => b.buffer().clone(),
            Self::I64(b) => b.buffer().clone(),
            Self::BF16(b) => b.buffer().clone(),
            Self::F16(b) => b.buffer().clone(),
            Self::F32(b) => b.buffer().clone(),
            Self::F64(b) => b.buffer().clone(),
            Self::F8E4M3(b) => b.buffer().clone(),
            Self::F6E2M3(b) => b.buffer().clone(),
            Self::F6E3M2(b) => b.buffer().clone(),
            Self::F4(b) => b.buffer().clone(),
            Self::F8E8M0(b) => b.buffer().clone(),
        }
    }
    /// A byte view over the whole device buffer.
    pub fn as_u8(&self) -> Subbuffer<[u8]> {
        match self {
            Self::U8(b) => b.clone(),
            other => Subbuffer::<[u8]>::new(other.buffer().clone()),
        }
    }
}
#[derive(Debug)]
pub struct VulkanStorage {
    buffer: VulkanStorageBuffer,
    device: VulkanDevice,
    dtype: DType,
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
            DType::F8E4M3 => VulkanStorageBuffer::F8E4M3(Self::create_vram_buffer::<
                microfloat::f8e4m3,
            >(device, size)?),
            DType::F6E2M3 => VulkanStorageBuffer::F6E2M3(Self::create_vram_buffer::<
                microfloat::f6e2m3fn,
            >(device, size)?),
            DType::F6E3M2 => VulkanStorageBuffer::F6E3M2(Self::create_vram_buffer::<
                microfloat::f6e3m2fn,
            >(device, size)?),
            DType::F4 => VulkanStorageBuffer::F4(Self::create_vram_buffer::<microfloat::f4e2m1fn>(
                device, size,
            )?),
            DType::F8E8M0 => VulkanStorageBuffer::F8E8M0(Self::create_vram_buffer::<
                microfloat::f8e8m0fnu,
            >(device, size)?),
        };
        Ok(Self {
            buffer: slice,
            device: device.clone(),
            dtype,
        })
    }

    pub fn buffer(&self) -> &VulkanStorageBuffer {
        &self.buffer
    }

    fn create_vram_buffer<T>(device: &VulkanDevice, size: usize) -> Result<Subbuffer<[T]>>
    where
        T: BufferContents + Default + Clone,
    {
        let data = vec![T::default(); size];
        // Create the staging buffer on the host with the data that needs to be copied to VRAM.
        let source_buffer = Buffer::from_iter(
            device.mem_alloc(),
            &BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            &AllocationCreateInfo {
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
            device.mem_alloc(),
            &BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST
                    | BufferUsage::UNIFORM_BUFFER
                    | BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
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
            .copy_buffer(CopyBufferInfo::new(
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
    /// Copies a device-local (VRAM) buffer to a host-visible buffer and
    /// returns its contents as an owned `Vec<T>`.
    fn copy_to_host<T>(device: &VulkanDevice, vram: &Subbuffer<[T]>) -> Result<Vec<T>>
    where
        T: BufferContents + Default + Clone,
    {
        let len = vram.len();
        let destination_buffer = Buffer::new_slice::<T>(
            device.mem_alloc(),
            &BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            len,
        )
        .map_err(|error| {
            Error::Vulkan(format!("Unable to create host readback buffer: {error:?}").into())
        })?;
        let mut builder = AutoCommandBufferBuilder::primary(
            device.cbb_alloc().clone(),
            device.queue().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(|error| {
            Error::Vulkan(format!("Unable to create command buffer builder: {error:?}").into())
        })?;
        builder
            .copy_buffer(CopyBufferInfo::new(
                vram.clone(),
                destination_buffer.clone(),
            ))
            .map_err(|error| {
                Error::Vulkan(format!("Unable to set copy_buffer command: {error:?}").into())
            })?;
        let command_buffer = builder.build().map_err(|error| {
            Error::Vulkan(format!("Unable to create command buffer: {error:?}").into())
        })?;
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
        future.wait(None).map_err(|error| {
            Error::Vulkan(
                format!("Failure occurred waiting for readback to finish: {error:?}").into(),
            )
        })?;
        let data = destination_buffer.read().map_err(|error| {
            Error::Vulkan(
                format!("Unable to acquire read lock on readback buffer: {error:?}").into(),
            )
        })?;
        Ok(data.to_vec())
    }
    /// Runs the `test_fill_f16` slang kernel over this storage: element `i`
    /// is filled with the bits of `i` reinterpreted as an f16 value, so a
    /// 65,536-element storage ends up holding every possible f16 value.
    /// The storage must be F16.
    pub fn fill_f16(&self) -> Result<()> {
        let buffer = match &self.buffer {
            VulkanStorageBuffer::F16(buffer) => buffer,
            _ => {
                return Err(Error::Vulkan(
                    "fill_f16 requires F16 storage".to_string().into(),
                ))
            }
        };
        let device = self.device.clone();
        let mut builder = AutoCommandBufferBuilder::primary(
            device.cbb_alloc().clone(),
            device.queue().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(|error| {
            Error::Vulkan(format!("Unable to create command buffer builder: {error:?}").into())
        })?;
        candle_vulkan_kernels::call_test_fill_f16(
            &mut builder,
            device.kernels().as_ref(),
            buffer,
            buffer.len() as usize,
        )
        .map_err(VulkanError::KernelError)?;
        let command_buffer = builder.build().map_err(|error| {
            Error::Vulkan(format!("Unable to create command buffer: {error:?}").into())
        })?;
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
        future.wait(None).map_err(|error| {
            Error::Vulkan(
                format!("Failure occurred waiting for fill_f16 to finish: {error:?}").into(),
            )
        })?;
        Ok(())
    }
    /// Shared machinery for the `fill_*` test methods: build a one-time compute
    /// command buffer, run `dispatch` inside it, then execute and wait.
    fn run_fill(
        &self,
        label: &str,
        dispatch: impl FnOnce(
            &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        )
            -> std::result::Result<(), candle_vulkan_kernels::VulkanKernelError>,
    ) -> Result<()> {
        let device = self.device.clone();
        let mut builder = AutoCommandBufferBuilder::primary(
            device.cbb_alloc().clone(),
            device.queue().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(|error| {
            Error::Vulkan(format!("Unable to create command buffer builder: {error:?}").into())
        })?;
        dispatch(&mut builder).map_err(VulkanError::KernelError)?;
        let command_buffer = builder.build().map_err(|error| {
            Error::Vulkan(format!("Unable to create command buffer: {error:?}").into())
        })?;
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
        future.wait(None).map_err(|error| {
            Error::Vulkan(
                format!("Failure occurred waiting for {label} to finish: {error:?}").into(),
            )
        })?;
        Ok(())
    }
    /// Fills this u8 storage with all 256 possible byte values: element `i`
    /// holds `i`. The storage must be U8.
    pub fn fill_u8(&self) -> Result<()> {
        let buffer = match &self.buffer {
            VulkanStorageBuffer::U8(buffer) => buffer,
            _ => {
                return Err(Error::Vulkan(
                    "fill_u8 requires U8 storage".to_string().into(),
                ))
            }
        };
        let kernels = self.device.kernels().clone();
        let total = buffer.len() as usize;
        self.run_fill("fill_u8", move |cbb| {
            candle_vulkan_kernels::call_test_fill_u8(cbb, kernels.as_ref(), buffer, total)
        })
    }
    /// Fills this i16 storage with all 65,536 possible 16-bit bit patterns:
    /// element `i` holds the bit pattern of `i`. The storage must be I16.
    pub fn fill_i16(&self) -> Result<()> {
        let buffer = match &self.buffer {
            VulkanStorageBuffer::I16(buffer) => buffer,
            _ => {
                return Err(Error::Vulkan(
                    "fill_i16 requires I16 storage".to_string().into(),
                ))
            }
        };
        let kernels = self.device.kernels().clone();
        let total = buffer.len() as usize;
        self.run_fill("fill_i16", move |cbb| {
            candle_vulkan_kernels::call_test_fill_i16(cbb, kernels.as_ref(), buffer, total)
        })
    }
    /// Fills this bf16 storage with all 65,536 possible 16-bit bit patterns:
    /// element `i` holds the bit pattern of `i`. The storage must be BF16.
    ///
    /// Uses the emulated (raw 16-bit) fill path. vulkano 0.35.2 does not
    /// expose `VK_KHR_shader_bfloat16`, so native bfloat16 compute cannot be
    /// enabled on the device; the raw 16-bit store path round-trips the exact
    /// same BF16 bit patterns and is the path that is verifiable today. The
    /// native shader (`fill_native_bf16.slang`) and
    /// `call_test_fill_bf16_native` are kept for a vulkano release that can
    /// enable bfloat16.
    pub fn fill_bf16(&self) -> Result<()> {
        let buffer = match &self.buffer {
            VulkanStorageBuffer::BF16(buffer) => buffer,
            _ => {
                return Err(Error::Vulkan(
                    "fill_bf16 requires BF16 storage".to_string().into(),
                ))
            }
        };
        let kernels = self.device.kernels().clone();
        let total = buffer.len() as usize;
        self.run_fill("fill_bf16", move |cbb| {
            candle_vulkan_kernels::call_test_fill_bf16_emulated(
                cbb,
                kernels.as_ref(),
                buffer,
                total,
            )
        })
    }
    /// Fills this bf16 storage with all 65,536 possible 16-bit bit patterns
    /// using the emulated (raw 16-bit) path, regardless of native support.
    /// The storage must be BF16.
    pub fn fill_bf16_emulated(&self) -> Result<()> {
        let buffer = match &self.buffer {
            VulkanStorageBuffer::BF16(buffer) => buffer,
            _ => {
                return Err(Error::Vulkan(
                    "fill_bf16_emulated requires BF16 storage"
                        .to_string()
                        .into(),
                ))
            }
        };
        let kernels = self.device.kernels().clone();
        let total = buffer.len() as usize;
        self.run_fill("fill_bf16_emulated", move |cbb| {
            candle_vulkan_kernels::call_test_fill_bf16_emulated(
                cbb,
                kernels.as_ref(),
                buffer,
                total,
            )
        })
    }
    /// Fills this u32 storage with 256 distinct values, each beyond the 16-bit
    /// unsigned range. The storage must be U32.
    pub fn fill_u32(&self) -> Result<()> {
        let buffer = match &self.buffer {
            VulkanStorageBuffer::U32(buffer) => buffer,
            _ => {
                return Err(Error::Vulkan(
                    "fill_u32 requires U32 storage".to_string().into(),
                ))
            }
        };
        let kernels = self.device.kernels().clone();
        let total = buffer.len() as usize;
        self.run_fill("fill_u32", move |cbb| {
            candle_vulkan_kernels::call_test_fill_u32(cbb, kernels.as_ref(), buffer, total)
        })
    }
    /// Fills this i32 storage with 256 distinct signed values straddling the
    /// 16-bit range. The storage must be I32.
    pub fn fill_i32(&self) -> Result<()> {
        let buffer = match &self.buffer {
            VulkanStorageBuffer::I32(buffer) => buffer,
            _ => {
                return Err(Error::Vulkan(
                    "fill_i32 requires I32 storage".to_string().into(),
                ))
            }
        };
        let kernels = self.device.kernels().clone();
        let total = buffer.len() as usize;
        self.run_fill("fill_i32", move |cbb| {
            candle_vulkan_kernels::call_test_fill_i32(cbb, kernels.as_ref(), buffer, total)
        })
    }
    /// Fills this f32 storage with 256 distinct values, some beyond the f16
    /// range. The storage must be F32.
    pub fn fill_f32(&self) -> Result<()> {
        let buffer = match &self.buffer {
            VulkanStorageBuffer::F32(buffer) => buffer,
            _ => {
                return Err(Error::Vulkan(
                    "fill_f32 requires F32 storage".to_string().into(),
                ))
            }
        };
        let kernels = self.device.kernels().clone();
        let total = buffer.len() as usize;
        self.run_fill("fill_f32", move |cbb| {
            candle_vulkan_kernels::call_test_fill_f32(cbb, kernels.as_ref(), buffer, total)
        })
    }
    /// Fills this i64 storage with 256 distinct values, each beyond the 32-bit
    /// signed range. The storage must be I64.
    pub fn fill_i64(&self) -> Result<()> {
        let buffer = match &self.buffer {
            VulkanStorageBuffer::I64(buffer) => buffer,
            _ => {
                return Err(Error::Vulkan(
                    "fill_i64 requires I64 storage".to_string().into(),
                ))
            }
        };
        let kernels = self.device.kernels().clone();
        let total = buffer.len() as usize;
        self.run_fill("fill_i64", move |cbb| {
            candle_vulkan_kernels::call_test_fill_i64(cbb, kernels.as_ref(), buffer, total)
        })
    }
    /// Fills this f64 storage with 256 distinct values not exactly
    /// representable in f32. The storage must be F64.
    pub fn fill_f64(&self) -> Result<()> {
        let buffer = match &self.buffer {
            VulkanStorageBuffer::F64(buffer) => buffer,
            _ => {
                return Err(Error::Vulkan(
                    "fill_f64 requires F64 storage".to_string().into(),
                ))
            }
        };
        let kernels = self.device.kernels().clone();
        let total = buffer.len() as usize;
        self.run_fill("fill_f64", move |cbb| {
            candle_vulkan_kernels::call_test_fill_f64(cbb, kernels.as_ref(), buffer, total)
        })
    }

    /// Fills this F4 (4-bit) storage with all possible raw byte values:
    /// element `i` holds the bit pattern `i`. The storage must be F4.
    pub fn fill_f4(&self) -> Result<()> {
        let buffer = match &self.buffer {
            VulkanStorageBuffer::F4(buffer) => buffer,
            _ => {
                return Err(Error::Vulkan(
                    "fill_f4 requires F4 storage".to_string().into(),
                ))
            }
        };
        let kernels = self.device.kernels().clone();
        let total = buffer.len() as usize;
        self.run_fill("fill_f4", move |cbb| {
            candle_vulkan_kernels::call_test_fill_f4(cbb, kernels.as_ref(), buffer, total)
        })
    }

    /// Fills this F6E2M3 (6-bit) storage with all possible raw byte values:
    /// element `i` holds the bit pattern `i`. The storage must be F6E2M3.
    pub fn fill_f6e2m3(&self) -> Result<()> {
        let buffer = match &self.buffer {
            VulkanStorageBuffer::F6E2M3(buffer) => buffer,
            _ => {
                return Err(Error::Vulkan(
                    "fill_f6e2m3 requires F6E2M3 storage".to_string().into(),
                ))
            }
        };
        let kernels = self.device.kernels().clone();
        let total = buffer.len() as usize;
        self.run_fill("fill_f6e2m3", move |cbb| {
            candle_vulkan_kernels::call_test_fill_f6e2m3(cbb, kernels.as_ref(), buffer, total)
        })
    }

    /// Fills this F6E3M2 (6-bit) storage with all possible raw byte values:
    /// element `i` holds the bit pattern `i`. The storage must be F6E3M2.
    pub fn fill_f6e3m2(&self) -> Result<()> {
        let buffer = match &self.buffer {
            VulkanStorageBuffer::F6E3M2(buffer) => buffer,
            _ => {
                return Err(Error::Vulkan(
                    "fill_f6e3m2 requires F6E3M2 storage".to_string().into(),
                ))
            }
        };
        let kernels = self.device.kernels().clone();
        let total = buffer.len() as usize;
        self.run_fill("fill_f6e3m2", move |cbb| {
            candle_vulkan_kernels::call_test_fill_f6e3m2(cbb, kernels.as_ref(), buffer, total)
        })
    }

    /// Fills this F8E4M3 (FP8 E4M3 (emulated as raw bytes)) storage with all possible raw byte values:
    /// element `i` holds the bit pattern `i`. The storage must be F8E4M3.
    pub fn fill_f8e4m3(&self) -> Result<()> {
        let buffer = match &self.buffer {
            VulkanStorageBuffer::F8E4M3(buffer) => buffer,
            _ => {
                return Err(Error::Vulkan(
                    "fill_f8e4m3 requires F8E4M3 storage".to_string().into(),
                ))
            }
        };
        let kernels = self.device.kernels().clone();
        let total = buffer.len() as usize;
        self.run_fill("fill_f8e4m3", move |cbb| {
            candle_vulkan_kernels::call_test_fill_f8e4m3(cbb, kernels.as_ref(), buffer, total)
        })
    }

    /// Fills this F8E8M0 (FP8 E8M0 scale (emulated as raw bytes)) storage with all possible raw byte values:
    /// element `i` holds the bit pattern `i`. The storage must be F8E8M0.
    pub fn fill_f8e8m0(&self) -> Result<()> {
        let buffer = match &self.buffer {
            VulkanStorageBuffer::F8E8M0(buffer) => buffer,
            _ => {
                return Err(Error::Vulkan(
                    "fill_f8e8m0 requires F8E8M0 storage".to_string().into(),
                ))
            }
        };
        let kernels = self.device.kernels().clone();
        let total = buffer.len() as usize;
        self.run_fill("fill_f8e8m0", move |cbb| {
            candle_vulkan_kernels::call_test_fill_f8e8m0(cbb, kernels.as_ref(), buffer, total)
        })
    }
}

impl BackendStorage for VulkanStorage {
    type Device = VulkanDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        todo!()
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        let device = self.device.clone();
        // Drain any deferred encodes first: the readback copy is submitted
        // on the same queue and must not land ahead of pending work.
        device.synchronize()?;
        Ok(match &self.buffer {
            VulkanStorageBuffer::U8(b) => CpuStorage::U8(Self::copy_to_host(&device, b)?),
            VulkanStorageBuffer::U32(b) => CpuStorage::U32(Self::copy_to_host(&device, b)?),
            VulkanStorageBuffer::I16(b) => CpuStorage::I16(Self::copy_to_host(&device, b)?),
            VulkanStorageBuffer::I32(b) => CpuStorage::I32(Self::copy_to_host(&device, b)?),
            VulkanStorageBuffer::I64(b) => CpuStorage::I64(Self::copy_to_host(&device, b)?),
            VulkanStorageBuffer::BF16(b) => CpuStorage::BF16(Self::copy_to_host(&device, b)?),
            VulkanStorageBuffer::F16(b) => CpuStorage::F16(Self::copy_to_host(&device, b)?),
            VulkanStorageBuffer::F32(b) => CpuStorage::F32(Self::copy_to_host(&device, b)?),
            VulkanStorageBuffer::F64(b) => CpuStorage::F64(Self::copy_to_host(&device, b)?),
            VulkanStorageBuffer::F8E4M3(b) => CpuStorage::F8E4M3(Self::copy_to_host(&device, b)?),
            VulkanStorageBuffer::F6E2M3(b) => CpuStorage::F6E2M3(Self::copy_to_host(&device, b)?),
            VulkanStorageBuffer::F6E3M2(b) => CpuStorage::F6E3M2(Self::copy_to_host(&device, b)?),
            VulkanStorageBuffer::F4(b) => CpuStorage::F4(Self::copy_to_host(&device, b)?),
            VulkanStorageBuffer::F8E8M0(b) => CpuStorage::F8E8M0(Self::copy_to_host(&device, b)?),
        })
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
