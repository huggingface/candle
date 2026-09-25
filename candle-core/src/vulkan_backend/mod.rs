//! The Vulkan backend: storage + device trait implementations.
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryAutoCommandBuffer,
};

use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::sync::GpuFuture;

pub use crate::vulkan_backend::device::{VBuf, VulkanDevice};

use half::{bf16, f16};

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
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
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
                    (0..elem_count).map(|_| rng.sample(uniform)).collect();
                CpuStorage::BF16(data)
            }
            DType::F16 => {
                let uniform = rand::distr::Uniform::new(
                    half::f16::from_f64(min),
                    half::f16::from_f64(max),
                )
                .map_err(Error::wrap)?;
                let data: Vec<half::f16> =
                    (0..elem_count).map(|_| rng.sample(uniform)).collect();
                CpuStorage::F16(data)
            }
            DType::F32 => {
                let uniform =
                    rand::distr::Uniform::new(min as f32, max as f32).map_err(Error::wrap)?;
                let data: Vec<f32> =
                    (0..elem_count).map(|_| rng.sample(uniform)).collect();
                CpuStorage::F32(data)
            }
            DType::F64 => {
                let uniform =
                    rand::distr::Uniform::new(min, max).map_err(Error::wrap)?;
                let data: Vec<f64> =
                    (0..elem_count).map(|_| rng.sample(uniform)).collect();
                CpuStorage::F64(data)
            }
            DType::F8E4M3 => {
                let uniform = rand::distr::Uniform::new(
                    microfloat::f8e4m3::from_f64(min),
                    microfloat::f8e4m3::from_f64(max),
                )
                .map_err(Error::wrap)?;
                let data: Vec<microfloat::f8e4m3> =
                    (0..elem_count).map(|_| rng.sample(uniform)).collect();
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

    /// Shared avg/max 2D pool setup (NCHW, F32, contiguous).
    fn pool2d(
        &self,
        l: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        name: candle_vulkan_kernels::KernelName,
        op: &'static str,
    ) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::Vulkan(
                format!("{op}: only F32 supported on the Vulkan backend").into(),
            ));
        }
        let dims = l.dims();
        if dims.len() != 4 {
            return Err(Error::Vulkan(
                format!("{op}: only 4D NCHW layouts supported on the Vulkan backend").into(),
            ));
        }
        let (b_sz, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        let (k_h, k_w) = kernel_size;
        let (s_h, s_w) = stride;
        if k_h == 0 || k_w == 0 || s_h == 0 || s_w == 0 {
            return Err(Error::Vulkan(format!("{op}: zero kernel/stride").into()));
        }
        if h < k_h || w < k_w {
            return Err(Error::Vulkan(
                format!("{op}: input smaller than the kernel").into(),
            ));
        }
        let (start, len) = match l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    format!("{op}: non-contiguous layouts not supported on the Vulkan backend")
                        .into(),
                ))
            }
        };
        if len != b_sz * c * h * w {
            return Err(Error::Vulkan(format!("{op}: size mismatch").into()));
        }
        let h_out = (h - k_h) / s_h + 1;
        let w_out = (w - k_w) / s_w + 1;
        let planes = b_sz * c;
        let total = planes * h_out * w_out;
        let src_buf = match &self.buffer {
            VulkanStorageBuffer::F32(b) => b.clone().slice(start as u64..(start + len) as u64),
            _ => unreachable!("dtype checked above"),
        };
        let out = VulkanStorage::new(&self.device, total, DType::F32)?;
        let out_buf = match &out.buffer {
            VulkanStorageBuffer::F32(b) => b.clone(),
            _ => unreachable!(),
        };
        if total == 0 {
            return Ok(out);
        }
        let kernels = self.device.kernels();
        let params_buf = Buffer::from_iter(
            self.device.mem_alloc(),
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            vec![
                planes as f32,
                h as f32,
                w as f32,
                k_h as f32,
                k_w as f32,
                s_h as f32,
                s_w as f32,
                h_out as f32,
                w_out as f32,
            ],
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let params: Subbuffer<[f32]> = params_buf;
        let src_buf = src_buf.clone();
        let out_buf = out_buf.clone();
        self.device.execute(move |cbb| {
            candle_vulkan_kernels::call_pool2d_slang_f32(
                cbb, &kernels, name, &src_buf, &out_buf, &params, total,
            )
            .map_err(|e| e.to_string())
        })?;
        Ok(out)
    }

    /// GPU scatter-set: `dst[left, idx, right] = src[left, i, right]`.
    fn scatter_set_impl(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        if self.dtype != DType::F32 || src.dtype != DType::F32 {
            return Err(Error::Vulkan(
                "scatter: only F32 data supported on the Vulkan backend".to_string().into(),
            ));
        }
        if ids.dtype != DType::U32 {
            return Err(Error::Vulkan(
                "scatter: only U32 index tensors supported on the Vulkan backend".to_string().into(),
            ));
        }
        let (dst_start, dst_len) = match l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    "scatter: non-contiguous destinations not supported on the Vulkan backend"
                        .to_string()
                        .into(),
                ))
            }
        };
        let (src_start, src_len) = match src_l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    "scatter: non-contiguous sources not supported on the Vulkan backend".to_string().into(),
                ))
            }
        };
        let (ids_start, ids_len) = match ids_l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    "scatter: non-contiguous index tensors not supported on the Vulkan backend"
                        .to_string()
                        .into(),
                ))
            }
        };
        let dst_dims = l.dims();
        let ids_dims = ids_l.dims();
        if dst_dims.len() != ids_dims.len() || dim >= dst_dims.len() {
            return Err(Error::Vulkan(
                "scatter: rank mismatch on the Vulkan backend".to_string().into(),
            ));
        }
        if dst_len != dst_dims.iter().product::<usize>()
            || src_len != ids_len
            || ids_len != ids_dims.iter().product::<usize>()
        {
            return Err(Error::Vulkan(
                "scatter: size mismatch on the Vulkan backend".to_string().into(),
            ));
        }
        let ids_dim_len = ids_dims[dim];
        let dst_dim_len = dst_dims[dim];
        let ids_right_len = ids_dims[dim + 1..].iter().product::<usize>();
        let dst_right_len = dst_dims[dim + 1..].iter().product::<usize>();
        let ids_left_len = ids_dims[..dim].iter().product::<usize>();
        let total = ids_left_len * ids_dim_len * ids_right_len;
        let ids_sub: Subbuffer<[u32]> = match &ids.buffer {
            VulkanStorageBuffer::U32(b) => b.clone().slice(ids_start as u64..(ids_start + ids_len) as u64),
            _ => unreachable!("dtype checked above"),
        };
        let src_sub: Subbuffer<[f32]> = match &src.buffer {
            VulkanStorageBuffer::F32(b) => b.clone().slice(src_start as u64..(src_start + src_len) as u64),
            _ => unreachable!("dtype checked above"),
        };
        let dst_sub: Subbuffer<[f32]> = match &self.buffer {
            VulkanStorageBuffer::F32(b) => b.clone().slice(dst_start as u64..(dst_start + dst_len) as u64),
            _ => unreachable!("dtype checked above"),
        };
        if total == 0 {
            return Ok(());
        }
        let kernels = self.device.kernels();
        let params_buf = Buffer::from_iter(
            self.device.mem_alloc(),
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            vec![
                dst_dim_len as f32,
                dst_right_len as f32,
                ids_dim_len as f32,
                ids_right_len as f32,
                ids_left_len as f32,
            ],
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let params_sub: Subbuffer<[f32]> = params_buf;
        let ids_sub = ids_sub.clone();
        let src_sub = src_sub.clone();
        let dst_sub = dst_sub.clone();
        self.device.execute(move |cbb| {
            candle_vulkan_kernels::call_scatter_slang_f32(
                cbb,
                &kernels,
                candle_vulkan_kernels::KernelName::ScatterF32,
                &src_sub,
                &dst_sub,
                &ids_sub,
                &params_sub,
                total,
            )
            .map_err(|e| e.to_string())
        })?;
        Ok(())
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

    fn affine(&self, l: &Layout, mul: f64, add: f64) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::Vulkan(
                "affine: only F32 supported on the Vulkan backend".to_string().into(),
            ));
        }
        let (start, len) = match l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    "affine: non-contiguous layouts not supported on the Vulkan backend"
                        .to_string()
                        .into(),
                ))
            }
        };
        let input = match &self.buffer {
            VulkanStorageBuffer::F32(b) => b.clone().slice(start as u64..(start + len) as u64),
            _ => unreachable!("dtype checked above"),
        };
        let out = VulkanStorage::new(&self.device, len, DType::F32)?;
        let out_buf = match &out.buffer {
            VulkanStorageBuffer::F32(b) => b.clone(),
            _ => unreachable!(),
        };
        if len == 0 {
            return Ok(out);
        }
        let kernels = self.device.kernels();
        let input = input.clone();
        let out_buf = out_buf.clone();
        let mul = mul as f32;
        let add = add as f32;
        self.device.execute(move |cbb| {
            candle_vulkan_kernels::call_affine_f32(cbb, &kernels, &input, &out_buf, mul, add)
                .map_err(|e| e.to_string())
        })?;
        Ok(out)
    }

    fn powf(&self, l: &Layout, exponent: f64) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::Vulkan(
                "powf: only F32 supported on the Vulkan backend".to_string().into(),
            ));
        }
        let (start, len) = match l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    "powf: non-contiguous layouts not supported on the Vulkan backend"
                        .to_string()
                        .into(),
                ))
            }
        };
        let input = match &self.buffer {
            VulkanStorageBuffer::F32(b) => b.clone().slice(start as u64..(start + len) as u64),
            _ => unreachable!("dtype checked above"),
        };
        let out = VulkanStorage::new(&self.device, len, DType::F32)?;
        let out_buf = match &out.buffer {
            VulkanStorageBuffer::F32(b) => b.clone(),
            _ => unreachable!(),
        };
        if len == 0 {
            return Ok(out);
        }
        let kernels = self.device.kernels();
        let params_buf = Buffer::from_iter(
            self.device.mem_alloc(),
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            vec![len as f32, exponent as f32],
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let params: Subbuffer<[f32]> = params_buf;
        let input = input.clone();
        let out_buf = out_buf.clone();
        self.device.execute(move |cbb| {
            candle_vulkan_kernels::call_unary_slang_f32(
                cbb,
                &kernels,
                candle_vulkan_kernels::KernelName::UnaryPowF32,
                &input,
                &out_buf,
                &params,
                len,
            )
            .map_err(|e| e.to_string())
        })?;
        Ok(out)
    }

    fn elu(&self, l: &Layout, alpha: f64) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::Vulkan(
                "elu: only F32 supported on the Vulkan backend".to_string().into(),
            ));
        }
        let (start, len) = match l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    "elu: non-contiguous layouts not supported on the Vulkan backend".to_string().into(),
                ))
            }
        };
        let input = match &self.buffer {
            VulkanStorageBuffer::F32(b) => b.clone().slice(start as u64..(start + len) as u64),
            _ => unreachable!("dtype checked above"),
        };
        let out = VulkanStorage::new(&self.device, len, DType::F32)?;
        let out_buf = match &out.buffer {
            VulkanStorageBuffer::F32(b) => b.clone(),
            _ => unreachable!(),
        };
        if len == 0 {
            return Ok(out);
        }
        let kernels = self.device.kernels();
        let params_buf = Buffer::from_iter(
            self.device.mem_alloc(),
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            vec![len as f32, alpha as f32],
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let params: Subbuffer<[f32]> = params_buf;
        let input = input.clone();
        let out_buf = out_buf.clone();
        self.device.execute(move |cbb| {
            candle_vulkan_kernels::call_unary_slang_f32(
                cbb,
                &kernels,
                candle_vulkan_kernels::KernelName::UnaryEluF32,
                &input,
                &out_buf,
                &params,
                len,
            )
            .map_err(|e| e.to_string())
        })?;
        Ok(out)
    }

    fn reduce_op(&self, op: ReduceOp, l: &Layout, reduce_dims: &[usize]) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::Vulkan(
                "reduce_op: only F32 supported on the Vulkan backend".to_string().into(),
            ));
        }
        let dims = l.dims();
        let ndim = dims.len();
        if ndim == 0 {
            return Err(Error::Vulkan(
                "reduce_op: scalar layouts not supported on the Vulkan backend".to_string().into(),
            ));
        }
        // Only the last axis is reducible on the Vulkan backend for now.
        if reduce_dims != [ndim - 1] {
            return Err(Error::Vulkan(
                "reduce_op: only the last axis can be reduced on the Vulkan backend"
                    .to_string()
                    .into(),
            ));
        }
        let (start, len) = match l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    "reduce_op: non-contiguous layouts not supported on the Vulkan backend"
                        .to_string()
                        .into(),
                ))
            }
        };
        let input = match &self.buffer {
            VulkanStorageBuffer::F32(b) => b.clone().slice(start as u64..(start + len) as u64),
            _ => unreachable!("dtype checked above"),
        };
        let rows: usize = dims[..ndim - 1].iter().product();
        let cols = dims[ndim - 1];
        let out_dtype = match op {
            ReduceOp::Sum | ReduceOp::Min | ReduceOp::Max => DType::F32,
            ReduceOp::ArgMin | ReduceOp::ArgMax => DType::U32,
        };
        let out = VulkanStorage::new(&self.device, rows, out_dtype)?;
        if rows == 0 || cols == 0 {
            return Ok(out);
        }
        let kernels = self.device.kernels();
        match op {
            ReduceOp::Sum => {
                let out_buf = match &out.buffer {
                    VulkanStorageBuffer::F32(b) => b.clone(),
                    _ => unreachable!(),
                };
                let input = input.clone();
                self.device.execute(move |cbb| {
                    candle_vulkan_kernels::call_reduce_sum_f32(
                        cbb, &kernels, &input, &out_buf, rows, cols,
                    )
                    .map_err(|e| e.to_string())
                })?;
            }
            ReduceOp::Max => {
                let out_buf = match &out.buffer {
                    VulkanStorageBuffer::F32(b) => b.clone(),
                    _ => unreachable!(),
                };
                let input = input.clone();
                self.device.execute(move |cbb| {
                    candle_vulkan_kernels::call_reduce_max_f32(
                        cbb, &kernels, &input, &out_buf, rows, cols,
                    )
                    .map_err(|e| e.to_string())
                })?;
            }
            ReduceOp::Min | ReduceOp::ArgMin | ReduceOp::ArgMax => {
                let name = match op {
                    ReduceOp::Min => candle_vulkan_kernels::KernelName::ReduceMinF32,
                    ReduceOp::ArgMin => candle_vulkan_kernels::KernelName::ReduceArgMinF32,
                    _ => candle_vulkan_kernels::KernelName::ReduceArgMaxF32,
                };
                let (out_f, out_i) = match &out.buffer {
                    VulkanStorageBuffer::F32(b) => (Some(b.clone()), None),
                    VulkanStorageBuffer::U32(b) => (None, Some(b.clone())),
                    _ => unreachable!(),
                };
                let params_buf = Buffer::from_iter(
                    self.device.mem_alloc(),
                    &BufferCreateInfo {
                        usage: BufferUsage::STORAGE_BUFFER,
                        ..Default::default()
                    },
                    &AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_HOST
                            | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                        ..Default::default()
                    },
                    vec![rows as f32, cols as f32],
                )
                .map_err(|e| Error::Vulkan(e.to_string().into()))?;
                let params: Subbuffer<[f32]> = params_buf;
                let input = input.clone();
                let of = out_f.clone();
                let oi = out_i.clone();
                self.device.execute(move |cbb| {
                    candle_vulkan_kernels::call_reduce_slang_f32(
                        cbb, &kernels, name, &input, of.as_ref(), oi.as_ref(), &params, rows,
                    )
                    .map_err(|e| e.to_string())
                })?;
            }
        }
        Ok(out)
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        use candle_vulkan_kernels::KernelName;
        if self.dtype != DType::F32 || rhs.dtype != DType::F32 {
            return Err(Error::Vulkan(
                "cmp: only F32 inputs supported on the Vulkan backend".to_string().into(),
            ));
        }
        let (start, len) = match lhs_l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    "cmp: non-contiguous lhs not supported on the Vulkan backend".to_string().into(),
                ))
            }
        };
        let input = match &self.buffer {
            VulkanStorageBuffer::F32(b) => b.clone().slice(start as u64..(start + len) as u64),
            _ => unreachable!("dtype checked above"),
        };
        let rhs_buf = match &rhs.buffer {
            VulkanStorageBuffer::F32(b) => b.clone(),
            _ => unreachable!("dtype checked above"),
        };
        // U8 output storage (1 where the predicate holds, 0 otherwise).
        let out = VulkanStorage::new(&self.device, len, DType::U8)?;
        let out_buf = match &out.buffer {
            VulkanStorageBuffer::U8(b) => b.clone(),
            _ => unreachable!(),
        };
        if len == 0 {
            return Ok(out);
        }
        let kernels = self.device.kernels();
        let lhs_dims: Vec<usize> = lhs_l.dims().to_vec();
        if lhs_dims.len() > 4 {
            return Err(Error::Vulkan(
                "cmp: more than 4 dims not supported on the Vulkan backend".to_string().into(),
            ));
        }
        let rhs_dims: Vec<usize> = rhs_l
            .dims()
            .iter()
            .zip(rhs_l.stride().iter())
            .map(|(d, s)| if *s == 0 { 1 } else { *d })
            .collect();
        let name = match op {
            CmpOp::Eq => KernelName::CmpEqF32,
            CmpOp::Ne => KernelName::CmpNeF32,
            CmpOp::Lt => KernelName::CmpLtF32,
            CmpOp::Le => KernelName::CmpLeF32,
            CmpOp::Gt => KernelName::CmpGtF32,
            CmpOp::Ge => KernelName::CmpGeF32,
        };
        let mut params = [0.0f32; 11];
        params[0] = len as f32;
        params[1] = lhs_dims.len() as f32;
        params[2] = rhs_dims.len() as f32;
        for (slot, d) in params[3..7].iter_mut().zip(lhs_dims.iter()) {
            *slot = *d as f32;
        }
        for (slot, d) in params[7..11].iter_mut().zip(rhs_dims.iter()) {
            *slot = *d as f32;
        }
        let params_buf = Buffer::from_iter(
            self.device.mem_alloc(),
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            params.to_vec(),
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let params: Subbuffer<[f32]> = params_buf;
        let input = input.clone();
        let rhs_buf = rhs_buf.clone();
        let out_buf = out_buf.clone();
        self.device.execute(move |cbb| {
            candle_vulkan_kernels::call_cmp_slang_f32(
                cbb, &kernels, name, &input, &rhs_buf, &out_buf, &params, len,
            )
            .map_err(|e| e.to_string())
        })?;
        Ok(out)
    }

    fn to_dtype(&self, l: &Layout, dtype: DType) -> Result<Self> {
        let (start, len) = match l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    "to_dtype: non-contiguous layouts not supported on the Vulkan backend"
                        .to_string()
                        .into(),
                ))
            }
        };
        // Host-side conversion: drain the queue, convert the contiguous
        // block with the same `half`-crate casts as the CPU backend,
        // upload the result.
        let src = self.to_cpu_storage()?;
        let converted = match (&src, dtype) {
            (CpuStorage::F32(d), DType::F16) => {
                CpuStorage::F16(d[start..start + len].iter().map(|v| f16::from_f32(*v)).collect())
            }
            (CpuStorage::F16(d), DType::F32) => {
                CpuStorage::F32(d[start..start + len].iter().map(|v| f16::to_f32(*v)).collect())
            }
            (CpuStorage::F32(d), DType::BF16) => {
                CpuStorage::BF16(d[start..start + len].iter().map(|v| bf16::from_f32(*v)).collect())
            }
            (CpuStorage::BF16(d), DType::F32) => {
                CpuStorage::F32(d[start..start + len].iter().map(|v| bf16::to_f32(*v)).collect())
            }
            (CpuStorage::F16(d), DType::BF16) => CpuStorage::BF16(
                d[start..start + len]
                    .iter()
                    .map(|v| bf16::from_f32(f16::to_f32(*v)))
                    .collect(),
            ),
            (CpuStorage::BF16(d), DType::F16) => CpuStorage::F16(
                d[start..start + len]
                    .iter()
                    .map(|v| f16::from_f32(bf16::to_f32(*v)))
                    .collect(),
            ),
            _ => {
                let msg = format!(
                    "to_dtype: {:?} -> {:?} not supported on the Vulkan backend",
                    self.dtype, dtype
                );
                return Err(Error::Vulkan(msg.into()));
            }
        };
        self.device.storage_from_cpu_storage(&converted)
    }

    fn unary_impl<B: UnaryOpT>(&self, l: &Layout) -> Result<Self> {
        use candle_vulkan_kernels::KernelName;
        if self.dtype != DType::F32 {
            return Err(Error::Vulkan(
                "unary: only F32 supported on the Vulkan backend".to_string().into(),
            ));
        }
        let (start, len) = match l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    "unary: non-contiguous layout not supported on the Vulkan backend"
                        .to_string()
                        .into(),
                ))
            }
        };
        let input = match &self.buffer {
            VulkanStorageBuffer::F32(b) => b.clone().slice(start as u64..(start + len) as u64),
            _ => unreachable!("dtype checked above"),
        };
        let out = self.device.zeros_impl(l.shape(), DType::F32)?;
        let out_buf = match &out.buffer {
            VulkanStorageBuffer::F32(b) => b.clone(),
            _ => unreachable!(),
        };
        if len == 0 {
            return Ok(out);
        }
        let kernels = self.device.kernels();
        // Existing WGSL kernels.
        let wgsl: Option<KernelName> = match B::NAME {
            "exp" => Some(KernelName::ElemExpF32),
            "sin" => Some(KernelName::ElemSinF32),
            "cos" => Some(KernelName::ElemCosF32),
            "neg" => Some(KernelName::ElemNegF32),
            "sqrt" => Some(KernelName::ElemSqrtF32),
            "silu" => Some(KernelName::ElemSiluF32),
            _ => None,
        };
        if let Some(name) = wgsl {
            let input = input.clone();
            let out_buf = out_buf.clone();
            self.device.execute(move |cbb| {
                candle_vulkan_kernels::call_elem_unary_f32(cbb, &kernels, name, &input, &out_buf)
                    .map_err(|e| e.to_string())
            })?;
            return Ok(out);
        }
        // Slang kernels (params: [0] = element count).
        let name = match B::NAME {
            "log" => KernelName::UnaryLogF32,
            "abs" => KernelName::UnaryAbsF32,
            "recip" => KernelName::UnaryRecipF32,
            "sqr" => KernelName::UnarySqrF32,
            "gelu" => KernelName::UnaryGeluF32,
            "gelu_erf" => KernelName::UnaryGeluErfF32,
            "erf" => KernelName::UnaryErfF32,
            "relu" => KernelName::UnaryReluF32,
            "tanh" => KernelName::UnaryTanhF32,
            "floor" => KernelName::UnaryFloorF32,
            "ceil" => KernelName::UnaryCeilF32,
            "round" => KernelName::UnaryRoundF32,
            "sign" => KernelName::UnarySignF32,
            _ => {
                return Err(Error::Vulkan(
                    format!("unary: unsupported op {}", B::NAME).into(),
                ))
            }
        };
        let params_vec = vec![len as f32, 0.0f32];
        let params_buf = Buffer::from_iter(
            self.device.mem_alloc(),
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            params_vec,
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let params: Subbuffer<[f32]> = params_buf;
        let input = input.clone();
        let out_buf = out_buf.clone();
        self.device.execute(move |cbb| {
            candle_vulkan_kernels::call_unary_slang_f32(
                cbb, &kernels, name, &input, &out_buf, &params, len,
            )
            .map_err(|e| e.to_string())
        })?;
        Ok(out)
    }

    fn binary_impl<B: BinaryOpT>(&self, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        use candle_vulkan_kernels::KernelName;
        if self.dtype != DType::F32 || rhs.dtype != DType::F32 {
            return Err(Error::Vulkan(
                "binary: only F32 supported on the Vulkan backend".to_string().into(),
            ));
        }
        let (start, len) = match lhs_l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    "binary: non-contiguous lhs not supported on the Vulkan backend"
                        .to_string()
                        .into(),
                ))
            }
        };
        let input = match &self.buffer {
            VulkanStorageBuffer::F32(b) => b.clone().slice(start as u64..(start + len) as u64),
            _ => unreachable!("dtype checked above"),
        };
        // The rhs buffer is always bound whole: broadcast offsets are
        // computed relative to the buffer start.
        let rhs_buf = match &rhs.buffer {
            VulkanStorageBuffer::F32(b) => b.clone(),
            _ => unreachable!("dtype checked above"),
        };
        let out = self.device.zeros_impl(lhs_l.shape(), DType::F32)?;
        let out_buf = match &out.buffer {
            VulkanStorageBuffer::F32(b) => b.clone(),
            _ => unreachable!(),
        };
        if len == 0 {
            return Ok(out);
        }
        let kernels = self.device.kernels();
        let lhs_dims: Vec<usize> = lhs_l.dims().to_vec();
        if lhs_dims.len() > 4 {
            return Err(Error::Vulkan(
                "binary: more than 4 dims not supported on the Vulkan backend".to_string().into(),
            ));
        }
        // Broadcast dims (stride 0) index a single element.
        let rhs_dims: Vec<usize> = rhs_l
            .dims()
            .iter()
            .zip(rhs_l.stride().iter())
            .map(|(d, s)| if *s == 0 { 1 } else { *d })
            .collect();
        match B::NAME {
            "add" | "sub" | "mul" | "div" => {
                let name = match B::NAME {
                    "add" => KernelName::ElemAddF32,
                    "sub" => KernelName::ElemSubF32,
                    "mul" => KernelName::ElemMulF32,
                    "div" => KernelName::ElemDivF32,
                    _ => unreachable!(),
                };
                let input = input.clone();
                let rhs_buf = rhs_buf.clone();
                let out_buf = out_buf.clone();
                self.device.execute(move |cbb| {
                    candle_vulkan_kernels::call_elem_binary_f32(
                        cbb, &kernels, name, &input, &rhs_buf, &out_buf, &lhs_dims, &rhs_dims,
                    )
                    .map_err(|e| e.to_string())
                })?;
                Ok(out)
            }
            "maximum" | "minimum" => {
                let name = match B::NAME {
                    "maximum" => KernelName::BinaryMaximumF32,
                    "minimum" => KernelName::BinaryMinimumF32,
                    _ => unreachable!(),
                };
                let mut params = [0.0f32; 11];
                params[0] = len as f32;
                params[1] = lhs_dims.len() as f32;
                params[2] = rhs_dims.len() as f32;
                for (slot, d) in params[3..7].iter_mut().zip(lhs_dims.iter()) {
                    *slot = *d as f32;
                }
                for (slot, d) in params[7..11].iter_mut().zip(rhs_dims.iter()) {
                    *slot = *d as f32;
                }
                let params_buf = Buffer::from_iter(
                    self.device.mem_alloc(),
                    &BufferCreateInfo {
                        usage: BufferUsage::STORAGE_BUFFER,
                        ..Default::default()
                    },
                    &AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_HOST
                            | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                        ..Default::default()
                    },
                    params.to_vec(),
                )
                .map_err(|e| Error::Vulkan(e.to_string().into()))?;
                let params: Subbuffer<[f32]> = params_buf;
                let input = input.clone();
                let rhs_buf = rhs_buf.clone();
                let out_buf = out_buf.clone();
                self.device.execute(move |cbb| {
                    candle_vulkan_kernels::call_binary_slang_f32(
                        cbb, &kernels, name, &input, &rhs_buf, &out_buf, &params, len,
                    )
                    .map_err(|e| e.to_string())
                })?;
                Ok(out)
            }
            _ => Err(Error::Vulkan(format!("binary: unsupported op {}", B::NAME).into())),
        }
    }

    fn where_cond(&self, l: &Layout, t: &Self, t_l: &Layout, f: &Self, f_l: &Layout) -> Result<Self> {
        if t.dtype != DType::F32 || f.dtype != DType::F32 {
            return Err(Error::Vulkan(
                "where_cond: only F32 values supported on the Vulkan backend".to_string().into(),
            ));
        }
        let (start, len) = match l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    "where_cond: non-contiguous pred not supported on the Vulkan backend"
                        .to_string()
                        .into(),
                ))
            }
        };
        let pred = match &self.buffer {
            VulkanStorageBuffer::U8(b) => b.clone().slice(start as u64..(start + len) as u64),
            _ => {
                return Err(Error::Vulkan(
                    "where_cond: only U8 predicates supported on the Vulkan backend".to_string().into(),
                ))
            }
        };
        let t_buf = match &t.buffer {
            VulkanStorageBuffer::F32(b) => b.clone(),
            _ => unreachable!("dtype checked above"),
        };
        let f_buf = match &f.buffer {
            VulkanStorageBuffer::F32(b) => b.clone(),
            _ => unreachable!("dtype checked above"),
        };
        let out = VulkanStorage::new(&self.device, len, DType::F32)?;
        let out_buf = match &out.buffer {
            VulkanStorageBuffer::F32(b) => b.clone(),
            _ => unreachable!(),
        };
        if len == 0 {
            return Ok(out);
        }
        let kernels = self.device.kernels();
        let ndim = l.dims().len();
        if ndim > 4 {
            return Err(Error::Vulkan(
                "where_cond: more than 4 dims not supported on the Vulkan backend".to_string().into(),
            ));
        }
        let eff = |ly: &Layout| -> Vec<usize> {
            ly.dims()
                .iter()
                .zip(ly.stride().iter())
                .map(|(d, s)| if *s == 0 { 1 } else { *d })
                .collect()
        };
        let p_dims = eff(l);
        let t_dims = eff(t_l);
        let f_dims = eff(f_l);
        let mut params = [0.0f32; 14];
        params[0] = len as f32;
        params[1] = ndim as f32;
        for (slot, d) in params[2..6].iter_mut().zip(p_dims.iter()) {
            *slot = *d as f32;
        }
        for (slot, d) in params[6..10].iter_mut().zip(t_dims.iter()) {
            *slot = *d as f32;
        }
        for (slot, d) in params[10..14].iter_mut().zip(f_dims.iter()) {
            *slot = *d as f32;
        }
        let params_buf = Buffer::from_iter(
            self.device.mem_alloc(),
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            params.to_vec(),
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let params: Subbuffer<[f32]> = params_buf;
        let pred = pred.clone();
        let t_buf = t_buf.clone();
        let f_buf = f_buf.clone();
        let out_buf = out_buf.clone();
        self.device.execute(move |cbb| {
            candle_vulkan_kernels::call_where_slang_f32(
                cbb,
                &kernels,
                candle_vulkan_kernels::KernelName::WhereF32,
                &pred,
                &t_buf,
                &f_buf,
                &out_buf,
                &params,
                len,
            )
            .map_err(|e| e.to_string())
        })?;
        Ok(out)
    }

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        if self.dtype != DType::F32 || kernel.dtype != DType::F32 {
            return Err(Error::Vulkan(
                "conv1d: only F32 supported on the Vulkan backend".to_string().into(),
            ));
        }
        let (in_start, in_len) = match l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    "conv1d: non-contiguous inputs not supported on the Vulkan backend".to_string().into(),
                ))
            }
        };
        let (w_start, w_len) = match kernel_l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    "conv1d: non-contiguous kernels not supported on the Vulkan backend".to_string().into(),
                ))
            }
        };
        if in_len != params.b_size * params.c_in * params.l_in
            || w_len != params.c_out * params.c_in * params.k_size
        {
            return Err(Error::Vulkan(
                "conv1d: size mismatch on the Vulkan backend".to_string().into(),
            ));
        }
        let l_out = params.l_out();
        let total = params.b_size * params.c_out * l_out;
        let in_buf = match &self.buffer {
            VulkanStorageBuffer::F32(b) => b.clone().slice(in_start as u64..(in_start + in_len) as u64),
            _ => unreachable!("dtype checked above"),
        };
        let w_buf = match &kernel.buffer {
            VulkanStorageBuffer::F32(b) => b.clone().slice(w_start as u64..(w_start + w_len) as u64),
            _ => unreachable!("dtype checked above"),
        };
        let out = VulkanStorage::new(&self.device, total, DType::F32)?;
        let out_buf = match &out.buffer {
            VulkanStorageBuffer::F32(b) => b.clone(),
            _ => unreachable!(),
        };
        if total == 0 {
            return Ok(out);
        }
        let kernels = self.device.kernels();
        let params_buf = Buffer::from_iter(
            self.device.mem_alloc(),
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            vec![
                params.b_size as f32,
                params.c_in as f32,
                params.c_out as f32,
                params.k_size as f32,
                params.l_in as f32,
                l_out as f32,
                params.padding as f32,
                params.stride as f32,
                params.dilation as f32,
            ],
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let params_sub: Subbuffer<[f32]> = params_buf;
        let in_buf = in_buf.clone();
        let w_buf = w_buf.clone();
        let out_buf = out_buf.clone();
        self.device.execute(move |cbb| {
            candle_vulkan_kernels::call_conv_slang_f32(
                cbb,
                &kernels,
                candle_vulkan_kernels::KernelName::Conv1dF32,
                &in_buf,
                &w_buf,
                &out_buf,
                &params_sub,
                total,
            )
            .map_err(|e| e.to_string())
        })?;
        Ok(out)
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        if self.dtype != DType::F32 || kernel.dtype != DType::F32 {
            return Err(Error::Vulkan(
                "conv_transpose1d: only F32 supported on the Vulkan backend".to_string().into(),
            ));
        }
        let (in_start, in_len) = match l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    "conv_transpose1d: non-contiguous inputs not supported on the Vulkan backend"
                        .to_string()
                        .into(),
                ))
            }
        };
        let (w_start, w_len) = match kernel_l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    "conv_transpose1d: non-contiguous kernels not supported on the Vulkan backend"
                        .to_string()
                        .into(),
                ))
            }
        };
        if in_len != params.b_size * params.c_in * params.l_in
            || w_len != params.c_in * params.c_out * params.k_size
        {
            return Err(Error::Vulkan(
                "conv_transpose1d: size mismatch on the Vulkan backend".to_string().into(),
            ));
        }
        let l_out = params.l_out();
        let total = params.b_size * params.c_out * l_out;
        let in_buf = match &self.buffer {
            VulkanStorageBuffer::F32(b) => b.clone().slice(in_start as u64..(in_start + in_len) as u64),
            _ => unreachable!("dtype checked above"),
        };
        let w_buf = match &kernel.buffer {
            VulkanStorageBuffer::F32(b) => b.clone().slice(w_start as u64..(w_start + w_len) as u64),
            _ => unreachable!("dtype checked above"),
        };
        let out = VulkanStorage::new(&self.device, total, DType::F32)?;
        let out_buf = match &out.buffer {
            VulkanStorageBuffer::F32(b) => b.clone(),
            _ => unreachable!(),
        };
        if total == 0 {
            return Ok(out);
        }
        let kernels = self.device.kernels();
        let params_buf = Buffer::from_iter(
            self.device.mem_alloc(),
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            vec![
                params.b_size as f32,
                params.c_in as f32,
                params.c_out as f32,
                params.k_size as f32,
                params.l_in as f32,
                l_out as f32,
                params.padding as f32,
                params.stride as f32,
                params.dilation as f32,
            ],
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let params_sub: Subbuffer<[f32]> = params_buf;
        let in_buf = in_buf.clone();
        let w_buf = w_buf.clone();
        let out_buf = out_buf.clone();
        self.device.execute(move |cbb| {
            candle_vulkan_kernels::call_conv_slang_f32(
                cbb,
                &kernels,
                candle_vulkan_kernels::KernelName::ConvTranspose1dF32,
                &in_buf,
                &w_buf,
                &out_buf,
                &params_sub,
                total,
            )
            .map_err(|e| e.to_string())
        })?;
        Ok(out)
    }

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        if self.dtype != DType::F32 || kernel.dtype != DType::F32 {
            return Err(Error::Vulkan(
                "conv2d: only F32 supported on the Vulkan backend".to_string().into(),
            ));
        }
        let (in_start, in_len) = match l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    "conv2d: non-contiguous inputs not supported on the Vulkan backend".to_string().into(),
                ))
            }
        };
        let (w_start, w_len) = match kernel_l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    "conv2d: non-contiguous kernels not supported on the Vulkan backend".to_string().into(),
                ))
            }
        };
        if in_len != params.b_size * params.c_in * params.i_h * params.i_w
            || w_len != params.c_out * params.c_in * params.k_h * params.k_w
        {
            return Err(Error::Vulkan(
                "conv2d: size mismatch on the Vulkan backend".to_string().into(),
            ));
        }
        let (o_h, o_w) = (params.out_h(), params.out_w());
        let total = params.b_size * params.c_out * o_h * o_w;
        let in_buf = match &self.buffer {
            VulkanStorageBuffer::F32(b) => b.clone().slice(in_start as u64..(in_start + in_len) as u64),
            _ => unreachable!("dtype checked above"),
        };
        let w_buf = match &kernel.buffer {
            VulkanStorageBuffer::F32(b) => b.clone().slice(w_start as u64..(w_start + w_len) as u64),
            _ => unreachable!("dtype checked above"),
        };
        let out = VulkanStorage::new(&self.device, total, DType::F32)?;
        let out_buf = match &out.buffer {
            VulkanStorageBuffer::F32(b) => b.clone(),
            _ => unreachable!(),
        };
        if total == 0 {
            return Ok(out);
        }
        let kernels = self.device.kernels();
        let params_buf = Buffer::from_iter(
            self.device.mem_alloc(),
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            vec![
                params.b_size as f32,
                params.c_in as f32,
                params.c_out as f32,
                params.k_h as f32,
                params.k_w as f32,
                params.i_h as f32,
                params.i_w as f32,
                o_h as f32,
                o_w as f32,
                params.padding as f32,
                params.stride as f32,
                params.dilation as f32,
            ],
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let params_sub: Subbuffer<[f32]> = params_buf;
        let in_buf = in_buf.clone();
        let w_buf = w_buf.clone();
        let out_buf = out_buf.clone();
        self.device.execute(move |cbb| {
            candle_vulkan_kernels::call_conv_slang_f32(
                cbb,
                &kernels,
                candle_vulkan_kernels::KernelName::Conv2dF32,
                &in_buf,
                &w_buf,
                &out_buf,
                &params_sub,
                total,
            )
            .map_err(|e| e.to_string())
        })?;
        Ok(out)
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        if self.dtype != DType::F32 || kernel.dtype != DType::F32 {
            return Err(Error::Vulkan(
                "conv_transpose2d: only F32 supported on the Vulkan backend".to_string().into(),
            ));
        }
        let (in_start, in_len) = match l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    "conv_transpose2d: non-contiguous inputs not supported on the Vulkan backend"
                        .to_string()
                        .into(),
                ))
            }
        };
        let (w_start, w_len) = match kernel_l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    "conv_transpose2d: non-contiguous kernels not supported on the Vulkan backend"
                        .to_string()
                        .into(),
                ))
            }
        };
        if in_len != params.b_size * params.c_in * params.i_h * params.i_w
            || w_len != params.c_in * params.c_out * params.k_h * params.k_w
        {
            return Err(Error::Vulkan(
                "conv_transpose2d: size mismatch on the Vulkan backend".to_string().into(),
            ));
        }
        let (o_h, o_w) = (params.out_h(), params.out_w());
        let total = params.b_size * params.c_out * o_h * o_w;
        let in_buf = match &self.buffer {
            VulkanStorageBuffer::F32(b) => b.clone().slice(in_start as u64..(in_start + in_len) as u64),
            _ => unreachable!("dtype checked above"),
        };
        let w_buf = match &kernel.buffer {
            VulkanStorageBuffer::F32(b) => b.clone().slice(w_start as u64..(w_start + w_len) as u64),
            _ => unreachable!("dtype checked above"),
        };
        let out = VulkanStorage::new(&self.device, total, DType::F32)?;
        let out_buf = match &out.buffer {
            VulkanStorageBuffer::F32(b) => b.clone(),
            _ => unreachable!(),
        };
        if total == 0 {
            return Ok(out);
        }
        let kernels = self.device.kernels();
        let params_buf = Buffer::from_iter(
            self.device.mem_alloc(),
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            vec![
                params.b_size as f32,
                params.c_in as f32,
                params.c_out as f32,
                params.k_h as f32,
                params.k_w as f32,
                params.i_h as f32,
                params.i_w as f32,
                o_h as f32,
                o_w as f32,
                params.padding as f32,
                params.stride as f32,
                params.dilation as f32,
            ],
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let params_sub: Subbuffer<[f32]> = params_buf;
        let in_buf = in_buf.clone();
        let w_buf = w_buf.clone();
        let out_buf = out_buf.clone();
        self.device.execute(move |cbb| {
            candle_vulkan_kernels::call_conv_slang_f32(
                cbb,
                &kernels,
                candle_vulkan_kernels::KernelName::ConvTranspose2dF32,
                &in_buf,
                &w_buf,
                &out_buf,
                &params_sub,
                total,
            )
            .map_err(|e| e.to_string())
        })?;
        Ok(out)
    }

    fn avg_pool2d(
        &self,
        l: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        self.pool2d(
            l,
            kernel_size,
            stride,
            candle_vulkan_kernels::KernelName::AvgPool2dF32,
            "avg_pool2d",
        )
    }

    fn max_pool2d(
        &self,
        l: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        self.pool2d(
            l,
            kernel_size,
            stride,
            candle_vulkan_kernels::KernelName::MaxPool2dF32,
            "max_pool2d",
        )
    }

    fn upsample_nearest1d(&self, l: &Layout, dst_sz: usize) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::Vulkan(
                "upsample_nearest1d: only F32 supported on the Vulkan backend".to_string().into(),
            ));
        }
        let dims = l.dims();
        if dims.len() != 3 {
            return Err(Error::Vulkan(
                "upsample_nearest1d: only 3D (b, c, sz) layouts supported on the Vulkan backend"
                    .to_string()
                    .into(),
            ));
        }
        let (b_sz, c, src_sz) = (dims[0], dims[1], dims[2]);
        if src_sz == 0 {
            return Err(Error::Vulkan(
                "upsample_nearest1d: empty source".to_string().into(),
            ));
        }
        let (start, len) = match l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    "upsample_nearest1d: non-contiguous layouts not supported on the Vulkan backend"
                        .to_string()
                        .into(),
                ))
            }
        };
        if len != b_sz * c * src_sz {
            return Err(Error::Vulkan(
                "upsample_nearest1d: size mismatch on the Vulkan backend".to_string().into(),
            ));
        }
        let planes = b_sz * c;
        let total = planes * dst_sz;
        let src_buf = match &self.buffer {
            VulkanStorageBuffer::F32(b) => b.clone().slice(start as u64..(start + len) as u64),
            _ => unreachable!("dtype checked above"),
        };
        let out = VulkanStorage::new(&self.device, total, DType::F32)?;
        let out_buf = match &out.buffer {
            VulkanStorageBuffer::F32(b) => b.clone(),
            _ => unreachable!(),
        };
        if total == 0 {
            return Ok(out);
        }
        let kernels = self.device.kernels();
        let params_buf = Buffer::from_iter(
            self.device.mem_alloc(),
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            vec![planes as f32, src_sz as f32, dst_sz as f32],
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let params: Subbuffer<[f32]> = params_buf;
        let src_buf = src_buf.clone();
        let out_buf = out_buf.clone();
        self.device.execute(move |cbb| {
            candle_vulkan_kernels::call_upsample_slang_f32(
                cbb,
                &kernels,
                candle_vulkan_kernels::KernelName::UpsampleNearest1dF32,
                &src_buf,
                &out_buf,
                &params,
                total,
            )
            .map_err(|e| e.to_string())
        })?;
        Ok(out)
    }

    fn upsample_nearest2d(&self, l: &Layout, dst_h: usize, dst_w: usize) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::Vulkan(
                "upsample_nearest2d: only F32 supported on the Vulkan backend".to_string().into(),
            ));
        }
        let dims = l.dims();
        if dims.len() != 4 {
            return Err(Error::Vulkan(
                "upsample_nearest2d: only 4D NCHW layouts supported on the Vulkan backend".to_string().into(),
            ));
        }
        let (b_sz, c, src_h, src_w) = (dims[0], dims[1], dims[2], dims[3]);
        if src_h == 0 || src_w == 0 {
            return Err(Error::Vulkan(
                "upsample_nearest2d: empty source".to_string().into(),
            ));
        }
        let (start, len) = match l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    "upsample_nearest2d: non-contiguous layouts not supported on the Vulkan backend"
                        .to_string()
                        .into(),
                ))
            }
        };
        if len != b_sz * c * src_h * src_w {
            return Err(Error::Vulkan(
                "upsample_nearest2d: size mismatch on the Vulkan backend".to_string().into(),
            ));
        }
        let planes = b_sz * c;
        let total = planes * dst_h * dst_w;
        let src_buf = match &self.buffer {
            VulkanStorageBuffer::F32(b) => b.clone().slice(start as u64..(start + len) as u64),
            _ => unreachable!("dtype checked above"),
        };
        let out = VulkanStorage::new(&self.device, total, DType::F32)?;
        let out_buf = match &out.buffer {
            VulkanStorageBuffer::F32(b) => b.clone(),
            _ => unreachable!(),
        };
        if total == 0 {
            return Ok(out);
        }
        let kernels = self.device.kernels();
        let params_buf = Buffer::from_iter(
            self.device.mem_alloc(),
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            vec![
                planes as f32,
                src_h as f32,
                src_w as f32,
                dst_h as f32,
                dst_w as f32,
            ],
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let params: Subbuffer<[f32]> = params_buf;
        let src_buf = src_buf.clone();
        let out_buf = out_buf.clone();
        self.device.execute(move |cbb| {
            candle_vulkan_kernels::call_upsample_slang_f32(
                cbb,
                &kernels,
                candle_vulkan_kernels::KernelName::UpsampleNearest2dF32,
                &src_buf,
                &out_buf,
                &params,
                total,
            )
            .map_err(|e| e.to_string())
        })?;
        Ok(out)
    }

    fn upsample_bilinear2d(
        &self,
        l: &Layout,
        target_h: usize,
        target_w: usize,
        align_corners: bool,
        scale_h_factor: Option<f64>,
        scale_w_factor: Option<f64>,
    ) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::Vulkan(
                "upsample_bilinear2d: only F32 supported on the Vulkan backend".to_string().into(),
            ));
        }
        let dims = l.dims();
        if dims.len() != 4 {
            return Err(Error::Vulkan(
                "upsample_bilinear2d: only 4D NCHW layouts supported on the Vulkan backend"
                    .to_string()
                    .into(),
            ));
        }
        let (b_sz, c, src_h, src_w) = (dims[0], dims[1], dims[2], dims[3]);
        if src_h == 0 || src_w == 0 {
            return Err(Error::Vulkan(
                "upsample_bilinear2d: empty source".to_string().into(),
            ));
        }
        let (start, len) = match l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    "upsample_bilinear2d: non-contiguous layouts not supported on the Vulkan backend"
                        .to_string()
                        .into(),
                ))
            }
        };
        if len != b_sz * c * src_h * src_w {
            return Err(Error::Vulkan(
                "upsample_bilinear2d: size mismatch on the Vulkan backend".to_string().into(),
            ));
        }
        // Same scale computation as the CPU backend.
        let scale_h = if align_corners {
            if target_h > 1 {
                (src_h - 1) as f64 / (target_h - 1) as f64
            } else {
                0.0
            }
        } else if let Some(f) = scale_h_factor {
            1.0 / f
        } else {
            src_h as f64 / target_h as f64
        };
        let scale_w = if align_corners {
            if target_w > 1 {
                (src_w - 1) as f64 / (target_w - 1) as f64
            } else {
                0.0
            }
        } else if let Some(f) = scale_w_factor {
            1.0 / f
        } else {
            src_w as f64 / target_w as f64
        };
        let planes = b_sz * c;
        let total = planes * target_h * target_w;
        let src_buf = match &self.buffer {
            VulkanStorageBuffer::F32(b) => b.clone().slice(start as u64..(start + len) as u64),
            _ => unreachable!("dtype checked above"),
        };
        let out = VulkanStorage::new(&self.device, total, DType::F32)?;
        let out_buf = match &out.buffer {
            VulkanStorageBuffer::F32(b) => b.clone(),
            _ => unreachable!(),
        };
        if total == 0 {
            return Ok(out);
        }
        let kernels = self.device.kernels();
        let params_buf = Buffer::from_iter(
            self.device.mem_alloc(),
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            vec![
                planes as f32,
                src_h as f32,
                src_w as f32,
                target_h as f32,
                target_w as f32,
                scale_h as f32,
                scale_w as f32,
                (align_corners as u32) as f32,
            ],
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let params: Subbuffer<[f32]> = params_buf;
        let src_buf = src_buf.clone();
        let out_buf = out_buf.clone();
        self.device.execute(move |cbb| {
            candle_vulkan_kernels::call_upsample_slang_f32(
                cbb,
                &kernels,
                candle_vulkan_kernels::KernelName::UpsampleBilinear2dF32,
                &src_buf,
                &out_buf,
                &params,
                total,
            )
            .map_err(|e| e.to_string())
        })?;
        Ok(out)
    }

    fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::Vulkan(
                "gather: only F32 embeddings supported on the Vulkan backend".to_string().into(),
            ));
        }
        if dim != 0 {
            return Err(Error::Vulkan(
                "gather: only dim 0 supported on the Vulkan backend".to_string().into(),
            ));
        }
        let (emb_start, emb_len) = match l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    "gather: non-contiguous embeddings not supported on the Vulkan backend".to_string().into(),
                ))
            }
        };
        let emb_dims = l.dims();
        if emb_dims.len() != 2 {
            return Err(Error::Vulkan(
                "gather: only 2D embeddings supported on the Vulkan backend".to_string().into(),
            ));
        }
        let (rows, dim_len) = (emb_dims[0], emb_dims[1]);
        if emb_len != rows * dim_len {
            return Err(Error::Vulkan(
                "gather: embedding size mismatch on the Vulkan backend".to_string().into(),
            ));
        }
        // ids must be a contiguous 1D u32 vector.
        let ids_u32 = match &ids.buffer {
            VulkanStorageBuffer::U32(b) => b.clone(),
            _ => {
                return Err(Error::Vulkan(
                    "gather: only U32 ids supported on the Vulkan backend".to_string().into(),
                ))
            }
        };
        let (ids_start, ids_len) = match ids_l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    "gather: non-contiguous ids not supported on the Vulkan backend".to_string().into(),
                ))
            }
        };
        let ids_dims = ids_l.dims();
        if ids_dims.len() != 2 {
            return Err(Error::Vulkan(
                "gather: only 2D ids supported on the Vulkan backend".to_string().into(),
            ));
        }
        if ids_dims[1] > dim_len {
            return Err(Error::Vulkan(
                "gather: ids inner dim larger than the embedding dim on the Vulkan backend"
                    .to_string()
                    .into(),
            ));
        }
        if ids_len != ids_dims[0] * ids_dims[1] {
            return Err(Error::Vulkan(
                "gather: ids size mismatch on the Vulkan backend".to_string().into(),
            ));
        }
        let inner = ids_dims[1];
        let ids_buf = ids_u32.slice(ids_start as u64..(ids_start + ids_len) as u64);
        let emb_buf = match &self.buffer {
            VulkanStorageBuffer::F32(b) => b.clone().slice(emb_start as u64..(emb_start + emb_len) as u64),
            _ => unreachable!("dtype checked above"),
        };
        let out = VulkanStorage::new(&self.device, ids_len, DType::F32)?;
        let out_buf = match &out.buffer {
            VulkanStorageBuffer::F32(b) => b.clone(),
            _ => unreachable!(),
        };
        if ids_len == 0 {
            return Ok(out);
        }
        let kernels = self.device.kernels();
        let ids_buf = ids_buf.clone();
        let emb_buf = emb_buf.clone();
        let out_buf = out_buf.clone();
        let params_buf = Buffer::from_iter(
            self.device.mem_alloc(),
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            vec![ids_len as f32, inner as f32, dim_len as f32, 0.0f32],
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let params: Subbuffer<[f32]> = params_buf;
        self.device.execute(move |cbb| {
            candle_vulkan_kernels::call_gather_idx_slang_f32(
                cbb,
                &kernels,
                candle_vulkan_kernels::KernelName::GatherRowsF32,
                &emb_buf,
                &out_buf,
                &ids_buf,
                &params,
                ids_len,
            )
            .map_err(|e| e.to_string())
        })?;
        Ok(out)
    }

    fn scatter_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        self.scatter_set_impl(l, ids, ids_l, src, src_l, dim)
    }


    fn scatter_add_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        // Host-side: the add needs float atomics, which are not part of the
        // Vulkan core (SPV_EXT_shader_atomic_float_add is device-dependent).
        // Drain, apply the CPU scatter-add semantics in place on the contiguous
        // blocks, and upload the full buffer back.
        if self.dtype != DType::F32 || src.dtype != DType::F32 {
            return Err(Error::Vulkan(
                "scatter_add: only F32 data supported on the Vulkan backend".to_string().into(),
            ));
        }
        let dst_dims = l.dims();
        let ids_dims = ids_l.dims();
        if dst_dims.len() != ids_dims.len() || dim >= dst_dims.len() {
            return Err(Error::Vulkan(
                "scatter_add: rank mismatch on the Vulkan backend".to_string().into(),
            ));
        }
        let (dst_start, dst_len) = match l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    "scatter_add: non-contiguous destinations not supported on the Vulkan backend"
                        .to_string()
                        .into(),
                ))
            }
        };
        let (src_start, src_len) = match src_l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    "scatter_add: non-contiguous sources not supported on the Vulkan backend"
                        .to_string()
                        .into(),
                ))
            }
        };
        let (ids_start, ids_len) = match ids_l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    "scatter_add: non-contiguous index tensors not supported on the Vulkan backend"
                        .to_string()
                        .into(),
                ))
            }
        };
        if dst_len != dst_dims.iter().product::<usize>()
            || src_len != ids_len
            || ids_len != ids_dims.iter().product::<usize>()
        {
            return Err(Error::Vulkan(
                "scatter_add: size mismatch on the Vulkan backend".to_string().into(),
            ));
        }
        // Drain any deferred encodes (e.g. the copy that materialized this
        // storage) before reading the buffer contents.
        self.device.synchronize()?;
        let dst_v: Vec<f32> = match &self.buffer {
            VulkanStorageBuffer::F32(b) => Self::copy_to_host(&self.device, b)?,
            _ => unreachable!("dtype checked above"),
        };
        let src_v: Vec<f32> = match &src.buffer {
            VulkanStorageBuffer::F32(b) => Self::copy_to_host(&self.device, b)?,
            _ => unreachable!("dtype checked above"),
        };
        let ids_cpu = match &ids.buffer {
            VulkanStorageBuffer::U32(b) => {
                let v: Vec<u32> = Self::copy_to_host(&self.device, b)?;
                CpuStorage::U32(v)
            }
            VulkanStorageBuffer::I64(b) => {
                let v: Vec<i64> = Self::copy_to_host(&self.device, b)?;
                CpuStorage::I64(v)
            }
            VulkanStorageBuffer::U8(b) => {
                let v: Vec<u8> = Self::copy_to_host(&self.device, b)?;
                CpuStorage::U8(v)
            }
            _ => {
                return Err(Error::Vulkan(
                    "scatter_add: index dtype not supported on the Vulkan backend".to_string().into(),
                ))
            }
        };
        let mut dst_cpu = CpuStorage::F32(dst_v);
        let src_cpu = CpuStorage::F32(src_v);
        let l_c = Layout::contiguous_with_offset(dst_dims.to_vec(), dst_start);
        let s_c = Layout::contiguous_with_offset(ids_dims.to_vec(), src_start);
        let i_c = Layout::contiguous_with_offset(ids_dims.to_vec(), ids_start);
        dst_cpu.scatter_add_set(&l_c, &ids_cpu, &i_c, &src_cpu, &s_c, dim)?;
        let CpuStorage::F32(v) = dst_cpu else {
            unreachable!("scatter_add only handles F32 data")
        };
        let updated = self.device.storage_from_cpu_storage(&CpuStorage::F32(v))?;
        self.buffer = updated.buffer;
        Ok(())
    }

    fn index_select(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::Vulkan(
                "index_select: only F32 supported on the Vulkan backend".to_string().into(),
            ));
        }
        if dim != 0 {
            return Err(Error::Vulkan(
                "index_select: only dim 0 supported on the Vulkan backend".to_string().into(),
            ));
        }
        let src_dims = l.dims();
        if src_dims.len() != 2 {
            return Err(Error::Vulkan(
                "index_select: only 2D sources supported on the Vulkan backend".to_string().into(),
            ));
        }
        let (rows, dim_len) = (src_dims[0], src_dims[1]);
        let (src_start, src_len) = match l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    "index_select: non-contiguous sources not supported on the Vulkan backend"
                        .to_string()
                        .into(),
                ))
            }
        };
        if src_len != rows * dim_len {
            return Err(Error::Vulkan(
                "index_select: source size mismatch on the Vulkan backend".to_string().into(),
            ));
        }
        let ids_u32 = match &ids.buffer {
            VulkanStorageBuffer::U32(b) => b.clone(),
            _ => {
                return Err(Error::Vulkan(
                    "index_select: only U32 ids supported on the Vulkan backend".to_string().into(),
                ))
            }
        };
        let ids_dims = ids_l.dims();
        if ids_dims.len() != 1 {
            return Err(Error::Vulkan(
                "index_select: only 1D ids supported on the Vulkan backend".to_string().into(),
            ));
        }
        let n_ids = ids_dims[0];
        let (ids_start, ids_len) = match ids_l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    "index_select: non-contiguous ids not supported on the Vulkan backend".to_string().into(),
                ))
            }
        };
        if ids_len != n_ids {
            return Err(Error::Vulkan(
                "index_select: ids size mismatch on the Vulkan backend".to_string().into(),
            ));
        }
        let src_buf = match &self.buffer {
            VulkanStorageBuffer::F32(b) => b.clone().slice(src_start as u64..(src_start + src_len) as u64),
            _ => unreachable!("dtype checked above"),
        };
        let ids_buf = ids_u32.slice(ids_start as u64..(ids_start + ids_len) as u64);
        let out = VulkanStorage::new(&self.device, n_ids * dim_len, DType::F32)?;
        let out_buf = match &out.buffer {
            VulkanStorageBuffer::F32(b) => b.clone(),
            _ => unreachable!(),
        };
        if n_ids == 0 {
            return Ok(out);
        }
        let kernels = self.device.kernels();
        let params_buf = Buffer::from_iter(
            self.device.mem_alloc(),
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            vec![
                (n_ids * dim_len) as f32,
                dim_len as f32,
                dim_len as f32,
                0.0f32,
            ],
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let params: Subbuffer<[f32]> = params_buf;
        let src_buf = src_buf.clone();
        let ids_buf = ids_buf.clone();
        let out_buf = out_buf.clone();
        self.device.execute(move |cbb| {
            candle_vulkan_kernels::call_gather_idx_slang_f32(
                cbb,
                &kernels,
                candle_vulkan_kernels::KernelName::IndexSelectF32,
                &src_buf,
                &out_buf,
                &ids_buf,
                &params,
                n_ids * dim_len,
            )
            .map_err(|e| e.to_string())
        })?;
        Ok(out)
    }

    fn index_add(
        &self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        // Host-side: needs float atomics for duplicate index positions, which
        // are not part of the Vulkan core (SPV_EXT_shader_atomic_float_add is
        // device-dependent). Drain, apply the CPU index-add semantics, upload.
        if self.dtype != DType::F32 || src.dtype != DType::F32 {
            return Err(Error::Vulkan(
                "index_add: only F32 data supported on the Vulkan backend".to_string().into(),
            ));
        }
        if dim >= l.dims().len() {
            return Err(Error::Vulkan(
                "index_add: rank mismatch on the Vulkan backend".to_string().into(),
            ));
        }
        let (dst_start, dst_len) = match l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    "index_add: non-contiguous self tensors not supported on the Vulkan backend"
                        .to_string()
                        .into(),
                ))
            }
        };
        let (src_start, src_len) = match src_l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    "index_add: non-contiguous sources not supported on the Vulkan backend"
                        .to_string()
                        .into(),
                ))
            }
        };
        let (ids_start, ids_len) = match ids_l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    "index_add: non-contiguous index tensors not supported on the Vulkan backend"
                        .to_string()
                        .into(),
                ))
            }
        };
        if dst_len != l.dims().iter().product::<usize>()
            || src_len != src_l.dims().iter().product::<usize>()
            || ids_len != ids_l.dims().iter().product::<usize>()
        {
            return Err(Error::Vulkan(
                "index_add: size mismatch on the Vulkan backend".to_string().into(),
            ));
        }
        // Drain any deferred encodes before reading the buffer contents.
        self.device.synchronize()?;
        let v1: Vec<f32> = match &self.buffer {
            VulkanStorageBuffer::F32(b) => Self::copy_to_host(&self.device, b)?,
            _ => unreachable!("dtype checked above"),
        };
        let src_v: Vec<f32> = match &src.buffer {
            VulkanStorageBuffer::F32(b) => Self::copy_to_host(&self.device, b)?,
            _ => unreachable!("dtype checked above"),
        };
        let ids_cpu = match &ids.buffer {
            VulkanStorageBuffer::U32(b) => {
                let v: Vec<u32> = Self::copy_to_host(&self.device, b)?;
                CpuStorage::U32(v)
            }
            VulkanStorageBuffer::I64(b) => {
                let v: Vec<i64> = Self::copy_to_host(&self.device, b)?;
                CpuStorage::I64(v)
            }
            VulkanStorageBuffer::U8(b) => {
                let v: Vec<u8> = Self::copy_to_host(&self.device, b)?;
                CpuStorage::U8(v)
            }
            _ => {
                return Err(Error::Vulkan(
                    "index_add: index dtype not supported on the Vulkan backend".to_string().into(),
                ))
            }
        };
        let v1_cpu = CpuStorage::F32(v1);
        let src_cpu = CpuStorage::F32(src_v);
        let l_c = Layout::contiguous_with_offset(l.dims().to_vec(), dst_start);
        let s_c = Layout::contiguous_with_offset(src_l.dims().to_vec(), src_start);
        let i_c = Layout::contiguous_with_offset(ids_l.dims().to_vec(), ids_start);
        let out = v1_cpu.index_add(&l_c, &ids_cpu, &i_c, &src_cpu, &s_c, dim)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        if self.dtype != DType::F32 || rhs.dtype != DType::F32 {
            return Err(Error::Vulkan(
                "matmul: only F32 supported on the Vulkan backend".to_string().into(),
            ));
        }
        let (bsz, m, n, k) = bmnk;
        if m == 0 || n == 0 || k == 0 || bsz == 0 {
            return VulkanStorage::new(&self.device, 0, DType::F32);
        }
        let (lhs_start, lhs_len) = match lhs_l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    "matmul: non-contiguous lhs not supported on the Vulkan backend".to_string().into(),
                ))
            }
        };
        if lhs_len != bsz * m * k {
            return Err(Error::Vulkan(
                "matmul: lhs size mismatch on the Vulkan backend".to_string().into(),
            ));
        }
        let rank = rhs_l.dims().len();
        if rank < 2 {
            return Err(Error::Vulkan(
                "matmul: rank < 2 not supported on the Vulkan backend".to_string().into(),
            ));
        }
        let rs = rhs_l.stride();
        // Accept a contiguous (b,k,n) rhs or a contiguous (b,n,k) rhs viewed
        // as (b,k,n) (the transposed case). Anything else is rejected.
        let standard = rs[rank - 1] == 1
            && rs[rank - 2] == n
            && (rank == 2 || rs[rank - 3] == k * n);
        let transposed = rs[rank - 2] == 1
            && rs[rank - 1] == k
            && (rank == 2 || rs[rank - 3] == n * k);
        if !standard && !transposed {
            return Err(Error::Vulkan(
                "matmul: non-contiguous rhs not supported on the Vulkan backend".to_string().into(),
            ));
        }
        let rhs_transposed = transposed && !standard;
        let rhs_start = rhs_l.start_offset();
        let rhs_len = bsz * n * k;
        let input = match &self.buffer {
            VulkanStorageBuffer::F32(b) => b.clone().slice(lhs_start as u64..(lhs_start + lhs_len) as u64),
            _ => unreachable!("dtype checked above"),
        };
        let rhs_buf = match &rhs.buffer {
            VulkanStorageBuffer::F32(b) => b.clone().slice(rhs_start as u64..(rhs_start + rhs_len) as u64),
            _ => unreachable!("dtype checked above"),
        };
        let out = VulkanStorage::new(&self.device, bsz * m * n, DType::F32)?;
        let out_buf = match &out.buffer {
            VulkanStorageBuffer::F32(b) => b.clone(),
            _ => unreachable!(),
        };
        let kernels = self.device.kernels();
        let input = input.clone();
        let rhs_buf = rhs_buf.clone();
        let out_buf = out_buf.clone();
        self.device.execute(move |cbb| {
            candle_vulkan_kernels::call_gemm_f32(
                cbb, &kernels, &input, &rhs_buf, &out_buf, bsz, m, n, k, rhs_transposed,
            )
            .map_err(|e| e.to_string())
        })?;
        Ok(out)
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        if self.dtype != DType::F32 || dst.dtype != DType::F32 {
            return Err(Error::Vulkan(
                "copy_strided_src: only F32 supported on the Vulkan backend".to_string().into(),
            ));
        }
        let src_buf = match &self.buffer {
            VulkanStorageBuffer::F32(b) => b.clone(),
            _ => unreachable!("dtype checked above"),
        };
        let dst_buf = match &dst.buffer {
            VulkanStorageBuffer::F32(b) => b.clone(),
            _ => unreachable!("dtype checked above"),
        };
        match src_l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => {
                if len == 0 {
                    return Ok(());
                }
                let src_sub = src_buf.slice(start_offset as u64..(start_offset + len) as u64);
                let dst_sub = dst_buf
                    .slice(dst_offset as u64..(dst_offset + len) as u64);
                self.device.execute(move |cbb| {
                    cbb.copy_buffer(CopyBufferInfo::new(src_sub, dst_sub))
                        .map_err(|e| e.to_string())?;
                    Ok(())
                })?;
            }
            crate::StridedBlocks::UniformBlocks {
                start_offset,
                block_len,
                count,
                src_stride,
            } => {
                let total = count * block_len;
                if total == 0 {
                    return Ok(());
                }
                let kernels = self.device.kernels();
                let params_buf = Buffer::from_iter(
                    self.device.mem_alloc(),
                    &BufferCreateInfo {
                        usage: BufferUsage::STORAGE_BUFFER,
                        ..Default::default()
                    },
                    &AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_HOST
                            | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                        ..Default::default()
                    },
                    vec![
                        count as f32,
                        block_len as f32,
                        src_stride as f32,
                        block_len as f32,
                        start_offset as f32,
                        dst_offset as f32,
                    ],
                )
                .map_err(|e| Error::Vulkan(e.to_string().into()))?;
                let params: Subbuffer<[f32]> = params_buf;
                self.device.execute(move |cbb| {
                    candle_vulkan_kernels::call_copy2d_slang_f32(
                        cbb,
                        &kernels,
                        &src_buf,
                        &dst_buf,
                        &params,
                        total,
                    )
                    .map_err(|e| e.to_string())
                })?;
            }
            crate::StridedBlocks::MultipleBlocks {
                block_start_index,
                block_len,
            } => {
                let n = block_start_index.len();
                let total = n * block_len;
                if total == 0 {
                    return Ok(());
                }
                let indices: Vec<u32> = block_start_index.map(|s| s as u32).collect();
                let idx_buf = Buffer::from_iter(
                    self.device.mem_alloc(),
                    &BufferCreateInfo {
                        usage: BufferUsage::STORAGE_BUFFER,
                        ..Default::default()
                    },
                    &AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_HOST
                            | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                        ..Default::default()
                    },
                    indices,
                )
                .map_err(|e| Error::Vulkan(e.to_string().into()))?;
                let idx: Subbuffer<[u32]> = idx_buf;
                let kernels = self.device.kernels();
                let params_buf = Buffer::from_iter(
                    self.device.mem_alloc(),
                    &BufferCreateInfo {
                        usage: BufferUsage::STORAGE_BUFFER,
                        ..Default::default()
                    },
                    &AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_HOST
                            | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                        ..Default::default()
                    },
                    vec![total as f32, block_len as f32, dst_offset as f32],
                )
                .map_err(|e| Error::Vulkan(e.to_string().into()))?;
                let params: Subbuffer<[f32]> = params_buf;
                self.device.execute(move |cbb| {
                    candle_vulkan_kernels::call_gather_idx_slang_f32(
                        cbb,
                        &kernels,
                        candle_vulkan_kernels::KernelName::GatherIdxF32,
                        &src_buf,
                        &dst_buf,
                        &idx,
                        &params,
                        total,
                    )
                    .map_err(|e| e.to_string())
                })?;
            }
        }
        Ok(())
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_stride1: usize,
        dst_stride1: usize,
        src_offset: usize,
        dst_offset: usize,
    ) -> Result<()> {
        if self.dtype != DType::F32 || dst.dtype != DType::F32 {
            return Err(Error::Vulkan(
                "copy2d: only F32 supported on the Vulkan backend".to_string().into(),
            ));
        }
        let total = d1 * d2;
        if total == 0 {
            return Ok(());
        }
        let src_buf = match &self.buffer {
            VulkanStorageBuffer::F32(b) => b.clone(),
            _ => unreachable!("dtype checked above"),
        };
        let dst_buf = match &dst.buffer {
            VulkanStorageBuffer::F32(b) => b.clone(),
            _ => unreachable!("dtype checked above"),
        };
        let kernels = self.device.kernels();
        let params_buf = Buffer::from_iter(
            self.device.mem_alloc(),
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            vec![
                d1 as f32,
                d2 as f32,
                src_stride1 as f32,
                dst_stride1 as f32,
                src_offset as f32,
                dst_offset as f32,
            ],
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let params: Subbuffer<[f32]> = params_buf;
        self.device.execute(move |cbb| {
            candle_vulkan_kernels::call_copy2d_slang_f32(
                cbb,
                &kernels,
                &src_buf,
                &dst_buf,
                &params,
                total,
            )
            .map_err(|e| e.to_string())
        })?;
        Ok(())
    }

    fn const_set(&mut self, scalar: crate::scalar::Scalar, l: &Layout) -> Result<()> {
        use crate::scalar::Scalar;
        let value = match (self.dtype, &scalar) {
            (DType::F32, Scalar::F32(v)) => *v,
            _ => {
                return Err(Error::Vulkan(
                    "const_set: unsupported scalar/dtype combination".to_string().into(),
                ))
            }
        };
        let (start, len) = match l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => (start_offset, len),
            _ => {
                return Err(Error::Vulkan(
                    "const_set: non-contiguous layout not yet supported on the Vulkan backend"
                        .to_string()
                        .into(),
                ))
            }
        };
        if len == 0 {
            return Ok(());
        }
        let buffer = match &self.buffer {
            VulkanStorageBuffer::F32(b) => b.clone(),
            _ => unreachable!("dtype checked above"),
        };
        let sub = buffer.slice(start as u64..(start + len) as u64);
        // Params: [0] = value, [1] = count.
        let params = vec![value, len as f32];
        let param_buf = Buffer::from_iter(
            self.device.mem_alloc(),
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            params,
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let param_sub: Subbuffer<[f32]> = param_buf;
        let kernels = self.device.kernels();
        self.device.execute(move |cbb| {
            candle_vulkan_kernels::call_const_set_f32(cbb, &kernels, &sub, &param_sub, len)
                .map_err(|e| e.to_string())
        })?;
        Ok(())
    }
}
