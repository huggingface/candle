//! The Vulkan device: a lazily initialized vulkano instance + compute device.
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferSubmitInfo, CommandBufferUsage,
    PrimaryAutoCommandBuffer, SubmitInfo,
};
use vulkano::descriptor_set::allocator::{
    StandardDescriptorSetAllocator,
    StandardDescriptorSetAllocatorCreateInfo,
};
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator};
use vulkano::VulkanLibrary;

use candle_vulkan_kernels::Kernels;

use crate::{DType, Error, Result};

/// A device that can be used to run computations with the Vulkan backend.
pub struct VulkanDevice {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<vulkano::device::Queue>,
    mem_alloc: Arc<StandardMemoryAllocator>,
    cbb_alloc: Arc<StandardCommandBufferAllocator>,
    dss_alloc: Arc<StandardDescriptorSetAllocator>,
    kernels: Arc<Kernels>,
    seed: AtomicU64,
    _gpu_id: usize,
}

impl Clone for VulkanDevice {
    fn clone(&self) -> Self {
        Self {
            instance: self.instance.clone(),
            device: self.device.clone(),
            queue: self.queue.clone(),
            mem_alloc: self.mem_alloc.clone(),
            cbb_alloc: self.cbb_alloc.clone(),
            dss_alloc: self.dss_alloc.clone(),
            kernels: self.kernels.clone(),
            seed: std::sync::atomic::AtomicU64::new(
                self.seed.load(std::sync::atomic::Ordering::Relaxed),
            ),
            _gpu_id: self._gpu_id,
        }
    }
}

impl std::fmt::Debug for VulkanDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanDevice")
            .field("gpu_id", &self._gpu_id)
            .finish()
    }
}

impl VulkanDevice {
    pub fn new(gpu_id: usize) -> Result<Self> {
        let library = unsafe { VulkanLibrary::new() }
            .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                application_name: Some("candle".into()),
                ..Default::default()
            },
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let mut devices = instance
            .enumerate_physical_devices()
            .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let physical = devices
            .nth(gpu_id)
            .ok_or_else(|| Error::Vulkan(format!("no physical device {gpu_id}").into()))?;
        let queue_family = physical
            .queue_family_properties()
            .iter()
            .position(|q| q.queue_flags.contains(QueueFlags::COMPUTE))
            .ok_or_else(|| Error::Vulkan("no compute queue family".to_string().into()))?;
        let (device, queues) = Device::new(
            physical,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    flags: Default::default(),
                    queue_family_index: queue_family as u32,
                    queues: vec![1.0],
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let queue = queues
            .into_iter()
            .next()
            .ok_or_else(|| Error::Vulkan("no queue returned".to_string().into()))?;
        let mem_alloc = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let cbb_alloc = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));
        let dss_alloc = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo::default(),
        ));
        let kernels = Arc::new(Kernels::new(device.clone(), dss_alloc.clone()));
        Ok(Self {
            instance,
            device,
            queue,
            mem_alloc,
            cbb_alloc,
            dss_alloc,
            kernels,
            seed: AtomicU64::new(0),
            _gpu_id: gpu_id,
        })
    }

    /// Uploads `data` into a device f32 storage buffer.
    pub fn upload_f32(&self, data: &[f32]) -> Result<Subbuffer<[f32]>> {
        Buffer::from_iter(
            self.mem_alloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
            data.iter().copied(),
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))
    }

    /// Allocates a zero-filled f32 buffer of `len` elements.
    pub fn new_f32_buffer(&self, len: usize) -> Result<Subbuffer<[f32]>> {
        Buffer::from_iter(
            self.mem_alloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
            std::iter::repeat(0f32).take(len),
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))
    }

    /// Records `encode` onto a new command buffer, submits it and waits for
    /// completion. v0 syncs per submission; a fence ring replaces this once
    /// the backend is functional.
    pub fn execute<F>(&self, encode: F) -> Result<()>
    where
        F: FnOnce(
            &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        ) -> std::result::Result<(), String>,
    {
        let mut cbb = AutoCommandBufferBuilder::primary(
            self.cbb_alloc.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        encode(&mut cbb).map_err(|e| Error::Vulkan(e.into()))?;
        let cbb = cbb
            .build()
            .map_err(|e| Error::Vulkan(format!("command buffer build: {e}").into()))?;
        self.queue
            .with(|mut q| {
                unsafe {
                    q.submit(
                        &[SubmitInfo {
                            wait_semaphores: Vec::new(),
                            command_buffers: vec![CommandBufferSubmitInfo::new(cbb)],
                            signal_semaphores: Vec::new(),
                            ..Default::default()
                        }],
                        None,
                    )
                }
            })
            .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        self.queue
            .with(|mut q| q.wait_idle())
            .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        Ok(())
    }

    /// Waits for all pending GPU work to complete.
    pub fn synchronize(&self) -> Result<()> {
        self.queue
            .with(|mut q| q.wait_idle())
            .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        Ok(())
    }

    /// Copies a device f32 buffer back to the host.
    pub fn download_f32(&self, buffer: &Subbuffer<[f32]>) -> Result<Vec<f32>> {
        self.synchronize()?;
        let guard = buffer
            .read()
            .map_err(|e| Error::Vulkan(format!("buffer read: {e}").into()))?;
        Ok((*guard).to_vec())
    }

    pub fn seed_atomic(&self) -> &AtomicU64 {
        &self.seed
    }

    pub fn gpu_id(&self) -> usize {
        self._gpu_id
    }

    pub fn kernels(&self) -> &Kernels {
        &self.kernels
    }
}

/// A Vulkan tensor: a device f32 storage buffer plus its element count.
#[derive(Clone)]
pub struct VulkanStorage {
    pub(super) buffer: Subbuffer<[f32]>,
    pub(super) device: VulkanDevice,
    pub(super) data_len: usize,
    pub(super) dtype: DType,
}

impl std::fmt::Debug for VulkanStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanStorage")
            .field("data_len", &self.data_len)
            .field("dtype", &self.dtype)
            .finish()
    }
}

impl VulkanStorage {
    pub fn new(
        buffer: Subbuffer<[f32]>,
        device: &VulkanDevice,
        data_len: usize,
        dtype: DType,
    ) -> Self {
        Self {
            buffer,
            device: device.clone(),
            data_len,
            dtype,
        }
    }

    pub fn buffer(&self) -> &Subbuffer<[f32]> {
        &self.buffer
    }

    pub fn device(&self) -> &VulkanDevice {
        &self.device
    }

    pub fn len(&self) -> usize {
        self.data_len
    }

    pub fn is_empty(&self) -> bool {
        self.data_len == 0
    }
}
