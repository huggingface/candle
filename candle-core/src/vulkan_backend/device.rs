//! The Vulkan device: a lazily initialized vulkano instance + compute device.

use std::sync::{Arc, Mutex};
use std::sync::atomic::AtomicU64;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::CommandBufferUsage;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::memory::StandardMemoryAllocator;
use vulkano::sync::GpuFuture;
use vulkano::VulkanLibrary;

use candle_vulkan_kernels::Kernels;

use crate::{CpuStorage, DType, Error, Result};

/// A device that can be used to run computations with the Vulkan backend.
#[derive(Clone, Debug)]
pub struct VulkanDevice {
    instance: Arc<Instance>,
    physical: Arc<PhysicalDevice>,
    device: Arc<Device>,
    queue: Arc<vulkano::device::Queue>,
    mem_alloc: Arc<StandardMemoryAllocator>,
    cbb_alloc: Arc<StandardCommandBufferAllocator>,
    dss_alloc: Arc<StandardDescriptorSetAllocator>,
    kernels: Kernels,
    pending: Mutex<Vec<Box<dyn GpuFuture>>>,
    seed: AtomicU64,
    _gpu_id: usize,
}

impl VulkanDevice {
    pub fn new(gpu_id: usize) -> Result<Self> {
        let library = unsafe { VulkanLibrary::new() }
            .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let instance = Instance::new(
            Arc::new(library),
            InstanceCreateInfo {
                application_name: Some("candle".into()),
                ..Default::default()
            },
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;

        let physical = instance
            .enumerate_physical_devices()
            .into_iter()
            .nth(gpu_id)
            .ok_or_else(|| Error::Vulkan(format!("no physical device {gpu_id}").into()))?;

        let queue_family = physical
            .queue_family_properties()
            .iter()
            .position(|q| q.queue_flags.contains(vulkano::device::QueueFlags::COMPUTE))
            .ok_or_else(|| Error::Vulkan("no compute queue family".to_string().into()))?;

        let (device, queues) = Device::new(
            physical.clone(),
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

        let mem_alloc = Arc::new(StandardMemoryAllocator::new(device.clone()));
        let cbb_alloc = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));
        let dss_alloc = Arc::new(StandardDescriptorSetAllocator::new(device.clone()));
        let kernels = Kernels::new(device.clone(), dss_alloc.clone());

        Ok(Self {
            instance,
            physical,
            device,
            queue,
            mem_alloc,
            cbb_alloc,
            dss_alloc,
            kernels,
            pending: Mutex::new(Vec::new()),
            seed: AtomicU64::new(0),
            _gpu_id: gpu_id,
        })
    }

    /// Uploads `data` into a device-local f32 storage buffer.
    pub fn upload_f32(&self, data: &[f32]) -> Result<Subbuffer<[f32]>> {
        Buffer::from_iter(
            self.mem_alloc.clone(),
            BufferCreateInfo::default()
                .set_usage(BufferUsage::STORAGE_BUFFER)
                .set_flags(vulkano::memory::BufferDeviceLocal::default()),
            AllocationCreateInfo::default(),
            data.iter().copied(),
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))
    }

    /// Allocates a zero-filled device-local f32 output buffer of `len` elements.
    pub fn new_f32_buffer(&self, len: usize) -> Result<Subbuffer<[f32]>> {
        self.upload_f32(&vec![0f32; len])
    }

    /// Records `encode` onto a new command buffer and submits it to the queue.
    /// Work completes in submission order; `synchronize` drains the pending
    /// futures (no per-operation fence wait, mirroring the metal backend).
    pub fn execute<F>(&self, encode: F) -> Result<()>
    where
        F: FnOnce(
            &mut vulkano::command_buffer::AutoCommandBufferBuilder<
                vulkano::command_buffer::PrimaryAutoCommandBuffer,
            >,
        ) -> Result<(), String>,
    {
        let mut cbb = vulkano::command_buffer::AutoCommandBufferBuilder::primary(
            self.cbb_alloc.clone(),
            self.queue
                .family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        encode(&mut cbb).map_err(Error::Vulkan)?;
        let cbb = cbb
            .build()
            .map_err(|e| Error::Vulkan(format!("command buffer build: {e}").into()))?;
        let future = vulkano::sync::NowFuture::new(self.device.clone())
            .then_execute(self.device.clone(), cbb)
            .map_err(|e| Error::Vulkan(format!("submit: {e}").into()))?;
        self.pending
            .lock()
            .expect("pending futures mutex")
            .push(Box::new(future));
        Ok(())
    }

    /// Waits for all pending GPU work to complete.
    pub fn synchronize(&self) -> Result<()> {
        let mut pending = self.pending.lock().expect("pending futures mutex");
        for f in pending.drain(..) {
            f.wait(None)
                .map_err(|e| Error::Vulkan(format!("wait: {e}").into()))?;
        }
        Ok(())
    }

    /// Copies a device f32 buffer back to the host.
    pub fn download_f32(&self, buffer: &Subbuffer<[f32]>) -> Result<Vec<f32>> {
        self.synchronize()?;
        Ok(buffer
            .read()
            .expect("buffer read guard")
            .as_slice()
            .to_vec())
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

/// A Vulkan tensor: a device-local f32 storage buffer plus its element count.
#[derive(Clone)]
pub struct VulkanStorage {
    buffer: Subbuffer<[f32]>,
    device: VulkanDevice,
    data_len: usize,
    dtype: DType,
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
