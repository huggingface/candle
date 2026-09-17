//! The Vulkan device: a lazily initialized vulkano instance + compute device.
use std::sync::atomic::{AtomicU64, AtomicUsize};
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
    StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
};
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::sync::fence::{Fence, FenceCreateInfo};
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
    /// Fence ring bounding in-flight submissions (see `execute`); shared
    /// by all `VulkanDevice` clones since they submit to the same queue.
    fence_ring: Arc<std::sync::Mutex<Vec<InFlight>>>,
    fence_next: Arc<AtomicUsize>,
    /// Encodes recorded by `execute`, deferred until `synchronize` records
    /// them all onto a single command buffer and submits it. Shared by all
    /// `VulkanDevice` clones since they submit to the same queue.
    pending: Arc<std::sync::Mutex<Vec<Encode>>>,
    _gpu_id: usize,
}

/// A deferred kernel record: `execute` appends these, `synchronize` plays
/// them back onto one command buffer in order.
type Encode = Box<dyn FnOnce(&mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>) -> std::result::Result<(), String> + Send>;

/// One slot of the fence ring: the command buffer of the last submission
/// that used this slot (kept alive until the fence signals, because the
/// command buffer allocator recycles dropped command buffers) and the
/// fence the submission signals.
struct InFlight {
    cbb: Option<Arc<PrimaryAutoCommandBuffer>>,
    fence: Arc<Fence>,
}

/// Number of in-flight submissions allowed before `execute` waits for the
/// GPU to catch up.
const FENCE_RING_SIZE: usize = 128;

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
            fence_ring: self.fence_ring.clone(),
            fence_next: self.fence_next.clone(),
            pending: self.pending.clone(),
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
        let library = VulkanLibrary::new().map_err(|e| Error::Vulkan(e.to_string().into()))?;
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
        let mut fence_ring = Vec::with_capacity(FENCE_RING_SIZE);
        for _ in 0..FENCE_RING_SIZE {
            let fence = Fence::new(device.clone(), FenceCreateInfo::default())
                .map_err(|e| Error::Vulkan(e.to_string().into()))?;
            fence_ring.push(InFlight { cbb: None, fence: Arc::new(fence) });
        }
        Ok(Self {
            instance,
            device,
            queue,
            mem_alloc,
            cbb_alloc,
            dss_alloc,
            kernels,
            seed: AtomicU64::new(0),
            fence_ring: Arc::new(std::sync::Mutex::new(fence_ring)),
            fence_next: Arc::new(AtomicUsize::new(0)),
            pending: Arc::new(std::sync::Mutex::new(Vec::new())),
            _gpu_id: gpu_id,
        })
    }

    /// Allocation for the compute buffers: host-mapped (so that
    /// `Buffer::from_iter` can initialize them and `read()` can read
    /// them back) but device-preferred, so on RDNA the allocation
    /// still lands in VRAM.
    fn storage_alloc_info() -> AllocationCreateInfo {
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::HOST_RANDOM_ACCESS
                | MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        }
    }

    /// Uploads `data` into a device f32 storage buffer.
    pub fn upload_f32(&self, data: &[f32]) -> Result<Subbuffer<[f32]>> {
        Buffer::from_iter(
            self.mem_alloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            Self::storage_alloc_info(),
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
            Self::storage_alloc_info(),
            std::iter::repeat_n(0f32, len),
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))
    }

    /// Defers `encode` onto the pending batch. All pending encodes are
    /// recorded onto a single command buffer and submitted by the next
    /// `synchronize()`, which also drains the GPU. Callers that read back
    /// results must call `synchronize()` first (the readback paths do).
    /// Batching keeps the per-operation submission cost (command buffer +
    /// descriptor set allocation, `vkQueueSubmit`) off the hot path: a
    /// decode step is one submission instead of one per op.
    pub fn execute<F>(&self, encode: F) -> Result<()>
    where
        F: FnOnce(
            &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        ) -> std::result::Result<(), String> + Send + 'static,
    {
        self.pending.lock().unwrap().push(Box::new(encode));
        Ok(())
    }

    /// Records all pending encodes onto a single command buffer, submits it
    /// (bounded by the fence ring, which also keeps the command buffer and
    /// the buffers it uses alive until the work completes, since the
    /// allocators recycle dropped resources), and waits for completion.
    pub fn synchronize(&self) -> Result<()> {
        let encodes = std::mem::take(&mut *self.pending.lock().unwrap());
        if std::env::var("CANDLE_VULKAN_TRACE").is_ok() && !encodes.is_empty() {
            let counts = candle_vulkan_kernels::trace_counts();
            let total: u64 = counts.iter().map(|(_, c)| c).sum();
            eprintln!(
                "vulkan: drain {} dispatches ({})",
                encodes.len(),
                counts
                    .iter()
                    .map(|((src, name), c)| format!("{}::{}={}", src.as_ref(), name.as_ref(), c))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            let _ = total;
        }
        if !encodes.is_empty() {
            let n_encodes = encodes.len();
            let t0 = std::time::Instant::now();
            let mut cbb = AutoCommandBufferBuilder::primary(
                self.cbb_alloc.clone(),
                self.queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .map_err(|e| Error::Vulkan(e.to_string().into()))?;
            for encode in encodes {
                encode(&mut cbb).map_err(|e| Error::Vulkan(e.into()))?;
            }
            let t_build = t0.elapsed();
            let cbb = cbb.build().map_err(|e| {
                Error::Vulkan(format!("command buffer build: {e}").into())
            })?;
            let idx = self
                .fence_next
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                % FENCE_RING_SIZE;
            let cbb_submit = cbb.clone();
            let fence = {
                let mut ring = self.fence_ring.lock().unwrap();
                let slot = ring.get_mut(idx).expect("fence ring index in range");
                if slot.cbb.is_some() {
                    slot.fence
                        .wait(None)
                        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
                }
                unsafe { slot.fence.reset() }
                    .map_err(|e| Error::Vulkan(e.to_string().into()))?;
                slot.cbb = Some(cbb);
                slot.fence.clone()
            };
            self.queue
                .with(|mut q| unsafe {
                    q.submit(
                        &[SubmitInfo {
                            wait_semaphores: Vec::new(),
                            command_buffers: vec![CommandBufferSubmitInfo::new(cbb_submit)],
                            signal_semaphores: Vec::new(),
                            ..Default::default()
                        }],
                        Some(&fence),
                    )
                })
                .map_err(|e| Error::Vulkan(e.to_string().into()))?;
            let t_wait = std::time::Instant::now();
            self.queue
                .with(|mut q| q.wait_idle())
                .map_err(|e| Error::Vulkan(e.to_string().into()))?;
            if std::env::var("CANDLE_VULKAN_PROFILE").is_ok() {
                eprintln!(
                    "vulkan profile: {} encodes, encode+build {:?}, submit+wait {:?}",
                    n_encodes,
                    t_build,
                    t_wait.elapsed()
                );
            }
            return Ok(());
        }
        Ok(())
    }

    /// Copies a device f32 buffer back to the host (the buffers are
    /// allocated host-mapped, see `storage_alloc_info`).
    pub fn download_f32(&self, buffer: &Subbuffer<[f32]>) -> Result<Vec<f32>> {
        self.synchronize()?;
        let guard = buffer
            .read()
            .map_err(|e| Error::Vulkan(format!("buffer read: {e}").into()))?;
        Ok((*guard).to_vec())
    }

    /// Uploads `data` into a device u32 storage buffer.
    pub fn upload_u32(&self, data: &[u32]) -> Result<Subbuffer<[u32]>> {
        Buffer::from_iter(
            self.mem_alloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            Self::storage_alloc_info(),
            data.iter().copied(),
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))
    }

    /// Copies a device u32 buffer back to the host (the buffers are
    /// allocated host-mapped, see `storage_alloc_info`).
    pub fn download_u32(&self, buffer: &Subbuffer<[u32]>) -> Result<Vec<u32>> {
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

    pub fn kernels(&self) -> Arc<Kernels> {
        self.kernels.clone()
    }
}

/// The underlying buffer of a `VulkanStorage`: f32 or u32 elements.
#[derive(Clone)]
pub enum VBuf {
    F32(Subbuffer<[f32]>),
    U32(Subbuffer<[u32]>),
}

impl VBuf {
    pub fn as_f32(&self) -> Option<&Subbuffer<[f32]>> {
        match self {
            VBuf::F32(b) => Some(b),
            VBuf::U32(_) => None,
        }
    }

    pub fn as_u32(&self) -> Option<&Subbuffer<[u32]>> {
        match self {
            VBuf::F32(_) => None,
            VBuf::U32(b) => Some(b),
        }
    }
}

/// A Vulkan tensor: a device f32/u32 storage buffer plus its element count.
#[derive(Clone)]
pub struct VulkanStorage {
    pub(super) buffer: VBuf,
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
    pub fn new(buffer: VBuf, device: &VulkanDevice, data_len: usize, dtype: DType) -> Self {
        Self {
            buffer,
            device: device.clone(),
            data_len,
            dtype,
        }
    }

    pub fn as_f32(&self) -> Option<&Subbuffer<[f32]>> {
        self.buffer.as_f32()
    }

    pub fn as_u32(&self) -> Option<&Subbuffer<[u32]>> {
        self.buffer.as_u32()
    }

    pub fn len(&self) -> usize {
        self.data_len
    }

    pub fn is_empty(&self) -> bool {
        self.data_len == 0
    }
}
