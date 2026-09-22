//! The Vulkan device: a lazily initialized vulkano instance + compute device.
use std::sync::atomic::{AtomicU64, AtomicUsize};
use std::sync::Arc;

use half::f16;

use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferSubmitInfo, CommandBufferUsage, CopyBufferInfo,
    PrimaryAutoCommandBuffer, SubmitInfo,
};
use vulkano::descriptor_set::allocator::{
    StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
};
use vulkano::device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::sync::fence::{Fence, FenceCreateInfo};
use vulkano::sync::GpuFuture;
use vulkano::VulkanLibrary;

use candle_vulkan_kernels::Kernels;

use crate::{Error, Result};

/// A device that can be used to run computations with the Vulkan backend.
pub struct VulkanDevice {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
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
    /// Buffers allocated since the last `synchronize()`. The closures
    /// recorded by `execute` own temporary buffers (e.g. dequant scratch)
    /// and are consumed when the batch is recorded, so without this list
    /// those buffers would be freed before the GPU executes the command
    /// buffer that uses them. `synchronize` clears the list after the
    /// batch's fence signals, at which point the GPU is done with every
    /// buffer of the batch and the allocator may recycle the memory.
    keepalive_f32: Arc<std::sync::Mutex<Vec<Subbuffer<[f32]>>>>,
    keepalive_u32: Arc<std::sync::Mutex<Vec<Subbuffer<[u32]>>>>,
    _gpu_id: usize,
}

/// A deferred kernel record: `execute` appends these, `synchronize` plays
/// them back onto one command buffer in order.
type Encode = Box<
    dyn FnOnce(
            &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        ) -> std::result::Result<(), String>
        + Send,
>;

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
            keepalive_f32: self.keepalive_f32.clone(),
            keepalive_u32: self.keepalive_u32.clone(),
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
            fence_ring.push(InFlight {
                cbb: None,
                fence: Arc::new(fence),
            });
        }
        tracing::debug!("Initializing vulkan device '{gpu_id}'");
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
            keepalive_f32: Arc::new(std::sync::Mutex::new(Vec::new())),
            keepalive_u32: Arc::new(std::sync::Mutex::new(Vec::new())),
            pending: Arc::new(std::sync::Mutex::new(Vec::new())),
            _gpu_id: gpu_id,
        })
    }

    pub(crate) fn device(&self) -> &Arc<Device> {
        &self.device
    }

    pub(crate) fn mem_alloc(&self) -> &Arc<StandardMemoryAllocator> {
        &self.mem_alloc
    }

    pub(crate) fn cbb_alloc(&self) -> &Arc<StandardCommandBufferAllocator> {
        &self.cbb_alloc
    }

    pub(crate) fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }

    /// Allocation for the compute buffers: device-local (VRAM).
    /// Host-mapped allocations on this platform fall back to system
    /// RAM (the BAR aperture is small), which caps decode at the PCIe
    /// bandwidth; weights and activations live in VRAM instead, with
    /// one-shot staging copies for uploads and readback.
    fn storage_alloc_info() -> AllocationCreateInfo {
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        }
    }

    /// Host-visible staging memory for uploads and readback.
    fn staging_alloc_info() -> AllocationCreateInfo {
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        }
    }

    pub fn copy_data_to_device<T: Default + Clone>(&self, data: &T) -> Result<Subbuffer<T>>
    where
        T: BufferContents,
    {
        // Create the staging buffer on the host with the data that needs to be copied to VRAM.
        let source_buffer = Buffer::from_data(
            self.mem_alloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            data.clone(),
        )
        .map_err(|error| {
            Error::Vulkan(format!("Unable to create source buffer: {error:?}").into())
        })?;

        // Create the VRAM buffer.
        let vram_buffer = Buffer::from_data(
            self.mem_alloc.clone(),
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
            T::default(),
        )
        .map_err(|error| {
            Error::Vulkan(format!("Unable to create vram buffer: {error:?}").into())
        })?;

        // Build the copy command
        let mut builder = AutoCommandBufferBuilder::primary(
            self.cbb_alloc.clone(),
            self.queue.queue_family_index(),
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
        let future = vulkano::sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .map_err(|error| {
                Error::Vulkan(
                    format!("Unable to execute and flush command buffer: {error:?}").into(),
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

    pub fn copy_data_from_device<T: Default + Clone>(&self, vram_buffer: &Subbuffer<T>) -> Result<T>
    where
        T: BufferContents,
    {
        // Create the buffer on the host that will receive the VRAM buffer data.
        let destination_buffer = Buffer::from_data(
            self.mem_alloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            T::default(),
        )
        .map_err(|error| {
            Error::Vulkan(format!("Unable to create destination buffer: {error:?}").into())
        })?;

        // Build the copy command
        let mut builder = AutoCommandBufferBuilder::primary(
            self.cbb_alloc.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(|error| {
            Error::Vulkan(format!("Unable to create command buffer builder: {error:?}").into())
        })?;
        builder
            .copy_buffer(CopyBufferInfo::buffers(
                vram_buffer.clone(),
                destination_buffer.clone(),
            ))
            .map_err(|error| {
                Error::Vulkan(format!("Unable to set copy_buffer command: {error:?}").into())
            })?;
        let command_buffer = builder.build().map_err(|error| {
            Error::Vulkan(format!("Unable to create command buffer: {error:?}").into())
        })?;

        // Execute and flush the command buffer
        let future = vulkano::sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .map_err(|error| {
                Error::Vulkan(
                    format!("Unable to execute and flush command buffer: {error:?}").into(),
                )
            })?;

        // Block the CPU thread until the GPU finishes copying the data
        future.wait(None).map_err(|error| {
            Error::Vulkan(
                format!("Failure occurred waiting for data to be uploaded: {error:?}").into(),
            )
        })?;
        let data = destination_buffer.read().map_err(|error| {
            Error::Vulkan(
                format!("Unable to acquire read lock to data in destination buffer: {error:?}")
                    .into(),
            )
        })?;
        Ok((*data).clone())
    }

    pub fn supports_bf16(&self) -> bool {
        let supports_bf16 = self
            .device
            .physical_device()
            .extension_properties()
            .iter()
            .any(|property| property.extension_name.eq("VK_KHR_shader_bfloat16"));
        // TODO: Need to support bfloat16 eventually. For now, just log if the GPU supports it or not.
        if supports_bf16 {
            tracing::debug!("Vulkan device {} has bfloat16 support.", self._gpu_id);
        } else {
            tracing::debug!(
                "Vulkan device {} does not have bfloat16 support.",
                self._gpu_id
            );
        }
        false
    }

    /// Uploads `data` into a device f32 storage buffer in VRAM. A
    /// host-visible staging copy is recorded onto the pending batch
    /// and runs at the next `synchronize`.
    pub fn upload_f32(&self, data: &[f32]) -> Result<Subbuffer<[f32]>> {
        let buffer = Buffer::new_slice(
            self.mem_alloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            Self::storage_alloc_info(),
            data.len()
                .try_into()
                .map_err(|_| Error::Vulkan("f32 buffer length".to_string().into()))?,
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let staging = Buffer::from_iter(
            self.mem_alloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            Self::staging_alloc_info(),
            data.iter().copied(),
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let dst = buffer.clone();
        self.execute(move |cbb| {
            cbb.copy_buffer(CopyBufferInfo::buffers(staging, dst))
                .map_err(|e| e.to_string())?;
            Ok(())
        })?;
        Ok(buffer)
    }

    /// Allocates a zero-filled f32 buffer of `len` elements in VRAM
    /// (the fill is deferred onto the pending batch).
    pub fn new_f32_buffer(&self, len: usize) -> Result<Subbuffer<[f32]>> {
        let buffer = Buffer::new_slice(
            self.mem_alloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            Self::storage_alloc_info(),
            len.try_into()
                .map_err(|_| Error::Vulkan("f32 buffer length".to_string().into()))?,
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let dst: Subbuffer<[u32]> = buffer.clone().reinterpret();
        self.execute(move |cbb| {
            cbb.fill_buffer(dst, 0).map_err(|e| e.to_string())?;
            Ok(())
        })?;
        self.keepalive_f32.lock().unwrap().push(buffer.clone());
        Ok(buffer)
    }

    /// Allocate a zero-filled u32 storage buffer (the fill is deferred
    /// onto the pending batch).
    pub fn new_u32_buffer(&self, len: usize) -> Result<Subbuffer<[u32]>> {
        let buffer = Buffer::new_slice(
            self.mem_alloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            Self::storage_alloc_info(),
            len.try_into()
                .map_err(|_| Error::Vulkan("u32 buffer length".to_string().into()))?,
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let dst: Subbuffer<[u32]> = buffer.clone();
        self.execute(move |cbb| {
            cbb.fill_buffer(dst, 0).map_err(|e| e.to_string())?;
            Ok(())
        })?;
        self.keepalive_u32.lock().unwrap().push(buffer.clone());
        Ok(buffer)
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
            ) -> std::result::Result<(), String>
            + Send
            + 'static,
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
            let cbb = cbb
                .build()
                .map_err(|e| Error::Vulkan(format!("command buffer build: {e}").into()))?;
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
                unsafe { slot.fence.reset() }.map_err(|e| Error::Vulkan(e.to_string().into()))?;
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
            fence
                .wait(Some(std::time::Duration::from_secs(60)))
                .map_err(|e| Error::Vulkan(e.to_string().into()))?;
            // The batch is done on the GPU: release the keep-alive refs so
            // the allocator can recycle the memory for the next batch.
            self.keepalive_f32.lock().unwrap().clear();
            self.keepalive_u32.lock().unwrap().clear();
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

    /// Copies a device f32 buffer (VRAM) back to the host via a
    /// host-visible staging buffer, then returns the data.
    pub fn download_f32(&self, buffer: &Subbuffer<[f32]>) -> Result<Vec<f32>> {
        let staging = Buffer::new_slice(
            self.mem_alloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            Self::staging_alloc_info(),
            buffer
                .len()
                .try_into()
                .map_err(|_| Error::Vulkan("f32 buffer length".to_string().into()))?,
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let src = buffer.clone();
        let dst = staging.clone();
        self.execute(move |cbb| {
            cbb.copy_buffer(CopyBufferInfo::buffers(src, dst))
                .map_err(|e| e.to_string())?;
            Ok(())
        })?;
        self.synchronize()?;
        let guard = staging
            .read()
            .map_err(|e| Error::Vulkan(format!("buffer read: {e}").into()))?;
        Ok((*guard).to_vec())
    }

    /// Uploads `data` into a device u8 storage buffer in VRAM. A
    /// host-visible staging copy is recorded onto the pending batch
    /// and runs at the next `synchronize`.
    pub fn upload_u8(&self, data: &[u8]) -> Result<Subbuffer<[u8]>> {
        let buffer = Buffer::new_slice(
            self.mem_alloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            Self::storage_alloc_info(),
            data.len()
                .try_into()
                .map_err(|_| Error::Vulkan("u8 buffer length".to_string().into()))?,
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let staging = Buffer::from_iter(
            self.mem_alloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            Self::staging_alloc_info(),
            data.iter().copied(),
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let dst = buffer.clone();
        self.execute(move |cbb| {
            cbb.copy_buffer(CopyBufferInfo::buffers(staging, dst))
                .map_err(|e| e.to_string())?;
            Ok(())
        })?;
        Ok(buffer)
    }
    /// Downloads a device u8 storage buffer to the host.
    pub fn download_u8(&self, buffer: &Subbuffer<[u8]>) -> Result<Vec<u8>> {
        let staging = Buffer::new_slice(
            self.mem_alloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            Self::staging_alloc_info(),
            buffer
                .len()
                .try_into()
                .map_err(|_| Error::Vulkan("u8 buffer length".to_string().into()))?,
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let src = buffer.clone();
        let dst = staging.clone();
        self.execute(move |cbb| {
            cbb.copy_buffer(CopyBufferInfo::buffers(src, dst))
                .map_err(|e| e.to_string())?;
            Ok(())
        })?;
        self.synchronize()?;
        let guard = staging
            .read()
            .map_err(|e| Error::Vulkan(format!("buffer read: {e}").into()))?;
        Ok((*guard).to_vec())
    }
    /// Uploads `data` into a device u32 storage buffer in VRAM.
    pub fn upload_u32(&self, data: &[u32]) -> Result<Subbuffer<[u32]>> {
        let buffer = Buffer::new_slice(
            self.mem_alloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            Self::storage_alloc_info(),
            data.len()
                .try_into()
                .map_err(|_| Error::Vulkan("u32 buffer length".to_string().into()))?,
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let staging = Buffer::from_iter(
            self.mem_alloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            Self::staging_alloc_info(),
            data.iter().copied(),
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let dst = buffer.clone();
        self.execute(move |cbb| {
            cbb.copy_buffer(CopyBufferInfo::buffers(staging, dst))
                .map_err(|e| e.to_string())?;
            Ok(())
        })?;
        Ok(buffer)
    }

    /// Copies a device u32 buffer (VRAM) back to the host.
    pub fn download_u32(&self, buffer: &Subbuffer<[u32]>) -> Result<Vec<u32>> {
        let staging = Buffer::new_slice(
            self.mem_alloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            Self::staging_alloc_info(),
            buffer
                .len()
                .try_into()
                .map_err(|_| Error::Vulkan("u32 buffer length".to_string().into()))?,
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let src = buffer.clone();
        let dst = staging.clone();
        self.execute(move |cbb| {
            cbb.copy_buffer(CopyBufferInfo::buffers(src, dst))
                .map_err(|e| e.to_string())?;
            Ok(())
        })?;
        self.synchronize()?;
        let guard = staging
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

#[derive(Clone)]
pub enum VBuf {
    F16(Subbuffer<[f16]>),
    F32(Subbuffer<[f32]>),
    U32(Subbuffer<[u32]>),
}

impl VBuf {
    pub fn as_f32(&self) -> Option<&Subbuffer<[f32]>> {
        match self {
            VBuf::F16(_) => None,
            VBuf::F32(b) => Some(b),
            VBuf::U32(_) => None,
        }
    }

    pub fn as_u32(&self) -> Option<&Subbuffer<[u32]>> {
        match self {
            VBuf::F16(_) => None,
            VBuf::F32(_) => None,
            VBuf::U32(b) => Some(b),
        }
    }
}
