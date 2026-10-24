//! The Vulkan device: a lazily initialized vulkano instance + compute device.
use std::sync::atomic::{AtomicU64, AtomicUsize};
use std::sync::Arc;

use half::f16;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferSubmitInfo, CommandBufferUsage, CopyBufferInfo,
    PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract, SubmitInfo,
};
use vulkano::descriptor_set::allocator::{
    StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::{
    AllocationCreateInfo, GenericMemoryAllocatorCreateInfo, MemoryTypeFilter,
    StandardMemoryAllocator,
};
use vulkano::sync::fence::{Fence, FenceCreateInfo};
use vulkano::VulkanLibrary;

use candle_vulkan_kernels::Kernels;

use crate::backend::{BackendDevice, BackendStorage};
use crate::{DType, Error, Result};

/// A device that can be used to run computations with the Vulkan backend.
///
/// All state lives in one `DeviceInner` behind an `Arc`, so cloning the
/// device (one clone per tensor operation) is a single reference-count
/// bump instead of a dozen Arc/atomic copies.
#[derive(Clone)]
pub struct VulkanDevice {
    inner: Arc<DeviceInner>,
}

/// A count of live `VulkanStorage` instances holding one `Arc<Buffer>`:
/// shallow `Clone`s of the storage increment, `Drop`s decrement. The
/// count reaches zero when the last storage instance is gone, even while
/// pending encode closures still hold captured subbuffers of the same
/// buffer. The entry only exists while at least one holder is alive, so
/// the registry itself never needs to keep the buffer's Arc alive.
struct BufferInstance {
    count: std::sync::atomic::AtomicUsize,
}

/// A pooled buffer: its actual byte size and the buffer itself.
type PooledBuffer = (usize, Arc<Buffer>);

/// The free buffer pool: reusable device buffers keyed by the element
/// `DType` and a size class (the buffer's byte size rounded up to
/// `POOL_CLASS`). A take of `bytes` reuses the smallest pooled class
/// that fits (sliced down to `bytes`), so growing tensors find pooled
/// buffers instead of missing on exact-byte keys. `total_bytes` is the
/// running sum of the pooled buffers' actual sizes, kept on insert and
/// take so the cap checks stay O(1).
struct FreePool {
    /// (dtype, size class) -> pooled buffers
    map: std::collections::HashMap<(DType, usize), Vec<PooledBuffer>>,
    total_bytes: usize,
}

impl FreePool {
    fn new() -> Self {
        Self {
            map: std::collections::HashMap::new(),
            total_bytes: 0,
        }
    }
}

/// Shared state of a `VulkanDevice`.
struct DeviceInner {
    device: Arc<Device>,
    queue: Arc<Queue>,
    mem_alloc: Arc<StandardMemoryAllocator>,
    cbb_alloc: Arc<StandardCommandBufferAllocator>,
    kernels: Arc<Kernels>,
    seed: AtomicU64,
    /// Fence ring bounding in-flight submissions (see `execute`); shared
    /// by all `VulkanDevice` clones since they submit to the same queue.
    fence_ring: std::sync::Mutex<Vec<InFlight>>,
    fence_next: AtomicUsize,
    /// Encodes recorded by `execute`, deferred until `synchronize` records
    /// them all onto a single command buffer and submits it. Shared by all
    /// `VulkanDevice` clones since they submit to the same queue.
    pending: std::sync::Mutex<Vec<Encode>>,
    /// Buffers allocated since the last `synchronize()`. The closures
    /// recorded by `execute` own temporary buffers (e.g. dequant scratch)
    /// and are consumed when the batch is recorded, so without this list
    /// those buffers would be freed before the GPU executes the command
    /// buffer that uses them. `synchronize` clears the list after the
    /// batch's fence signals, at which point the GPU is done with every
    /// buffer of the batch and the allocator may recycle the memory.
    keepalive_f32: std::sync::Mutex<Vec<Subbuffer<[f32]>>>,
    keepalive_u32: std::sync::Mutex<Vec<Subbuffer<[u32]>>>,
    /// Buffers that can be handed out immediately, keyed by (dtype, size
    /// class): the GPU has been idle on them since the last
    /// `synchronize`. `Drop for VulkanStorage` returns eligible buffers
    /// here and `VulkanStorage::new` takes from here.
    pool_free: std::sync::Mutex<FreePool>,
    /// Buffers returned while a batch was pending; `synchronize`
    /// promotes them to `pool_free` after the batch's fence signals (the
    /// GPU is done with them then, like the keepalive lists).
    pool_pending: std::sync::Mutex<Vec<(DType, std::sync::Arc<Buffer>)>>,
    /// Live `VulkanStorage` instance counts per buffer (see
    /// `BufferInstance`), keyed by the buffer's address: an entry only
    /// exists while at least one live holder pins the allocation, so the
    /// address cannot be reused for another buffer while registered.
    buffer_instances:
        std::sync::Mutex<std::collections::HashMap<usize, std::sync::Arc<BufferInstance>>>,
    /// Reusable host-visible readback buffer for `copy_to_host`, grown on
    /// demand (never shrinks): a fresh allocation per readback would page
    /// in a new host mapping every time.
    readback: std::sync::Mutex<Option<Arc<Buffer>>>,
    /// Per-width atomic float add support (SPV_EXT_shader_atomic_float_add),
    /// as enabled at device creation: F32/F64 via VK_EXT_shader_atomic_float,
    /// F16 via VK_EXT_shader_atomic_float2.
    atomic_f32: bool,
    atomic_f64: bool,
    atomic_f16: bool,
    /// Native FloatE4M3 (FP8 E4M3) support, as enabled at device
    /// creation: the VK_EXT_shader_float8 device extension and the
    /// shaderFloat8 device feature both have to be reported by the
    /// physical device. When true, the `FloatE4M3` Slang type (the GPU
    /// storage type of `microfloat::f8e4m3`) can be used in shaders.
    supports_f8e4m3: bool,
    /// Native BFloat16 support, as enabled at device creation: the
    /// VK_KHR_shader_bfloat16 device extension and the shaderBFloat16Type
    /// device feature both have to be reported by the physical device.
    /// When true, the `BFloat16` Slang type (the GPU storage type of
    /// `half::bf16`) can be used in shaders.
    supports_bf16: bool,
    /// Cooperative-matrix support, as enabled at device creation: the
    /// VK_KHR_cooperative_matrix device extension plus the
    /// cooperativeMatrix device feature both have to be reported by the
    /// physical device. When true, the `linalg.CoopMat` Slang type (and
    /// the cooperative-matrix GEMM kernels) can be used in shaders.
    supports_coopmat: bool,
    /// The device's subgroup size; the compiled coop-mat GEMM shaders
    /// assume exactly one 64-thread subgroup per workgroup.
    subgroup_size: u32,
    /// Core `vulkanMemoryModel` feature enabled at device creation;
    /// the coop-mat GEMM shaders declare the VulkanMemoryModel
    /// SPIR-V capability, so it is required for them to load.
    vulkan_memory_model: bool,
    /// One-time probe results for the coop-mat GEMM pipelines: `None`
    /// until the first use, then the build outcome is cached for the
    /// lifetime of the device.
    coopmat_gemm_f16: std::sync::OnceLock<bool>,
    coopmat_gemm_bf16: std::sync::OnceLock<bool>,
    /// One-time probe results for the coop-mat affine (GEMM + bias)
    /// pipelines: `None` until the first use, then the build outcome
    /// is cached for the lifetime of the device.
    coopmat_affine_f16: std::sync::OnceLock<bool>,
    coopmat_affine_bf16: std::sync::OnceLock<bool>,
    gpu_id: usize,
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

/// Minimum size in bytes for a buffer to enter the pool: smaller
/// allocations are cheap and the pool bookkeeping is not worth it.
pub(crate) const POOL_MIN_BYTES: usize = 4096;
/// Maximum number of pooled buffers per (dtype, size class): decode
/// reuses a handful of recurring shapes, each with a few buffers live
/// at once.
const POOL_MAX_PER_SIZE: usize = 8;
/// Size-class granularity for pooled buffers: a buffer is pooled in the
/// class of its rounded-up byte size, and a take of `bytes` reuses the
/// smallest pooled class that fits, so growing tensors (e.g. per-token
/// attention intermediates) actually reuse pooled buffers instead of
/// missing on exact-byte keys.
const POOL_CLASS: usize = 64 * 1024;
/// Maximum number of distinct (dtype, size class) entries in the free
/// pool: caps the bookkeeping and the number of long-lived allocations.
const POOL_MAX_ENTRIES: usize = 512;
/// Maximum total bytes held by the free pool: a few percent of the
/// reference 32 GB device, comfortably above the recurring decode shapes
/// (the skip-never-evict policy keeps retained buffers in place).
const POOL_MAX_BYTES: usize = 2048 * 1024 * 1024;

impl std::fmt::Debug for VulkanDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanDevice")
            .field("gpu_id", &self.inner.gpu_id)
            .finish()
    }
}

impl VulkanDevice {
    pub fn new(gpu_id: usize) -> Result<Self> {
        let library =
            unsafe { VulkanLibrary::new() }.map_err(|e| Error::Vulkan(e.to_string().into()))?;
        // Shader `printf` debugging: when VK_LAYER_PRINTF_ENABLE is set,
        // create the instance with the Khronos validation layer enabled.
        // The layer intercepts the NonSemantic.DebugPrintf instructions
        // emitted for slang `printf()` calls and routes the output (e.g. to
        // stdout via VK_LAYER_PRINTF_TO_STDOUT).
        let printf_enabled = std::env::var("VK_LAYER_PRINTF_ENABLE").is_ok();
        let layers: Vec<&str> = if printf_enabled {
            vec!["VK_LAYER_KHRONOS_validation"]
        } else {
            Vec::new()
        };
        let instance = Instance::new(
            &library,
            &InstanceCreateInfo {
                application_name: Some("candle"),
                enabled_layers: &layers,
                ..Default::default()
            },
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let devices = instance
            .enumerate_physical_devices()
            .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let physical = devices
            .get(gpu_id)
            .ok_or_else(|| Error::Vulkan(format!("no physical device {gpu_id}").into()))?;
        let queue_family = physical
            .queue_family_properties()
            .iter()
            .position(|q| q.queue_flags.contains(QueueFlags::COMPUTE))
            .ok_or_else(|| Error::Vulkan("no compute queue family".to_string().into()))?;
        // Copy the supported-feature flags into locals before `physical` is
        // moved into `Device::new`, then enable exactly that subset.
        let supported = physical.supported_features();
        let fb8 = supported.storage_buffer8_bit_access;
        let fb8u = supported.uniform_and_storage_buffer8_bit_access;
        let fb16 = supported.storage_buffer16_bit_access;
        let fb16u = supported.uniform_and_storage_buffer16_bit_access;
        let f16 = supported.shader_float16;
        let i8 = supported.shader_int8;
        let i16 = supported.shader_int16;
        let i64 = supported.shader_int64;
        let f64 = supported.shader_float64;
        // FP8 (FloatE4M3) support: the VK_EXT_shader_float8 device
        // extension plus the shaderFloat8 device feature. The Slang
        // `FloatE4M3` type (the GPU storage type of `microfloat::f8e4m3`)
        // requires both. Enable only what the device reports, so creation
        // never fails.
        let fp8_ext = physical
            .extension_properties()
            .iter()
            .any(|e| e.extension_name.eq("VK_EXT_shader_float8"));
        let fp8 = fp8_ext && supported.shader_float8;
        // Native BFloat16 support: the VK_KHR_shader_bfloat16 device
        // extension plus the shaderBFloat16Type device feature. The Slang
        // `BFloat16` type (the GPU storage type of `half::bf16`) requires
        // both. Enable only what the device reports, so creation never
        // fails.
        let bf16_ext = physical
            .extension_properties()
            .iter()
            .any(|e| e.extension_name.eq("VK_KHR_shader_bfloat16"));
        let bf16 = bf16_ext && supported.shader_b_float16_type;
        // Cooperative matrix support: the VK_KHR_cooperative_matrix
        // device extension plus the cooperativeMatrix device feature.
        // Enables the `linalg.CoopMat` Slang type for hardware GEMM.
        // The companion features the extension defines (robust buffer
        // access, BFloat16 and FloatE4M3 cooperative matrices) are
        // enabled too when the device reports them. Enable only what
        // the device reports, so creation never fails.
        let cm_ext = physical
            .extension_properties()
            .iter()
            .any(|e| e.extension_name.eq("VK_KHR_cooperative_matrix"));
        let cm = cm_ext && supported.cooperative_matrix;
        let cm_robust = cm && supported.cooperative_matrix_robust_buffer_access;
        let cm_bf16 = cm && supported.shader_b_float16_cooperative_matrix;
        let cm_f8 = cm && supported.shader_float8_cooperative_matrix;
        // The compiled coop-mat GEMM shaders assume one 64-thread
        // subgroup per workgroup; the device's subgroup size is
        // captured before `physical` is moved into `Device::new`.
        let subgroup_size = physical.properties().subgroup_size.unwrap_or(0);
        // The coop-mat GEMM shaders also declare the VulkanMemoryModel
        // SPIR-V capability (the cooperative-matrix ops use subgroup
        // memory semantics), so the core feature must be enabled too.
        let vmm = supported.vulkan_memory_model;
        // With the VulkanMemoryModel feature enabled, SPIR-V instructions
        // that use `Device` as the memory scope (e.g. the atomics in
        // `index_add`) additionally require the
        // vulkanMemoryModelDeviceScope feature (VUID
        // RuntimeSpirv-vulkanMemoryModel-06265).
        let vmm_ds = vmm && supported.vulkan_memory_model_device_scope;
        // Atomic float add (used by `index_add`): F32/F64 need
        // VK_EXT_shader_atomic_float, F16 needs VK_EXT_shader_atomic_float2.
        // Enable only what the device reports, so creation never fails.
        let atomic_f32 = supported.shader_buffer_float32_atomic_add;
        let atomic_f64 = supported.shader_buffer_float64_atomic_add;
        let atomic_f16 = supported.shader_buffer_float16_atomic_add;
        let queue_ci = QueueCreateInfo {
            flags: Default::default(),
            queue_family_index: queue_family as u32,
            queues: &[1.0],
            ..Default::default()
        };
        let exts = DeviceExtensions {
            khr_16bit_storage: true,
            khr_shader_float16_int8: true,
            khr_shader_non_semantic_info: printf_enabled,
            ext_shader_atomic_float: atomic_f32 || atomic_f64,
            ext_shader_atomic_float2: atomic_f16,
            ext_shader_float8: fp8_ext,
            khr_shader_bfloat16: bf16_ext,
            khr_cooperative_matrix: cm_ext,
            ..Default::default()
        };
        let feats = DeviceFeatures {
            storage_buffer8_bit_access: fb8,
            uniform_and_storage_buffer8_bit_access: fb8u,
            storage_buffer16_bit_access: fb16,
            uniform_and_storage_buffer16_bit_access: fb16u,
            shader_float16: f16,
            shader_int8: i8,
            shader_int16: i16,
            shader_int64: i64,
            shader_float64: f64,
            shader_float8: fp8,
            shader_b_float16_type: bf16,
            cooperative_matrix: cm,
            cooperative_matrix_robust_buffer_access: cm_robust,
            shader_b_float16_cooperative_matrix: cm_bf16,
            shader_float8_cooperative_matrix: cm_f8,
            vulkan_memory_model: vmm,
            vulkan_memory_model_device_scope: vmm_ds,
            shader_buffer_float32_atomic_add: atomic_f32,
            shader_buffer_float64_atomic_add: atomic_f64,
            shader_buffer_float16_atomic_add: atomic_f16,
            ..Default::default()
        };
        let (device, queues) = Device::new(
            physical,
            &DeviceCreateInfo {
                queue_create_infos: &[queue_ci],
                enabled_extensions: &exts,
                enabled_features: &feats,
                ..Default::default()
            },
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let queue = queues
            .into_iter()
            .next()
            .ok_or_else(|| Error::Vulkan("no queue returned".to_string().into()))?;
        let mem_alloc = Arc::new(StandardMemoryAllocator::new(
            &device,
            &GenericMemoryAllocatorCreateInfo::default(),
        ));
        let cbb_alloc = Arc::new(StandardCommandBufferAllocator::new(
            &device,
            &StandardCommandBufferAllocatorCreateInfo::default(),
        ));
        let dss_alloc = Arc::new(StandardDescriptorSetAllocator::new(
            &device,
            &StandardDescriptorSetAllocatorCreateInfo::default(),
        ));
        let kernels = Arc::new(Kernels::new(
            device.clone(),
            dss_alloc.clone(),
            mem_alloc.clone(),
        ));
        let mut fence_ring = Vec::with_capacity(FENCE_RING_SIZE);
        for _ in 0..FENCE_RING_SIZE {
            let fence = Fence::new(&device, &FenceCreateInfo::default())
                .map_err(|e| Error::Vulkan(e.to_string().into()))?;
            fence_ring.push(InFlight {
                cbb: None,
                fence: Arc::new(fence),
            });
        }
        tracing::debug!(
            "Initializing vulkan device '{gpu_id}' (f8e4m3_native={fp8} bf16_native={bf16})"
        );
        Ok(Self {
            inner: Arc::new(DeviceInner {
                device,
                queue,
                mem_alloc,
                cbb_alloc,
                kernels,
                seed: AtomicU64::new(0),
                fence_ring: std::sync::Mutex::new(fence_ring),
                fence_next: AtomicUsize::new(0),
                keepalive_f32: std::sync::Mutex::new(Vec::new()),
                keepalive_u32: std::sync::Mutex::new(Vec::new()),
                pool_free: std::sync::Mutex::new(FreePool::new()),
                pool_pending: std::sync::Mutex::new(Vec::new()),
                buffer_instances: std::sync::Mutex::new(std::collections::HashMap::new()),
                readback: std::sync::Mutex::new(None),
                pending: std::sync::Mutex::new(Vec::new()),
                atomic_f32,
                atomic_f64,
                atomic_f16,
                supports_f8e4m3: fp8,
                supports_bf16: bf16,
                supports_coopmat: cm,
                subgroup_size,
                vulkan_memory_model: vmm,
                coopmat_gemm_f16: std::sync::OnceLock::new(),
                coopmat_gemm_bf16: std::sync::OnceLock::new(),
                coopmat_affine_f16: std::sync::OnceLock::new(),
                coopmat_affine_bf16: std::sync::OnceLock::new(),
                gpu_id,
            }),
        })
    }

    /// True when the device can execute atomic float add for `dtype`
    /// (F32/F64 require VK_EXT_shader_atomic_float, F16 requires
    /// VK_EXT_shader_atomic_float2), as enabled at device creation.
    pub fn supports_atomic_fadd(&self, dtype: DType) -> bool {
        match dtype {
            DType::F32 => self.inner.atomic_f32,
            DType::F64 => self.inner.atomic_f64,
            DType::F16 => self.inner.atomic_f16,
            _ => false,
        }
    }

    /// True when native FloatE4M3 (FP8 E4M3) support was enabled at
    /// device creation (VK_EXT_shader_float8 + shaderFloat8 feature):
    /// shaders may use the Slang `FloatE4M3` type, the GPU storage type
    /// of `microfloat::f8e4m3`.
    pub fn supports_f8e4m3(&self) -> bool {
        self.inner.supports_f8e4m3
    }

    pub(crate) fn device(&self) -> &Arc<Device> {
        &self.inner.device
    }

    pub(crate) fn mem_alloc(&self) -> &Arc<StandardMemoryAllocator> {
        &self.inner.mem_alloc
    }

    pub(crate) fn cbb_alloc(&self) -> &Arc<StandardCommandBufferAllocator> {
        &self.inner.cbb_alloc
    }

    /// The shared readback state (see the `readback` field on
    /// `DeviceInner`).
    pub(crate) fn readback(&self) -> &std::sync::Mutex<Option<Arc<Buffer>>> {
        &self.inner.readback
    }

    /// Takes a pooled buffer of `dtype` that can hold `bytes` bytes, if
    /// one is available (see the `pool_free` field on `DeviceInner`):
    /// the smallest pooled class that fits is popped and sliced down to
    /// exactly `bytes`.
    pub(crate) fn pool_take(&self, dtype: DType, bytes: usize) -> Option<Subbuffer<[u8]>> {
        let mut free = self.inner.pool_free.lock().ok()?;
        // The smallest class that holds at least one buffer of `bytes`
        // bytes: a class `c` entry holds at most `c` bytes, and a
        // buffer's actual size can be smaller than its class.
        let mut best: Option<usize> = None;
        for ((dt, class), v) in free.map.iter() {
            if *dt != dtype || *class < bytes {
                continue;
            }
            if v.iter().any(|(actual, _)| *actual >= bytes) && best.is_none_or(|b| class < &b) {
                best = Some(*class);
            }
        }
        let class = best?;
        let (actual, arc) = {
            let entry = free.map.get_mut(&(dtype, class))?;
            let pos = entry.iter().position(|(a, _)| *a >= bytes)?;
            entry.remove(pos)
        };
        if free.map.get(&(dtype, class)).is_none_or(Vec::is_empty) {
            free.map.remove(&(dtype, class));
        }
        free.total_bytes -= actual;
        Some(Subbuffer::new(arc).slice(0..bytes as u64))
    }

    /// Returns a buffer of `dtype` to the pool. While a batch is pending
    /// the buffer is held in `pool_pending` until that batch's fence
    /// signals (the GPU may still be using it); otherwise it goes
    /// straight to the free pool. Buffers that would exceed the
    /// per-class, per-entry or total caps are dropped.
    pub(crate) fn pool_return(&self, dtype: DType, buf: Arc<Buffer>) {
        let pending_batch = match self.inner.pending.lock().ok() {
            Some(p) => !p.is_empty(),
            None => true,
        };
        if pending_batch {
            let mut pending = match self.inner.pool_pending.lock().ok() {
                Some(p) => p,
                None => return,
            };
            pending.push((dtype, buf));
            return;
        }
        let mut free = match self.inner.pool_free.lock().ok() {
            Some(f) => f,
            None => return,
        };
        Self::pool_insert(&mut free, dtype, buf);
    }

    /// Inserts one buffer into the free pool, honoring the caps. The
    /// buffer's actual byte size counts against `total_bytes`; its size
    /// class (rounded up) is the pool key.
    fn pool_insert(free: &mut FreePool, dtype: DType, buf: Arc<Buffer>) {
        let actual = buf.size() as usize;
        let class = actual.div_ceil(POOL_CLASS) * POOL_CLASS;
        let key = (dtype, class);
        if !free.map.contains_key(&key) && free.map.len() >= POOL_MAX_ENTRIES {
            return;
        }
        let entry = free.map.entry(key).or_default();
        if entry.len() < POOL_MAX_PER_SIZE && free.total_bytes + actual <= POOL_MAX_BYTES {
            entry.push((actual, buf));
            free.total_bytes += actual;
        }
    }

    /// Registers a new `VulkanStorage` instance holding `buffer`
    /// (see `BufferInstance`).
    pub(crate) fn buffer_instance_add(&self, buffer: &std::sync::Arc<Buffer>) {
        let mut map = match self.inner.buffer_instances.lock() {
            Ok(m) => m,
            Err(_) => return,
        };
        let entry = map
            .entry(std::sync::Arc::as_ptr(buffer) as usize)
            .or_insert_with(|| {
                std::sync::Arc::new(BufferInstance {
                    count: std::sync::atomic::AtomicUsize::new(0),
                })
            });
        entry
            .count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Unregisters a `VulkanStorage` instance holding `buffer`; true
    /// when this was the last instance of the buffer.
    pub(crate) fn buffer_instance_sub(&self, buffer: &std::sync::Arc<Buffer>) -> bool {
        let mut map = match self.inner.buffer_instances.lock() {
            Ok(m) => m,
            Err(_) => return false,
        };
        let Some(entry) = map.get(&(std::sync::Arc::as_ptr(buffer) as usize)).cloned() else {
            return false;
        };
        let prev = entry
            .count
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        if prev == 1 {
            map.remove(&(std::sync::Arc::as_ptr(buffer) as usize));
            return true;
        }
        false
    }

    pub(crate) fn queue(&self) -> &Arc<Queue> {
        &self.inner.queue
    }

    /// Allocation for the compute buffers: device-local (VRAM).
    /// Host-mapped allocations on this platform fall back to system
    /// RAM (the BAR aperture is small), which caps decode at the PCIe
    /// bandwidth; weights and activations live in VRAM instead, with
    /// one-shot staging copies for uploads and readback.
    fn storage_alloc_info() -> AllocationCreateInfo<'static> {
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        }
    }

    /// Host-visible staging memory for uploads and readback.
    fn staging_alloc_info() -> AllocationCreateInfo<'static> {
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        }
    }

    /// True when native BFloat16 support was enabled at device creation
    /// (VK_KHR_shader_bfloat16 + shaderBFloat16Type feature): shaders may
    /// use the Slang `BFloat16` type, the GPU storage type of
    /// `half::bf16`.
    pub fn supports_bf16(&self) -> bool {
        self.inner.supports_bf16
    }

    /// True when the cooperative-matrix GEMM kernels can run: the
    /// VK_KHR_cooperative_matrix extension + cooperativeMatrix feature
    /// were enabled at device creation and the device's subgroup size
    /// matches the compiled shaders (one 64-thread subgroup per
    /// workgroup).
    pub fn supports_coopmat_gemm(&self) -> bool {
        self.inner.supports_coopmat
            && self.inner.vulkan_memory_model
            && self.inner.subgroup_size == 64
    }

    /// True when the cooperative-matrix GEMM pipeline for `dtype` can
    /// run on this device: the feature/subgroup gate above plus a
    /// one-time correctness probe. Some drivers advertise the
    /// cooperative-matrix features but miscompute the runtime-base
    /// load/store (or reject the pipeline build), so the first use
    /// runs a known 16x16 product through the kernel once and caches
    /// the outcome; a failed probe keeps the classic tiled kernel.
    pub(crate) fn coopmat_gemm_available(&self, dtype: DType) -> bool {
        if !self.supports_coopmat_gemm() {
            return false;
        }
        let lock = match dtype {
            DType::F16 => &self.inner.coopmat_gemm_f16,
            DType::BF16 => &self.inner.coopmat_gemm_bf16,
            _ => return false,
        };
        *lock.get_or_init(|| {
            let (source, name) = match dtype {
                DType::F16 => (
                    candle_vulkan_kernels::Source::GemmCoopMatF16,
                    candle_vulkan_kernels::KernelName::GemmCoopMatF16,
                ),
                _ => (
                    candle_vulkan_kernels::Source::GemmCoopMatBf16,
                    candle_vulkan_kernels::KernelName::GemmCoopMatBf16,
                ),
            };
            if self.inner.kernels.load_entry(source, name).is_err() {
                return false;
            }
            match dtype {
                DType::F16 => self.probe_coopmat_f16(source, name),
                _ => self.probe_coopmat_bf16(source, name),
            }
        })
    }

    /// Correctness probe for the f16 cooperative-matrix GEMM: a known
    /// 16x16 product (A all 1.0, B[i] = (i % 16) * 0.5, so every
    /// element of A @ B is 30.0) run through `source`/`name` and
    /// checked against the expected value.
    fn probe_coopmat_f16(
        &self,
        source: candle_vulkan_kernels::Source,
        name: candle_vulkan_kernels::KernelName,
    ) -> bool {
        let a = vec![f16::from_f32(1.0); 256];
        let b: Vec<f16> = (0..256)
            .map(|i| f16::from_f32((i % 16) as f32 * 0.5))
            .collect();
        let out = vec![f16::from_f32(0.0); 256];
        let a_s = match self.storage_from_cpu_storage(&crate::CpuStorage::F16(a)) {
            Ok(s) => s,
            Err(_) => return false,
        };
        let b_s = match self.storage_from_cpu_storage(&crate::CpuStorage::F16(b)) {
            Ok(s) => s,
            Err(_) => return false,
        };
        let out_s = match self.storage_from_cpu_storage(&crate::CpuStorage::F16(out)) {
            Ok(s) => s,
            Err(_) => return false,
        };
        let (a_b, b_b, out_b) = (a_s.buffer(), b_s.buffer(), out_s.buffer());
        let (a_b, b_b, out_b) = match (a_b, b_b, out_b) {
            (
                super::VulkanStorageBuffer::F16(a),
                crate::vulkan_backend::VulkanStorageBuffer::F16(b),
                super::VulkanStorageBuffer::F16(o),
            ) => (a.clone(), b.clone(), o.clone()),
            _ => return false,
        };
        let params = match Buffer::from_iter(
            &self.inner.mem_alloc,
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            vec![1u64, 16, 16, 16, 0, 16, 16],
        ) {
            Ok(b) => b,
            Err(_) => return false,
        };
        let kernels = self.kernels();
        let (a_b, b_b, out_b, params) = (a_b.clone(), b_b.clone(), out_b.clone(), params.clone());
        if self
            .execute(move |cbb| {
                candle_vulkan_kernels::call_gemm_slang::<f16>(
                    cbb, &kernels, source, name, &a_b, &b_b, &out_b, &params, 1, 16, 16,
                )
                .map_err(|e| e.to_string())
            })
            .is_err()
        {
            return false;
        }
        match out_s.to_cpu_storage() {
            Ok(cpu) => match cpu.as_slice::<f16>() {
                Ok(got) => got
                    .iter()
                    .enumerate()
                    .all(|(i, v)| (v.to_f32() - 8.0 * (i % 16) as f32).abs() < 0.5),
                Err(_) => false,
            },
            Err(_) => false,
        }
    }

    /// Correctness probe for the bf16 cooperative-matrix GEMM (same
    /// known product as the f16 probe; the bf16 kernel accumulates in
    /// f16 and stores bf16, so the tolerance is the bf16 step at 30).
    fn probe_coopmat_bf16(
        &self,
        source: candle_vulkan_kernels::Source,
        name: candle_vulkan_kernels::KernelName,
    ) -> bool {
        let a = vec![half::bf16::from_f32(1.0); 256];
        let b: Vec<half::bf16> = (0..256)
            .map(|i| half::bf16::from_f32((i % 16) as f32 * 0.5))
            .collect();
        let out = vec![half::bf16::from_f32(0.0); 256];
        let a_s = match self.storage_from_cpu_storage(&crate::CpuStorage::BF16(a)) {
            Ok(s) => s,
            Err(_) => return false,
        };
        let b_s = match self.storage_from_cpu_storage(&crate::CpuStorage::BF16(b)) {
            Ok(s) => s,
            Err(_) => return false,
        };
        let out_s = match self.storage_from_cpu_storage(&crate::CpuStorage::BF16(out)) {
            Ok(s) => s,
            Err(_) => return false,
        };
        let (a_b, b_b, out_b) = (a_s.buffer(), b_s.buffer(), out_s.buffer());
        let (a_b, b_b, out_b) = match (a_b, b_b, out_b) {
            (
                super::VulkanStorageBuffer::BF16(a),
                super::VulkanStorageBuffer::BF16(b),
                super::VulkanStorageBuffer::BF16(o),
            ) => (a.clone(), b.clone(), o.clone()),
            _ => return false,
        };
        let params = match Buffer::from_iter(
            &self.inner.mem_alloc,
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            vec![1u64, 16, 16, 16, 0, 16, 16],
        ) {
            Ok(b) => b,
            Err(_) => return false,
        };
        let kernels = self.kernels();
        let (a_b, b_b, out_b, params) = (a_b.clone(), b_b.clone(), out_b.clone(), params.clone());
        if self
            .execute(move |cbb| {
                candle_vulkan_kernels::call_gemm_slang::<half::bf16>(
                    cbb, &kernels, source, name, &a_b, &b_b, &out_b, &params, 1, 16, 16,
                )
                .map_err(|e| e.to_string())
            })
            .is_err()
        {
            return false;
        }
        match out_s.to_cpu_storage() {
            Ok(cpu) => match cpu.as_slice::<half::bf16>() {
                Ok(got) => got
                    .iter()
                    .enumerate()
                    .all(|(i, v)| (v.to_f32() - 8.0 * (i % 16) as f32).abs() < 0.5),
                Err(_) => false,
            },
            Err(_) => false,
        }
    }

    /// True when the cooperative-matrix affine (GEMM + bias) pipeline
    /// for `dtype` can run on this device: the cooperative-matrix GEMM
    /// gate (which includes its own correctness probe) plus a one-time
    /// probe of the affine kernel itself. The probe checks the part of
    /// the kernel the GEMM probe does not cover: the bias tile loaded
    /// with a zero row stride (f16) / the per-element bias add (bf16).
    /// A failed probe keeps the unfused matmul + add path.
    pub(crate) fn coopmat_affine_available(&self, dtype: DType) -> bool {
        if !self.coopmat_gemm_available(dtype) {
            return false;
        }
        let lock = match dtype {
            DType::F16 => &self.inner.coopmat_affine_f16,
            DType::BF16 => &self.inner.coopmat_affine_bf16,
            _ => return false,
        };
        *lock.get_or_init(|| match dtype {
            DType::F16 => self.probe_coopmat_affine_f16(),
            _ => self.probe_coopmat_affine_bf16(),
        })
    }

    /// Correctness probe for the f16 cooperative-matrix affine:
    /// A all 1.0, B all 0.0, so the product is zero and every output
    /// row must equal the bias vector (this exercises the
    /// zero-row-stride bias load and the store).
    fn probe_coopmat_affine_f16(&self) -> bool {
        let a = vec![half::f16::from_f32(1.0); 256];
        let b = vec![half::f16::from_f32(0.0); 256];
        let bias: Vec<half::f16> = (0..16)
            .map(|i| half::f16::from_f32((i as f32) * 0.5))
            .collect();
        let out = vec![half::f16::from_f32(0.0); 256];
        let (a_s, b_s, bias_s, out_s) = match (
            self.storage_from_cpu_storage(&crate::CpuStorage::F16(a)),
            self.storage_from_cpu_storage(&crate::CpuStorage::F16(b)),
            self.storage_from_cpu_storage(&crate::CpuStorage::F16(bias)),
            self.storage_from_cpu_storage(&crate::CpuStorage::F16(out)),
        ) {
            (Ok(a), Ok(b), Ok(c), Ok(d)) => (a, b, c, d),
            _ => return false,
        };
        let (a_b, b_b, bias_b, out_b) =
            match (a_s.buffer(), b_s.buffer(), bias_s.buffer(), out_s.buffer()) {
                (
                    super::VulkanStorageBuffer::F16(a),
                    crate::vulkan_backend::VulkanStorageBuffer::F16(b),
                    super::VulkanStorageBuffer::F16(c),
                    super::VulkanStorageBuffer::F16(d),
                ) => (a.clone(), b.clone(), c.clone(), d.clone()),
                _ => return false,
            };
        let params = match Buffer::from_iter(
            &self.inner.mem_alloc,
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            vec![1u64, 16, 16, 16, 0, 16, 16],
        ) {
            Ok(b) => b,
            Err(_) => return false,
        };
        let kernels = self.kernels();
        let (a_b, b_b, bias_b, out_b, params) = (
            a_b.clone(),
            b_b.clone(),
            bias_b.clone(),
            out_b.clone(),
            params.clone(),
        );
        if self
            .execute(move |cbb| {
                candle_vulkan_kernels::call_affine_coopmat_slang::<half::f16>(
                    cbb,
                    &kernels,
                    candle_vulkan_kernels::Source::AffineCoopMatF16,
                    candle_vulkan_kernels::KernelName::AffineCoopMatF16,
                    &a_b,
                    &b_b,
                    &bias_b,
                    &out_b,
                    &params,
                    1,
                    16,
                    16,
                )
                .map_err(|e| e.to_string())
            })
            .is_err()
        {
            return false;
        }
        match out_s.to_cpu_storage() {
            Ok(cpu) => match cpu.as_slice::<half::f16>() {
                Ok(got) => got
                    .iter()
                    .enumerate()
                    .all(|(i, v)| (v.to_f32() - ((i % 16) as f32) * 0.5).abs() < 0.05),
                Err(_) => false,
            },
            Err(_) => false,
        }
    }

    /// Correctness probe for the bf16 cooperative-matrix affine (same
    /// known case as the f16 probe; the kernel accumulates in f16 and
    /// stores bf16, so the tolerance is the bf16 step at 7.5).
    fn probe_coopmat_affine_bf16(&self) -> bool {
        let a = vec![half::bf16::from_f32(1.0); 256];
        let b = vec![half::bf16::from_f32(0.0); 256];
        let bias: Vec<half::bf16> = (0..16)
            .map(|i| half::bf16::from_f32((i as f32) * 0.5))
            .collect();
        let out = vec![half::bf16::from_f32(0.0); 256];
        let (a_s, b_s, bias_s, out_s) = match (
            self.storage_from_cpu_storage(&crate::CpuStorage::BF16(a)),
            self.storage_from_cpu_storage(&crate::CpuStorage::BF16(b)),
            self.storage_from_cpu_storage(&crate::CpuStorage::BF16(bias)),
            self.storage_from_cpu_storage(&crate::CpuStorage::BF16(out)),
        ) {
            (Ok(a), Ok(b), Ok(c), Ok(d)) => (a, b, c, d),
            _ => return false,
        };
        let (a_b, b_b, bias_b, out_b) =
            match (a_s.buffer(), b_s.buffer(), bias_s.buffer(), out_s.buffer()) {
                (
                    super::VulkanStorageBuffer::BF16(a),
                    super::VulkanStorageBuffer::BF16(b),
                    super::VulkanStorageBuffer::BF16(c),
                    super::VulkanStorageBuffer::BF16(d),
                ) => (a.clone(), b.clone(), c.clone(), d.clone()),
                _ => return false,
            };
        let params = match Buffer::from_iter(
            &self.inner.mem_alloc,
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            vec![1u64, 16, 16, 16, 0, 16, 16],
        ) {
            Ok(b) => b,
            Err(_) => return false,
        };
        let kernels = self.kernels();
        let (a_b, b_b, bias_b, out_b, params) = (
            a_b.clone(),
            b_b.clone(),
            bias_b.clone(),
            out_b.clone(),
            params.clone(),
        );
        if self
            .execute(move |cbb| {
                candle_vulkan_kernels::call_affine_coopmat_slang::<half::bf16>(
                    cbb,
                    &kernels,
                    candle_vulkan_kernels::Source::AffineCoopMatBf16,
                    candle_vulkan_kernels::KernelName::AffineCoopMatBf16,
                    &a_b,
                    &b_b,
                    &bias_b,
                    &out_b,
                    &params,
                    1,
                    16,
                    16,
                )
                .map_err(|e| e.to_string())
            })
            .is_err()
        {
            return false;
        }
        match out_s.to_cpu_storage() {
            Ok(cpu) => match cpu.as_slice::<half::bf16>() {
                Ok(got) => got
                    .iter()
                    .enumerate()
                    .all(|(i, v)| (v.to_f32() - ((i % 16) as f32) * 0.5).abs() < 0.5),
                Err(_) => false,
            },
            Err(_) => false,
        }
    }

    /// The underlying Vulkan physical device (for querying properties
    /// and extensions).
    pub fn physical_device(&self) -> &vulkano::device::physical::PhysicalDevice {
        self.inner.device.physical_device()
    }

    /// Uploads `data` into a device f32 storage buffer in VRAM. A
    /// host-visible staging copy is recorded onto the pending batch
    /// and runs at the next `synchronize`.
    pub fn upload_f32(&self, data: &[f32]) -> Result<Subbuffer<[f32]>> {
        let buffer = Buffer::new_slice(
            &self.inner.mem_alloc,
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            &Self::storage_alloc_info(),
            data.len()
                .try_into()
                .map_err(|_| Error::Vulkan("f32 buffer length".to_string().into()))?,
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let staging = Buffer::from_iter(
            &self.inner.mem_alloc,
            &BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            &Self::staging_alloc_info(),
            data.iter().copied(),
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let dst = buffer.clone();
        self.execute(move |cbb| {
            cbb.copy_buffer(CopyBufferInfo::new(staging, dst))
                .map_err(|e| e.to_string())?;
            Ok(())
        })?;
        Ok(buffer)
    }

    /// Allocates a zero-filled f32 buffer of `len` elements in VRAM
    /// (the fill is deferred onto the pending batch).
    pub fn new_f32_buffer(&self, len: usize) -> Result<Subbuffer<[f32]>> {
        let buffer = Buffer::new_slice(
            &self.inner.mem_alloc,
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            &Self::storage_alloc_info(),
            len.try_into()
                .map_err(|_| Error::Vulkan("f32 buffer length".to_string().into()))?,
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let dst: Subbuffer<[u32]> = buffer.clone().reinterpret();
        self.execute(move |cbb| {
            cbb.fill_buffer(dst, 0).map_err(|e| e.to_string())?;
            Ok(())
        })?;
        self.inner
            .keepalive_f32
            .lock()
            .map_err(|e| Error::Vulkan(e.to_string().into()))?
            .push(buffer.clone());
        Ok(buffer)
    }

    /// Allocate a zero-filled u32 storage buffer (the fill is deferred
    /// onto the pending batch).
    pub fn new_u32_buffer(&self, len: usize) -> Result<Subbuffer<[u32]>> {
        let buffer = Buffer::new_slice(
            &self.inner.mem_alloc,
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            &Self::storage_alloc_info(),
            len.try_into()
                .map_err(|_| Error::Vulkan("u32 buffer length".to_string().into()))?,
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let dst: Subbuffer<[u32]> = buffer.clone();
        self.execute(move |cbb| {
            cbb.fill_buffer(dst, 0).map_err(|e| e.to_string())?;
            Ok(())
        })?;
        self.inner
            .keepalive_u32
            .lock()
            .map_err(|e| Error::Vulkan(e.to_string().into()))?
            .push(buffer.clone());
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
        self.inner
            .pending
            .lock()
            .map_err(|e| Error::Vulkan(e.to_string().into()))?
            .push(Box::new(encode));
        Ok(())
    }

    /// Records all pending encodes onto a single command buffer, submits it
    /// (bounded by the fence ring, which also keeps the command buffer and
    /// the buffers it uses alive until the work completes, since the
    /// allocators recycle dropped resources), and waits for completion.
    pub fn synchronize(&self) -> Result<()> {
        let mut pending = self
            .inner
            .pending
            .lock()
            .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let encodes = std::mem::take(&mut *pending);
        if std::env::var("CANDLE_VULKAN_TRACE").is_ok() && !encodes.is_empty() {
            let counts = candle_vulkan_kernels::trace_counts();
            let total: u64 = counts.iter().map(|(_, c)| c).sum();
            tracing::debug!(
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
                self.inner.cbb_alloc.clone(),
                self.inner.queue.queue_family_index(),
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
                .inner
                .fence_next
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                % FENCE_RING_SIZE;
            let cbb_submit = cbb.clone();
            let fence = {
                let mut ring = self
                    .inner
                    .fence_ring
                    .lock()
                    .map_err(|e| Error::Vulkan(e.to_string().into()))?;
                let slot = ring.get_mut(idx).ok_or_else(|| {
                    Error::Vulkan("fence ring index out of range".to_string().into())
                })?;
                if slot.cbb.is_some() {
                    slot.fence
                        .wait(None)
                        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
                }
                unsafe { slot.fence.reset() }.map_err(|e| Error::Vulkan(e.to_string().into()))?;
                slot.cbb = Some(cbb);
                slot.fence.clone()
            };
            self.inner
                .queue
                .with(|mut q| unsafe {
                    q.submit(
                        &[SubmitInfo {
                            wait_semaphores: &[],
                            command_buffers: &[CommandBufferSubmitInfo::new(cbb_submit.as_raw())],
                            signal_semaphores: &[],
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
            self.inner
                .keepalive_f32
                .lock()
                .map_err(|e| Error::Vulkan(e.to_string().into()))?
                .clear();
            self.inner
                .keepalive_u32
                .lock()
                .map_err(|e| Error::Vulkan(e.to_string().into()))?
                .clear();
            // The batch is done on the GPU: promote the pooled buffers
            // that were returned during it.
            let drained = {
                let mut pending = self
                    .inner
                    .pool_pending
                    .lock()
                    .map_err(|e| Error::Vulkan(e.to_string().into()))?;
                std::mem::take(&mut *pending)
            };
            if !drained.is_empty() {
                let mut free = self
                    .inner
                    .pool_free
                    .lock()
                    .map_err(|e| Error::Vulkan(e.to_string().into()))?;
                for (dtype, buf) in drained {
                    Self::pool_insert(&mut free, dtype, buf);
                }
            }
            if std::env::var("CANDLE_VULKAN_PROFILE").is_ok() {
                tracing::debug!(
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
    pub fn upload_u8(&self, data: &[u8]) -> Result<Subbuffer<[u8]>> {
        let buffer = Buffer::new_slice(
            &self.inner.mem_alloc,
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            &Self::storage_alloc_info(),
            data.len()
                .try_into()
                .map_err(|_| Error::Vulkan("u8 buffer length".to_string().into()))?,
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let staging = Buffer::from_iter(
            &self.inner.mem_alloc,
            &BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            &Self::staging_alloc_info(),
            data.iter().copied(),
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let dst = buffer.clone();
        self.execute(move |cbb| {
            cbb.copy_buffer(CopyBufferInfo::new(staging, dst))
                .map_err(|e| e.to_string())?;
            Ok(())
        })?;
        Ok(buffer)
    }
    /// Downloads a device u8 storage buffer to the host.
    pub fn upload_u32(&self, data: &[u32]) -> Result<Subbuffer<[u32]>> {
        let buffer = Buffer::new_slice(
            &self.inner.mem_alloc,
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            &Self::storage_alloc_info(),
            data.len()
                .try_into()
                .map_err(|_| Error::Vulkan("u32 buffer length".to_string().into()))?,
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let staging = Buffer::from_iter(
            &self.inner.mem_alloc,
            &BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            &Self::staging_alloc_info(),
            data.iter().copied(),
        )
        .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let dst = buffer.clone();
        self.execute(move |cbb| {
            cbb.copy_buffer(CopyBufferInfo::new(staging, dst))
                .map_err(|e| e.to_string())?;
            Ok(())
        })?;
        Ok(buffer)
    }

    /// Copies a device u32 buffer (VRAM) back to the host.
    pub fn seed_atomic(&self) -> &AtomicU64 {
        &self.inner.seed
    }

    pub fn gpu_id(&self) -> usize {
        self.inner.gpu_id
    }

    pub fn kernels(&self) -> Arc<Kernels> {
        self.inner.kernels.clone()
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

#[cfg(test)]
mod coopmat_regressions {
    //! Regression tests for driver-specific ordering bugs affecting the
    //! cooperative-matrix kernels (see `Kernels::prime_coopmat_output`).

    use super::*;

    /// RADV (observed on the R9700 with radv 26.2.4) reports the output of
    /// a cooperative-matrix dispatch to a *transfer* command recorded after
    /// the dispatch in the same command buffer as if it were the buffer's
    /// content *before* the dispatch — unless a transfer had already written
    /// to that buffer earlier in the same command buffer. Compute commands
    /// are not affected. `Kernels::prime_coopmat_output` records a tiny
    /// zero copy into the output buffer right before each cooperative-matrix
    /// dispatch to work around the bug.
    ///
    /// This test reproduces the failing shape: an upload batch, a
    /// `synchronize`, then the GEMM and its readback batched together in a
    /// command buffer that is not the first one submitted to the queue.
    /// Without the prime the readback returns the pre-GEMM contents
    /// (all zeros).
    #[test]
    fn coopmat_gemm_readback_same_batch() {
        let gpu = std::env::var("CANDLE_VULKAN_TEST_GPU")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let dev = match VulkanDevice::new(gpu) {
            Ok(d) => d,
            Err(_) => return,
        };
        if !dev.coopmat_gemm_available(DType::F16) {
            eprintln!("skipping: no f16 cooperative-matrix support");
            return;
        }
        let d = crate::Device::Vulkan(dev.clone());
        let data: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01 + 0.5).collect();
        let a = crate::Tensor::new(data.as_slice(), &d)
            .expect("a")
            .reshape(&[16, 16])
            .expect("a2")
            .to_dtype(DType::F16)
            .expect("a16");
        let b = crate::Tensor::new(data.as_slice(), &d)
            .expect("b")
            .reshape(&[16, 16])
            .expect("b2")
            .to_dtype(DType::F16)
            .expect("b16");
        dev.synchronize().expect("mid-sync");
        // GEMM + readback are queued into one pending batch, so they
        // record into a single command buffer that is NOT the first one
        // submitted to the queue.
        let c = a.matmul(&b).expect("matmul");
        let got: Vec<Vec<half::f16>> = c.to_vec2().expect("rb");
        let refv: Vec<Vec<f32>> = (0..16)
            .map(|m| {
                (0..16)
                    .map(|n| (0..16).map(|k| data[m * 16 + k] * data[k * 16 + n]).sum())
                    .collect()
            })
            .collect();
        let bad = got
            .iter()
            .zip(refv.iter())
            .flat_map(|(row, rrow)| {
                row.iter()
                    .zip(rrow.iter())
                    .map(|(g, r)| (g.to_f32() - r).abs() > 0.01 * r.abs().max(1.0))
            })
            .filter(|b| *b)
            .count();
        assert_eq!(bad, 0, "coopmat readback saw stale output");
    }
}
