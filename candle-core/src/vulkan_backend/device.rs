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
    AllocationCreateInfo, DeviceLayout, GenericMemoryAllocatorCreateInfo, MemoryTypeFilter,
    StandardMemoryAllocator,
};
use vulkano::sync::fence::{Fence, FenceCreateInfo};
use vulkano::VulkanLibrary;

use candle_vulkan_kernels::Kernels;

use super::{VulkanStorage, VulkanStorageBuffer};
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
    /// Held across a full readback (copy + wait + host read) so
    /// concurrent readbacks on different threads cannot overwrite the
    /// shared host-visible readback buffer while another thread is
    /// still reading it.
    readback_lock: std::sync::Mutex<()>,
    /// Pinned (host-visible) id-readback ring for the decode pipeline
    /// (`submit_id_readback`): `ID_RING_SLOTS` slots of
    /// `ID_RING_SLOT_BYTES`; the GPU copies each sampled token id into the
    /// next slot and the CPU reads it back one step behind.
    id_ring: std::sync::Mutex<Option<Arc<Buffer>>>,
    id_ring_next: AtomicUsize,
    /// Adaptive estimate (ns) of the GPU time per encoded op: a smoothed
    /// measurement of recent batches (see `finish_batch`), which
    /// `execute` uses to bound each submitted batch below
    /// `BATCH_TIME_LIMIT_NS`.
    batch_ns_per_encode: AtomicU64,
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
    /// BFloat16 cooperative-matrix support, as enabled at device
    /// creation: the `shaderBFloat16CooperativeMatrix` device feature
    /// (the base `cooperativeMatrix` feature is not enough for the bf16
    /// coop-mat GEMM/affine kernels, whose shaders declare the
    /// BFloat16CooperativeMatrixKHR SPIR-V capability). When false the
    /// classic tiled bf16 kernel is used instead.
    coopmat_bf16: bool,
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

/// A submitted batch: the fence it signals, its fence-ring index, the number
/// of encodes it recorded, and its slot generation (see `InFlight`).
type SubmittedBatch = (Arc<Fence>, usize, usize, u64);

/// One slot of the fence ring: the fence the last submission using this
/// slot signals, plus (as a safety net only) a pin on that submission's
/// command buffer while its wait is outstanding: the pin is cleared as
/// soon as the wait succeeds, because holding it would keep every buffer
/// of the batch alive for FENCE_RING_SIZE syncs.
struct InFlight {
    cbb: Option<Arc<PrimaryAutoCommandBuffer>>,
    fence: Arc<Fence>,
    /// Wall-clock instant the last submission using this slot was enqueued,
    /// used to estimate per-encode GPU time for the adaptive batch cap.
    submit_time: std::time::Instant,
    /// Monotonic submission generation: bumped each time the slot is reused,
    /// so `finish_batch` can detect when its (fence, index) has been stale
    /// behind a wrapped ring and skip clearing a slot that now belongs to a
    /// later submission.
    generation: u64,
}

/// A sampled token id that the GPU is copying (or has copied) into the
/// pinned id-readback ring. `wait` blocks until the owning batch's fence
/// signals - in the decode pipeline the GPU is typically already running
/// the next step by then - performs the batch's post-wait cleanup, and
/// returns the id.
pub struct PipedId {
    device: VulkanDevice,
    fence: Arc<Fence>,
    ring_idx: usize,
    n_encodes: usize,
    generation: u64,
    slot: usize,
}

impl PipedId {
    /// Blocks until the id is available and returns it. Consumes the
    /// handle.
    pub fn wait(self) -> Result<u32> {
        self.device.read_id_slot(
            &self.fence,
            self.ring_idx,
            self.n_encodes,
            self.generation,
            self.slot,
        )
    }
}

/// Number of in-flight submissions allowed before `execute` waits for the
/// GPU to catch up.
const FENCE_RING_SIZE: usize = 128;

/// Maximum total GPU time of one submitted command buffer (one driver
/// job). The amdgpu driver aborts any gfx job that runs longer than its
/// lockup timeout (2 s by default): the job is reset mid-flight and its
/// writes are lost, so a later readback returns stale garbage (e.g. an
/// "A weight is negative" sampling error) or the device is reported as
/// lost. The bound is 4x under the timeout; per-encode GPU time varies
/// widely with device load (see `BATCH_EMA_INIT_NS` for the measured
/// range), so the batch cap adapts to the measured per-encode time (see
/// `VulkanInner::batch_ns_per_encode`).
const BATCH_TIME_LIMIT_NS: u64 = 500_000_000;
/// Initial estimate of the per-encode GPU time (ns), used before the
/// adaptive cap has measured the device. It is deliberately far above
/// any per-encode time seen on a contended device, so that the very
/// first submitted batches hold a single encode: on the Granite Ridge
/// iGPU (no dedicated VRAM, system-RAM bandwidth shared with the CPU)
/// prefill averages ~0.3 ms per encode when the CPU is idle, but under
/// a saturated 16-core CPU load the iGPU loses most of its memory
/// bandwidth and the measured per-encode time is ~190 ms with a wide
/// variance (a single encode has been observed to cross the driver's
/// 2 s per-job timeout on its own). Starting from single-encode
/// batches is the only start that cannot trip the driver timeout
/// before the first measurement exists. The EMA then converges to the
/// measured per-encode time over a few dozen batches, so fast devices
/// (per-encode in the microseconds) pay only a one-time cost of a few
/// dozen extra submissions at the start of a workload.
const BATCH_EMA_INIT_NS: u64 = 2_000_000_000;
/// Hard cap on encodes per submitted batch: a backstop for the adaptive
/// time cap (with the measured per-encode time floored at 1 us the time
/// cap alone would allow unbounded batches of trivially small ops).
const MAX_BATCH_OPS: usize = 64;

/// Slots in the pinned id-readback ring (see
/// `VulkanDevice::submit_id_readback`): the decode pipeline keeps at most
/// two batches in flight, so two slots would suffice; four leaves headroom.
const ID_RING_SLOTS: usize = 4;
/// Bytes per id-readback slot (only the first 4 hold the u32 id).
const ID_RING_SLOT_BYTES: u64 = 64;

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
        // Integer dot product (DP4A / V_DOT4C_I32_I8): the
        // VK_KHR_shader_integer_dot_product device extension plus the
        // shaderIntegerDotProduct device feature, enabling the Slang
        // `dot4add_i8packed`/`dot4add_u8packed` intrinsics. Enable only
        // what the device reports, so creation never fails.
        let dp4a_ext = physical
            .extension_properties()
            .iter()
            .any(|e| e.extension_name.eq("VK_KHR_shader_integer_dot_product"));
        let dp4a = dp4a_ext && supported.shader_integer_dot_product;
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
            khr_shader_integer_dot_product: dp4a_ext,
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
            shader_integer_dot_product: dp4a,
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
                submit_time: std::time::Instant::now(),
                generation: 0,
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
                readback_lock: std::sync::Mutex::new(()),
                id_ring: std::sync::Mutex::new(None),
                id_ring_next: AtomicUsize::new(0),
                batch_ns_per_encode: AtomicU64::new(BATCH_EMA_INIT_NS),
                pending: std::sync::Mutex::new(Vec::new()),
                atomic_f32,
                atomic_f64,
                atomic_f16,
                supports_f8e4m3: fp8,
                supports_bf16: bf16,
                supports_coopmat: cm,
                coopmat_bf16: cm_bf16,
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

    /// The readback exclusion lock (see the `readback_lock` field on
    /// `DeviceInner`).
    pub(crate) fn readback_lock(&self) -> &std::sync::Mutex<()> {
        &self.inner.readback_lock
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
    /// per-class, per-entry or total caps are dropped. The pending list
    /// is bounded by the same total byte budget: when adding `buf`
    /// would exceed it, the batch is submitted and waited on first, so
    /// the list is flushed through the caps and the excess is actually
    /// freed (holding it longer is impossible — the GPU owns it, and
    /// freeing it early would release memory the GPU is still using).
    pub(crate) fn pool_return(&self, dtype: DType, buf: Arc<Buffer>) {
        let pending_batch = match self.inner.pending.lock().ok() {
            Some(p) => !p.is_empty(),
            None => true,
        };
        if pending_batch {
            let over_cap = {
                let pending = match self.inner.pool_pending.lock().ok() {
                    Some(p) => p,
                    None => return,
                };
                let held: usize = pending.iter().map(|(_, b)| b.size() as usize).sum();
                held + buf.size() as usize > POOL_MAX_BYTES
            };
            if !over_cap {
                let mut pending = match self.inner.pool_pending.lock().ok() {
                    Some(p) => p,
                    None => return,
                };
                pending.push((dtype, buf));
                return;
            }
            // The pending list would exceed the pool's byte budget and
            // the GPU still owns every held buffer, so the excess can
            // only be released after the batch completes: submit and
            // wait, which flushes the list through the caps (excess is
            // freed) and leaves the GPU idle.
            if self.synchronize().is_err() {
                // A sync failure means the device is already broken;
                // keep holding the buffer (the old behavior) rather than
                // freeing one the GPU may still reference.
                let mut pending = match self.inner.pool_pending.lock().ok() {
                    Some(p) => p,
                    None => return,
                };
                pending.push((dtype, buf));
                return;
            }
            // Fall through: the GPU is idle on `buf`; pool it (the caps
            // apply) or drop it if it exceeds them.
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
    pub fn coopmat_gemm_available(&self, dtype: DType) -> bool {
        if !self.supports_coopmat_gemm() {
            return false;
        }
        // The bf16 coop-mat kernels additionally require the
        // shaderBFloat16CooperativeMatrix device feature; the base
        // cooperativeMatrix feature is not enough for them. On devices
        // that expose only the base feature (e.g. RDNA 3.5 iGPUs) the
        // bf16 pipeline build would fail, so keep the classic tiled
        // kernel.
        if dtype == DType::BF16 && !self.inner.coopmat_bf16 {
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
    /// element [i] of A @ B is 8 * (i % 16)) run through `source`/`name`
    /// and checked against the expected value.
    fn probe_coopmat_f16(
        &self,
        source: candle_vulkan_kernels::Source,
        name: candle_vulkan_kernels::KernelName,
    ) -> bool {
        // m = n = 16, k = 32 (two slabs): the multi-slab K catches devices
        // (e.g. RDNA 3.5 iGPU drivers) that miscompute the f32 cooperative-
        // matrix accumulator beyond a single 16-wide slab.
        let a = vec![f16::from_f32(1.0); 16 * 32];
        let b: Vec<f16> = (0..32 * 16)
            .map(|i| f16::from_f32((i % 16) as f32 * 0.5))
            .collect();
        let out = vec![0.0f32; 16 * 16];
        let a_s = match self.storage_from_cpu_storage(&crate::CpuStorage::F16(a)) {
            Ok(s) => s,
            Err(_) => return false,
        };
        let b_s = match self.storage_from_cpu_storage(&crate::CpuStorage::F16(b)) {
            Ok(s) => s,
            Err(_) => return false,
        };
        let out_s = match self.storage_from_cpu_storage(&crate::CpuStorage::F32(out)) {
            Ok(s) => s,
            Err(_) => return false,
        };
        let (a_b, b_b, out_b) = (a_s.buffer(), b_s.buffer(), out_s.buffer());
        let (a_b, b_b, out_b) = match (a_b, b_b, out_b) {
            (
                super::VulkanStorageBuffer::F16(a),
                crate::vulkan_backend::VulkanStorageBuffer::F16(b),
                super::VulkanStorageBuffer::F32(o),
            ) => (a.clone(), b.clone(), o.clone()),
            _ => return false,
        };
        let params = candle_vulkan_kernels::push_constants::GemmParams {
            batch: 1,
            m: 16,
            n: 16,
            k: 32,
            rhs_transposed: 0,
            sk: 32,
            sn: 16,
        };
        let kernels = self.kernels();
        let (a_b, b_b, out_b) = (a_b.clone(), b_b.clone(), out_b.clone());
        if self
            .execute(move |cbb| {
                candle_vulkan_kernels::call_gemm_coopmat_f32_out::<f16>(
                    cbb, &kernels, source, name, &a_b, &b_b, &out_b, &params, 1, 16, 16,
                )
                .map_err(|e| e.to_string())
            })
            .is_err()
        {
            return false;
        }
        match out_s.to_cpu_storage() {
            Ok(cpu) => match cpu.as_slice::<f32>() {
                Ok(got) => got
                    .iter()
                    .enumerate()
                    .all(|(i, v)| (v - 16.0 * (i % 16) as f32).abs() < 0.05),
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
        let params = candle_vulkan_kernels::push_constants::GemmParams {
            batch: 1,
            m: 16,
            n: 16,
            k: 16,
            rhs_transposed: 0,
            sk: 16,
            sn: 16,
        };
        let kernels = self.kernels();
        let (a_b, b_b, out_b) = (a_b.clone(), b_b.clone(), out_b.clone());
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
        let params = candle_vulkan_kernels::push_constants::GemmParams {
            batch: 1,
            m: 16,
            n: 16,
            k: 16,
            rhs_transposed: 0,
            sk: 16,
            sn: 16,
        };
        let kernels = self.kernels();
        let (a_b, b_b, bias_b, out_b) = (a_b.clone(), b_b.clone(), bias_b.clone(), out_b.clone());
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
        let params = candle_vulkan_kernels::push_constants::GemmParams {
            batch: 1,
            m: 16,
            n: 16,
            k: 16,
            rhs_transposed: 0,
            sk: 16,
            sn: 16,
        };
        let kernels = self.kernels();
        let (a_b, b_b, bias_b, out_b) = (a_b.clone(), b_b.clone(), bias_b.clone(), out_b.clone());
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
        let reached_cap = {
            let mut pending = self
                .inner
                .pending
                .lock()
                .map_err(|e| Error::Vulkan(e.to_string().into()))?;
            pending.push(Box::new(encode));
            let n = pending.len() as u64;
            let est_ns = n.saturating_mul(
                self.inner
                    .batch_ns_per_encode
                    .load(std::sync::atomic::Ordering::Relaxed),
            );
            n >= MAX_BATCH_OPS as u64 || est_ns > BATCH_TIME_LIMIT_NS
        };
        if reached_cap {
            // Flush the batch; see BATCH_TIME_LIMIT_NS.
            self.synchronize()?;
        }
        Ok(())
    }

    /// Records all pending encodes onto a single command buffer, submits it
    /// (bounded by the fence ring), and waits for completion. The command
    /// buffer (and with it every buffer of the batch) is kept alive only
    /// until the wait returns, then released.
    pub fn synchronize(&self) -> Result<()> {
        match self.submit_batch()? {
            Some((fence, ring_idx, n_encodes, generation)) => {
                self.finish_batch(&fence, ring_idx, n_encodes, generation)
            }
            None => Ok(()),
        }
    }

    /// Records all pending encodes onto a single command buffer and
    /// submits it (bounded by the fence ring) without waiting. Returns the
    /// batch's fence, fence-ring index, and encode count, or `None` when
    /// there were no pending encodes. The command buffer (and with it every buffer of the
    /// batch) stays pinned in the ring slot until `finish_batch` releases
    /// it, so in-flight batches keep their buffers alive.
    pub fn submit_batch(&self) -> Result<Option<SubmittedBatch>> {
        let mut pending = self
            .inner
            .pending
            .lock()
            .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        let encodes = std::mem::take(&mut *pending);
        // Release the pending lock before recording: recording is the slow part
        // and must not serialize other threads' `execute` calls (or deadlock if
        // an encode re-enters `execute`).
        drop(pending);
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
        if encodes.is_empty() {
            return Ok(None);
        }
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
        let (fence, generation) = {
            let mut ring = self
                .inner
                .fence_ring
                .lock()
                .map_err(|e| Error::Vulkan(e.to_string().into()))?;
            let slot = ring
                .get_mut(idx)
                .ok_or_else(|| Error::Vulkan("fence ring index out of range".to_string().into()))?;
            if slot.cbb.is_some() {
                slot.fence
                    .wait(None)
                    .map_err(|e| Error::Vulkan(e.to_string().into()))?;
            }
            unsafe { slot.fence.reset() }.map_err(|e| Error::Vulkan(e.to_string().into()))?;
            slot.cbb = Some(cbb);
            slot.submit_time = std::time::Instant::now();
            slot.generation = slot.generation.wrapping_add(1);
            (slot.fence.clone(), slot.generation)
        };
        let submit_result = self.inner.queue.with(|mut q| unsafe {
            q.submit(
                &[SubmitInfo {
                    wait_semaphores: &[],
                    command_buffers: &[CommandBufferSubmitInfo::new(cbb_submit.as_raw())],
                    signal_semaphores: &[],
                    ..Default::default()
                }],
                Some(&fence),
            )
        });
        if submit_result.is_err() {
            // The fence was reset before submission; a failed submit leaves a
            // never-signaling fence pinned to the slot, which would deadlock
            // the next submission to it. Un-arm the slot's command-buffer pin.
            let mut ring = self
                .inner
                .fence_ring
                .lock()
                .map_err(|e| Error::Vulkan(e.to_string().into()))?;
            if let Some(slot) = ring.get_mut(idx) {
                slot.cbb = None;
            }
        }
        submit_result.map_err(|e| Error::Vulkan(e.to_string().into()))?;
        if std::env::var("CANDLE_VULKAN_PROFILE").is_ok() {
            tracing::debug!(
                "vulkan profile: {} encodes, encode+build {:?}",
                n_encodes,
                t_build
            );
        }
        Ok(Some((fence, idx, n_encodes, generation)))
    }

    /// Waits for a batch submitted by `submit_batch` and releases everything
    /// the batch was pinning: the command buffer's pin on every buffer of
    /// the batch (the ring slot keeps it only as a safety net for a wait
    /// timeout, when the buffers are still in flight), the keepalive lists
    /// (so the allocator can recycle the memory), and the promotion of the
    /// pooled buffers that were returned during the batch.
    pub fn finish_batch(
        &self,
        fence: &Arc<Fence>,
        ring_idx: usize,
        n_encodes: usize,
        generation: u64,
    ) -> Result<()> {
        let t_wait = std::time::Instant::now();
        fence
            .wait(Some(std::time::Duration::from_secs(60)))
            .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        if n_encodes > 0 {
            // Update the adaptive batch cap. Measure from submission, not the
            // fence wait: in the pipelined decode path the fence is already
            // signaled by the time `finish_batch` runs, so the wait duration is
            // ~0 and would decay the estimate to its 1 us floor, disabling the
            // driver-timeout protection the cap exists to provide.
            let submit_elapsed_ns = {
                let ring = self
                    .inner
                    .fence_ring
                    .lock()
                    .map_err(|e| Error::Vulkan(e.to_string().into()))?;
                ring.get(ring_idx)
                    .filter(|s| s.generation == generation)
                    .map(|s| s.submit_time.elapsed().as_nanos() as u64)
                    .unwrap_or(0)
            };
            let measured_ns = std::cmp::max(1_000, submit_elapsed_ns / n_encodes as u64);
            let old = self
                .inner
                .batch_ns_per_encode
                .load(std::sync::atomic::Ordering::Relaxed);
            let new = (old.saturating_mul(3) + measured_ns) / 4;
            self.inner
                .batch_ns_per_encode
                .store(new, std::sync::atomic::Ordering::Relaxed);
        }
        {
            let mut ring = self
                .inner
                .fence_ring
                .lock()
                .map_err(|e| Error::Vulkan(e.to_string().into()))?;
            if let Some(slot) = ring.get_mut(ring_idx) {
                // Only clear the slot if it still belongs to this submission: a
                // wrapped fence ring may have reused the slot for a later batch,
                // and clearing it here would drop that batch's pin.
                if slot.generation == generation {
                    slot.cbb = None;
                }
            }
        }
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
            tracing::debug!("vulkan profile: submit+wait {:?}", t_wait.elapsed());
        }
        Ok(())
    }

    /// Lazily creates the pinned id-readback ring: `ID_RING_SLOTS`
    /// host-visible slots of `ID_RING_SLOT_BYTES` that `submit_id_readback`
    /// copies sampled token ids into.
    fn id_ring_buffer(&self) -> Result<Subbuffer<[u8]>> {
        let bytes = ID_RING_SLOTS as u64 * ID_RING_SLOT_BYTES;
        let mut ring = self
            .inner
            .id_ring
            .lock()
            .map_err(|e| Error::Vulkan(e.to_string().into()))?;
        if let Some(b) = ring.as_ref() {
            return Ok(Subbuffer::from(b.clone()).slice(0..bytes));
        }
        let buffer = Buffer::new(
            self.mem_alloc(),
            &BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            DeviceLayout::from_size_alignment(bytes, 1).ok_or_else(|| {
                Error::Vulkan("id readback ring: invalid device layout".to_string().into())
            })?,
        )
        .map_err(|error| {
            Error::Vulkan(format!("Unable to create host id-readback buffer: {error:?}").into())
        })?;
        *ring = Some(buffer.clone());
        Ok(Subbuffer::from(buffer).slice(0..bytes))
    }

    /// Records a 4-byte copy of the one-element U32 storage `id` into the
    /// next slot of the pinned id-readback ring, then submits the pending
    /// batch without waiting: the decode pipeline keeps the GPU running the
    /// next step while the caller reads this id back one step behind via
    /// `PipedId::wait`.
    pub fn submit_id_readback(&self, id: &VulkanStorage) -> Result<PipedId> {
        let src: Subbuffer<[u8]> = match id.buffer() {
            VulkanStorageBuffer::U32(sub) if sub.len() == 1 && sub.offset() == 0 => {
                // Below the pool minimum, so the subbuffer is the whole
                // (freshly allocated) buffer: the byte view is exact.
                id.buffer().as_u8().slice(0..4)
            }
            _ => {
                return Err(Error::Vulkan(
                    "submit_id_readback: id must be a one-element U32 storage"
                        .to_string()
                        .into(),
                ))
            }
        };
        let ring = self.id_ring_buffer()?;
        let slot = self
            .inner
            .id_ring_next
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            % ID_RING_SLOTS;
        let offset = slot as u64 * ID_RING_SLOT_BYTES;
        let dst = ring.slice(offset..offset + 4);
        // Record the id copy directly onto the pending batch, bypassing the
        // cap-flushing `execute`: `execute` can synchronously flush the batch
        // when this copy trips the cap, which would leave nothing for the
        // follow-up `submit_batch` to submit and surface as a spurious "no
        // batch was submitted" error.
        {
            let mut pending = self
                .inner
                .pending
                .lock()
                .map_err(|e| Error::Vulkan(e.to_string().into()))?;
            pending.push(Box::new(move |cbb| {
                cbb.copy_buffer(CopyBufferInfo::new(src, dst))
                    .map_err(|e| e.to_string())?;
                Ok(())
            }));
        }
        let (fence, ring_idx, n_encodes, generation) = self.submit_batch()?.ok_or_else(|| {
            Error::Vulkan(
                "submit_id_readback: no batch was submitted"
                    .to_string()
                    .into(),
            )
        })?;
        Ok(PipedId {
            device: self.clone(),
            fence,
            ring_idx,
            n_encodes,
            generation,
            slot,
        })
    }

    /// Waits for the batch that copied `slot` into the pinned id-readback
    /// ring, performs the batch's post-wait cleanup, and reads the id.
    fn read_id_slot(
        &self,
        fence: &Arc<Fence>,
        ring_idx: usize,
        n_encodes: usize,
        generation: u64,
        slot: usize,
    ) -> Result<u32> {
        self.finish_batch(fence, ring_idx, n_encodes, generation)?;
        let ring = self.id_ring_buffer()?;
        let offset = slot as u64 * ID_RING_SLOT_BYTES;
        let slot_buf = ring.slice(offset..offset + 4);
        let bytes = slot_buf.read().map_err(|e| {
            Error::Vulkan(
                format!("Unable to acquire read lock on id-readback buffer: {e:?}").into(),
            )
        })?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// Uploads host `u8` data to a new device-local storage buffer via a
    /// host-visible staging buffer.
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
    /// Uploads host `u32` data to a new device-local storage buffer.
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

    /// Returns the device's atomic seed counter.
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
