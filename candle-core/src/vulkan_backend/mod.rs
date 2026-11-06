//! Vulkan backend for candle-core.
//!
//! Enabled with the `vulkan` feature.  Uses `ash` with its `loaded` feature,
//! so `libvulkan.so` is pulled in at runtime rather than hard-linked, and
//! `naga` to compile the embedded GLSL kernels to SPIR-V, so no external
//! `glslc` is needed to build or run.
//!
//! Every tensor lives in a device buffer sub-allocated from large memory
//! blocks (drivers cap the number of `VkDeviceMemory` allocations, often at
//! 4096) of a host-visible, host-coherent memory type — device-local when
//! the hardware offers a unified type, as an iGPU does — so uploads and
//! read-backs are plain `memcpy`s through the mapped pointer.  Every
//! operator runs as a compute kernel ([`kernels`], [`glsl`]); kernels are
//! recorded into one command buffer per device and submitted lazily (see
//! [`kernels::Exec`]).  Block-quantized weights stay in their GGUF format on
//! the device ([`QVulkanStorage`]) and are dequantized inside the matmul.
#![allow(clippy::missing_safety_doc)]

pub mod glsl;
pub mod kernels;
pub mod wgsl;
pub use kernels::{
    fallback_count, native_enabled, native_exec_count, reset_stats, stats, VulkanStats,
};

/// Enabled only after the end-to-end B50 model qualification is complete.
pub const VALIDATED_B50_PRECISE_MATH: bool = true;

use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::quantized::GgmlDType;
use crate::{CpuStorage, DType, Error, Layout, Result, Shape};
use ash::vk;
use kernels::{Buf, Idx, MatStrides};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

/// The `DeviceLocation::Vulkan` variant carries a `gpu_id`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

/// Device limits the kernels are shaped by.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Work-group size of the 1-D kernels (a power of two ≤ 256).
    pub wg: usize,
    /// Side of the square GEMM work-group (16 → 64×64 tiles, 8 → 32×32).
    pub gemm_tx: usize,
    /// `maxStorageBufferRange`: the largest buffer a kernel may bind.
    pub max_storage_range: u64,
    /// `maxComputeWorkGroupCount[0]`.
    pub max_groups_x: u32,
    /// Whether tensor memory is device-local as well as host-visible
    /// (unified memory: an iGPU or a software device).
    pub host_unified: bool,
    /// The quantized-GEMV kernel for this device ([`kernels::Qgemv`]).
    pub qgemv: kernels::Qgemv,
    /// Whether the `shaderIntegerDotProduct` feature is enabled (packed
    /// int8 dot products, SPIR-V `OpSDot`).
    pub int_dot: bool,
}

// ─── Memory sub-allocator ────────────────────────────────────────────────────

/// A range inside one memory block.
#[derive(Clone, Copy, Debug)]
pub struct Alloc {
    block: usize,
    offset: u64,
    size: u64,
}

struct Block {
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    size: u64,
    /// Free ranges `(offset, len)`, sorted by offset, coalesced.
    free: Vec<(u64, u64)>,
    dedicated: bool,
}

unsafe impl Send for Block {}

/// First-fit sub-allocator over large `VkDeviceMemory` blocks.
struct Allocator {
    blocks: Vec<Option<Block>>,
    chunk: u64,
    mem_type: u32,
}

/// Default block size; requests above a quarter of it get a dedicated block.
const CHUNK_BYTES: u64 = 128 << 20;

fn align_up(v: u64, a: u64) -> u64 {
    if a <= 1 {
        v
    } else {
        v.div_ceil(a) * a
    }
}

impl Allocator {
    fn alloc(&mut self, device: &ash::Device, size: u64, align: u64) -> Result<(Alloc, *mut u8)> {
        let size = size.max(1);
        let dedicated = size > self.chunk / 4;
        if !dedicated {
            for (bi, slot) in self.blocks.iter_mut().enumerate() {
                let Some(b) = slot else { continue };
                if b.dedicated {
                    continue;
                }
                for fi in 0..b.free.len() {
                    let (off, len) = b.free[fi];
                    let start = align_up(off, align);
                    if start + size > off + len {
                        continue;
                    }
                    let end = start + size;
                    let mut repl = Vec::with_capacity(2);
                    if start > off {
                        repl.push((off, start - off));
                    }
                    if off + len > end {
                        repl.push((end, off + len - end));
                    }
                    b.free.splice(fi..fi + 1, repl);
                    let ptr = unsafe { b.mapped.add(start as usize) };
                    return Ok((
                        Alloc {
                            block: bi,
                            offset: start,
                            size,
                        },
                        ptr,
                    ));
                }
            }
        }
        let bsize = if dedicated { size } else { self.chunk };
        let memory = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(bsize)
                    .memory_type_index(self.mem_type),
                None,
            )
        }
        .map_err(|e| {
            Error::Msg(format!(
                "vulkan allocate_memory({bsize} bytes) failed: {e:?}"
            ))
        })?;
        let mapped = match unsafe {
            device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        } {
            Ok(p) => p as *mut u8,
            Err(e) => {
                unsafe { device.free_memory(memory, None) };
                return Err(Error::Msg(format!("vulkan map_memory failed: {e:?}")));
            }
        };
        let free = if bsize > size {
            vec![(size, bsize - size)]
        } else {
            Vec::new()
        };
        let block = Block {
            memory,
            mapped,
            size: bsize,
            free,
            dedicated,
        };
        let bi = match self.blocks.iter().position(|b| b.is_none()) {
            Some(i) => {
                self.blocks[i] = Some(block);
                i
            }
            None => {
                self.blocks.push(Some(block));
                self.blocks.len() - 1
            }
        };
        Ok((
            Alloc {
                block: bi,
                offset: 0,
                size,
            },
            mapped,
        ))
    }

    fn free(&mut self, device: &ash::Device, a: Alloc) {
        let Some(Some(b)) = self.blocks.get_mut(a.block) else {
            return;
        };
        let pos = b.free.partition_point(|&(o, _)| o < a.offset);
        b.free.insert(pos, (a.offset, a.size));
        if pos + 1 < b.free.len() && b.free[pos].0 + b.free[pos].1 == b.free[pos + 1].0 {
            let (_, l2) = b.free.remove(pos + 1);
            b.free[pos].1 += l2;
        }
        if pos > 0 && b.free[pos - 1].0 + b.free[pos - 1].1 == b.free[pos].0 {
            let (_, l) = b.free.remove(pos);
            b.free[pos - 1].1 += l;
        }
        let empty = b.free.len() == 1 && b.free[0] == (0, b.size);
        if empty {
            let dedicated = b.dedicated;
            // Return dedicated blocks at once; keep one empty chunk around.
            let another_empty = self.blocks.iter().enumerate().any(|(i, s)| {
                i != a.block
                    && s.as_ref().is_some_and(|x| {
                        !x.dedicated && x.free.len() == 1 && x.free[0] == (0, x.size)
                    })
            });
            if dedicated || another_empty {
                let b = self.blocks[a.block].take().unwrap();
                unsafe {
                    device.unmap_memory(b.memory);
                    device.free_memory(b.memory, None);
                }
            }
        }
    }
}

// ─── Device ──────────────────────────────────────────────────────────────────

/// One private, reusable dequantization target. It deliberately owns no
/// VulkanDevice/Arc: storing a VulkanStorage in its device would form a cycle.
struct QuantGemmScratch {
    buffer: vk::Buffer,
    alloc: Alloc,
    capacity: u64,
}

/// One Vulkan logical device with everything the storage and the kernels
/// share, behind an `Arc` so every clone keeps the handles alive.
struct VulkanContext {
    /// Kept so the loader stays loaded for the life of the instance.
    #[allow(dead_code)]
    entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    name: String,
    info: VulkanAdapterInfo,
    adapter_id: (u32, u32),
    shader_float64: bool,
    limits: Limits,
    buffer_align: u64,
    mem: Mutex<Allocator>,
    // Lock order: quant_gemm_scratch -> exec -> mem. Hold this guard through
    // both dispatches so concurrent requests cannot overwrite another GEMM's
    // weights between dequantization and use.
    quant_gemm_scratch: Mutex<Option<QuantGemmScratch>>,
    exec: std::mem::ManuallyDrop<Mutex<kernels::Exec>>,
    pipes: Mutex<kernels::PipeMap>,
    layouts: Mutex<HashMap<u32, Arc<kernels::Layouts>>>,
}

unsafe impl Send for VulkanContext {}
unsafe impl Sync for VulkanContext {}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            // The recorder owns its command pool / fence / ring: destroy it
            // before the device.
            std::mem::ManuallyDrop::drop(&mut self.exec);
            if let Some(scratch) = self
                .quant_gemm_scratch
                .get_mut()
                .unwrap_or_else(|p| p.into_inner())
                .take()
            {
                self.device.destroy_buffer(scratch.buffer, None);
            }
            for (_, p) in self.pipes.lock().unwrap_or_else(|p| p.into_inner()).drain() {
                self.device.destroy_pipeline(p.pipeline, None);
            }
            for (_, l) in self
                .layouts
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .drain()
            {
                self.device.destroy_pipeline_layout(l.layout, None);
                self.device.destroy_descriptor_set_layout(l.dsl, None);
            }
            let mut mem = self.mem.lock().unwrap_or_else(|p| p.into_inner());
            for b in mem.blocks.drain(..).flatten() {
                self.device.unmap_memory(b.memory);
                self.device.free_memory(b.memory, None);
            }
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// A Vulkan device: an immutable `gpu_id` + the shared context.
pub struct VulkanDevice {
    gpu_id: usize,
    inner: Arc<VulkanContext>,
    // Per-handle policy, not shared mutable device state. Models may share
    // buffers/queues without changing each other's arithmetic dispatch.
    precise_math: bool,
}

impl Clone for VulkanDevice {
    fn clone(&self) -> Self {
        VulkanDevice {
            gpu_id: self.gpu_id,
            inner: self.inner.clone(),
            precise_math: self.precise_math,
        }
    }
}

impl std::fmt::Debug for VulkanDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanDevice")
            .field("gpu_id", &self.gpu_id)
            .field("name", &self.inner.name)
            .finish()
    }
}

fn cstr_name(raw: &[std::ffi::c_char]) -> String {
    let bytes: Vec<u8> = raw
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).to_string()
}

/// Usage every tensor buffer is created with (and the memory-type probe).
const TENSOR_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
        | vk::BufferUsageFlags::TRANSFER_SRC.as_raw()
        | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
);

/// Serialises first-time initialisation (some loaders mis-enumerate under
/// concurrent instance creation).
static INIT: Mutex<()> = Mutex::new(());

/// A Vulkan physical device as reported by the driver, for device listing,
/// selection and "which GPU actually runs this" reporting.
#[derive(Clone, Debug)]
pub struct VulkanAdapterInfo {
    /// Index in `vkEnumeratePhysicalDevices` order (the `gpu_id` / ordinal).
    pub index: usize,
    /// `VkPhysicalDeviceProperties::deviceName`, e.g. `AMD BC-250 (RADV GFX1013)`.
    pub name: String,
    /// `integrated`, `discrete`, `virtual`, `cpu` or `other`.
    pub device_type: &'static str,
    pub vendor_id: u32,
    pub device_id: u32,
    /// `VkPhysicalDeviceDriverProperties::driverName`, e.g. `radv`.
    pub driver_name: String,
    /// `VkPhysicalDeviceDriverProperties::driverInfo`, e.g. `Mesa 26.2.5`.
    pub driver_info: String,
    /// Supported Vulkan API version `(major, minor, patch)`.
    pub api_version: (u32, u32, u32),
    /// Total size of the device-local memory heaps, in bytes.
    pub device_local_bytes: u64,
    /// Whether some memory type is device-local and host-visible (unified
    /// memory, as on an APU such as the BC-250).
    pub unified_memory: bool,
    /// Whether the device supports `shaderFloat64`.
    pub shader_float64: bool,
    /// Whether the device offers `VK_KHR_shader_integer_dot_product`.
    pub integer_dot_product: bool,
}

impl VulkanAdapterInfo {
    /// A CPU implementation (lavapipe / llvmpipe / SwiftShader): running on it
    /// is CPU execution, not GPU acceleration.
    pub fn is_software(&self) -> bool {
        self.device_type == "cpu"
    }
}

impl std::fmt::Display for VulkanAdapterInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (ma, mi, pa) = self.api_version;
        write!(
            f,
            "vulkan:{} {} ({}, driver {} {}, Vulkan {ma}.{mi}.{pa}, {:.1} GiB device-local{})",
            self.index,
            self.name,
            self.device_type,
            self.driver_name,
            self.driver_info,
            self.device_local_bytes as f64 / (1u64 << 30) as f64,
            if self.unified_memory {
                ", unified memory"
            } else {
                ""
            }
        )
    }
}

fn adapter_info(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    index: usize,
) -> VulkanAdapterInfo {
    let props = unsafe { instance.get_physical_device_properties(physical) };
    let mut driver = vk::PhysicalDeviceDriverProperties::default();
    {
        let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut driver);
        unsafe { instance.get_physical_device_properties2(physical, &mut p2) };
    }
    let mem = unsafe { instance.get_physical_device_memory_properties(physical) };
    let heaps = &mem.memory_heaps[..mem.memory_heap_count as usize];
    let types = &mem.memory_types[..mem.memory_type_count as usize];
    let device_local_bytes = heaps
        .iter()
        .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .map(|h| h.size)
        .sum();
    let unified_memory = types.iter().any(|t| {
        t.property_flags
            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE)
    });
    let features = unsafe { instance.get_physical_device_features(physical) };
    let integer_dot_product = unsafe { instance.enumerate_device_extension_properties(physical) }
        .unwrap_or_default()
        .iter()
        .any(|e| {
            e.extension_name_as_c_str()
                .is_ok_and(|n| n == ash::khr::shader_integer_dot_product::NAME)
        });
    let device_type = match props.device_type {
        vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated",
        vk::PhysicalDeviceType::DISCRETE_GPU => "discrete",
        vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual",
        vk::PhysicalDeviceType::CPU => "cpu",
        _ => "other",
    };
    VulkanAdapterInfo {
        index,
        name: cstr_name(&props.device_name),
        device_type,
        vendor_id: props.vendor_id,
        device_id: props.device_id,
        driver_name: cstr_name(&driver.driver_name),
        driver_info: cstr_name(&driver.driver_info),
        api_version: (
            vk::api_version_major(props.api_version),
            vk::api_version_minor(props.api_version),
            vk::api_version_patch(props.api_version),
        ),
        device_local_bytes,
        unified_memory,
        shader_float64: features.shader_float64 == vk::TRUE,
        integer_dot_product,
    }
}

/// Whether software Vulkan implementations (lavapipe) may be opened as a
/// device: off unless `CANDLE_VULKAN_ALLOW_CPU=1`, so that a missing GPU
/// driver is never mistaken for GPU acceleration.
fn allow_software_adapter() -> bool {
    std::env::var("CANDLE_VULKAN_ALLOW_CPU").is_ok_and(|v| v == "1")
}

/// Lists the Vulkan physical devices visible through the loader, in ordinal
/// order. Fails when no Vulkan loader can be loaded.
pub fn enumerate_adapters() -> Result<Vec<VulkanAdapterInfo>> {
    let _guard = INIT.lock().unwrap_or_else(|p| p.into_inner());
    let entry = unsafe { ash::Entry::load() }.map_err(|e| {
        Error::Msg(format!(
            "vulkan: failed to load libvulkan ({e}); is a Vulkan loader + ICD installed?"
        ))
    })?;
    let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);
    let instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::default().application_info(&app_info),
            None,
        )
    }
    .map_err(|e| Error::Msg(format!("vulkan: create_instance failed: {e:?}")))?;
    let out = match unsafe { instance.enumerate_physical_devices() } {
        Ok(list) => Ok(list
            .iter()
            .enumerate()
            .map(|(i, &p)| adapter_info(&instance, p, i))
            .collect()),
        Err(e) => Err(Error::Msg(format!(
            "vulkan: enumerate_physical_devices failed: {e:?}"
        ))),
    };
    unsafe { instance.destroy_instance(None) };
    out
}

fn init_vulkan(gpu_id: usize, qgemv: Option<kernels::Qgemv>) -> Result<VulkanDevice> {
    let _guard = INIT.lock().unwrap_or_else(|p| p.into_inner());
    let entry = unsafe { ash::Entry::load() }.map_err(|e| {
        Error::Msg(format!(
            "vulkan: failed to load libvulkan.so ({e}); is a Vulkan loader + ICD installed?"
        ))
    })?;
    let app_name = std::ffi::CString::new("candle-vulkan").unwrap();
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::API_VERSION_1_2);
    let instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::default().application_info(&app_info),
            None,
        )
    }
    .map_err(|e| Error::Msg(format!("vulkan: create_instance failed: {e:?}")))?;
    macro_rules! fail {
        ($($arg:tt)*) => {{
            unsafe { instance.destroy_instance(None) };
            return Err(Error::Msg(format!($($arg)*)));
        }};
    }
    let phys_list = match unsafe { instance.enumerate_physical_devices() } {
        Ok(ps) if !ps.is_empty() => ps,
        Ok(_) => fail!("vulkan: no physical device found"),
        Err(e) => fail!("vulkan: enumerate_physical_devices failed: {e:?}"),
    };
    let Some(&physical) = phys_list.get(gpu_id) else {
        fail!(
            "vulkan: gpu_id {gpu_id} out of range ({} physical device(s) available)",
            phys_list.len()
        )
    };
    let props = unsafe { instance.get_physical_device_properties(physical) };
    let name = cstr_name(&props.device_name);
    let info = adapter_info(&instance, physical, gpu_id);
    if info.is_software() && !allow_software_adapter() {
        fail!(
            "vulkan: gpu_id {gpu_id} ({name}) is a software (CPU) Vulkan implementation, not a GPU; \
             select a hardware device or set CANDLE_VULKAN_ALLOW_CPU=1 to use it anyway"
        )
    }
    let queues = unsafe { instance.get_physical_device_queue_family_properties(physical) };
    // Prefer a compute-only family (not shared with graphics), else any
    // compute-capable one.
    let queue_family = queues
        .iter()
        .position(|q| {
            q.queue_flags.contains(vk::QueueFlags::COMPUTE)
                && !q.queue_flags.contains(vk::QueueFlags::GRAPHICS)
        })
        .or_else(|| {
            queues
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
        });
    let Some(queue_family) = queue_family.map(|i| i as u32) else {
        fail!("vulkan: device {name} has no compute queue")
    };
    let priorities = [1.0f32];
    let qci = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family)
        .queue_priorities(&priorities)];
    // Packed int8 dot products (Mali's int8 GEMV): enabled when the device
    // offers VK_KHR_shader_integer_dot_product and its feature.
    let int_dot_ext = unsafe { instance.enumerate_device_extension_properties(physical) }
        .unwrap_or_default()
        .iter()
        .any(|e| {
            e.extension_name_as_c_str()
                .is_ok_and(|n| n == ash::khr::shader_integer_dot_product::NAME)
        });
    let int_dot = int_dot_ext && {
        let mut idp = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default();
        let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut idp);
        unsafe { instance.get_physical_device_features2(physical, &mut f2) };
        idp.shader_integer_dot_product == vk::TRUE
    };
    let ext_names = [ash::khr::shader_integer_dot_product::NAME.as_ptr()];
    let mut idp_on = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default()
        .shader_integer_dot_product(true);
    let f64_rms = kernels::f64_rms_enabled() || kernels::f64_dot_enabled();
    let supported = unsafe { instance.get_physical_device_features(physical) };
    if f64_rms && supported.shader_float64 != vk::TRUE {
        fail!("vulkan: float64 RMS requires shaderFloat64");
    }
    let shader_float64 = f64_rms
        || (VALIDATED_B50_PRECISE_MATH
            && props.vendor_id == 0x8086
            && props.device_id == 0xE212
            && supported.shader_float64 == vk::TRUE);
    let features = vk::PhysicalDeviceFeatures::default().shader_float64(shader_float64);
    let mut dci = vk::DeviceCreateInfo::default()
        .queue_create_infos(&qci)
        .enabled_features(&features);
    if int_dot {
        dci = dci
            .enabled_extension_names(&ext_names)
            .push_next(&mut idp_on);
    }
    let device = match unsafe { instance.create_device(physical, &dci, None) } {
        Ok(d) => d,
        Err(e) => fail!("vulkan: create_device on {name} failed: {e:?}"),
    };
    let queue = unsafe { device.get_device_queue(queue_family, 0) };

    // Memory type for tensors: host-visible + coherent, device-local when
    // available, compatible with a buffer of the usage every tensor buffer
    // declares (`alloc_raw`).  Each later allocation still checks its own
    // `memoryTypeBits` against the choice.
    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical) };
    let probe = unsafe {
        device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(4096)
                .usage(TENSOR_USAGE)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )
    };
    let probe = match probe {
        Ok(b) => b,
        Err(e) => {
            unsafe { device.destroy_device(None) };
            fail!("vulkan: create_buffer failed: {e:?}")
        }
    };
    let req = unsafe { device.get_buffer_memory_requirements(probe) };
    unsafe { device.destroy_buffer(probe, None) };
    let pick = |want: vk::MemoryPropertyFlags| {
        mem_props.memory_types[..mem_props.memory_type_count as usize]
            .iter()
            .enumerate()
            .find(|(i, mt)| {
                (req.memory_type_bits & (1 << i)) != 0 && mt.property_flags.contains(want)
            })
            .map(|(i, _)| i as u32)
    };
    let hv = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let (mem_type, host_unified) = match pick(hv | vk::MemoryPropertyFlags::DEVICE_LOCAL) {
        Some(i) => (i, true),
        None => match pick(hv) {
            Some(i) => (i, false),
            None => {
                unsafe { device.destroy_device(None) };
                fail!("vulkan: {name} exposes no HOST_VISIBLE|HOST_COHERENT memory type")
            }
        },
    };
    let lim = &props.limits;
    let max_inv = lim.max_compute_work_group_invocations.max(64) as usize;
    let mut wg = 1usize;
    while wg * 2 <= max_inv.min(256) {
        wg *= 2;
    }
    let limits = Limits {
        wg,
        gemm_tx: if max_inv >= 256 { 16 } else { 8 },
        max_storage_range: lim.max_storage_buffer_range as u64,
        max_groups_x: lim.max_compute_work_group_count[0].max(1),
        host_unified,
        qgemv: qgemv.unwrap_or_else(|| {
            kernels::Qgemv::for_adapter(props.vendor_id, props.device_id, int_dot)
        }),
        int_dot,
    };
    let exec = match kernels::Exec::new(&device, queue, queue_family, mem_type) {
        Ok(e) => e,
        Err(e) => {
            unsafe { device.destroy_device(None) };
            fail!("{e}")
        }
    };
    let device = VulkanDevice {
        gpu_id,
        precise_math: false,
        inner: Arc::new(VulkanContext {
            entry,
            instance,
            device,
            name,
            info,
            adapter_id: (props.vendor_id, props.device_id),
            shader_float64,
            limits,
            buffer_align: req.alignment.max(16),
            mem: Mutex::new(Allocator {
                blocks: Vec::new(),
                chunk: CHUNK_BYTES,
                mem_type,
            }),
            quant_gemm_scratch: Mutex::new(None),
            exec: std::mem::ManuallyDrop::new(Mutex::new(exec)),
            pipes: Mutex::new(HashMap::new()),
            layouts: Mutex::new(HashMap::new()),
        }),
    };
    let weak = Arc::downgrade(&device.inner);
    crate::fault_slot::register_drain(Box::new(move |slot| match weak.upgrade() {
        Some(inner) => {
            // A throwaway handle: flush the pending batch, then clear the slot.
            let d = VulkanDevice {
                gpu_id: 0,
                inner,
                precise_math: false,
            };
            let _ = d.flush();
            d.exec().clear_fault(slot);
            true
        }
        None => false,
    }));
    Ok(device)
}

impl VulkanDevice {
    pub fn adapter_id(&self) -> (u32, u32) {
        self.inner.adapter_id
    }
    pub fn precise_math(&self) -> bool {
        self.precise_math
    }
    pub fn supports_precise_math(&self) -> bool {
        self.inner.shader_float64
    }

    /// A model-scoped arithmetic handle over the same logical device and
    /// allocator. Existing handles and tensors retain their original policy.
    pub fn with_precise_math(&self) -> Result<Self> {
        if !self.inner.shader_float64 {
            return Err(Error::Msg(
                "precise Vulkan arithmetic requires shaderFloat64".into(),
            ));
        }
        Ok(Self {
            gpu_id: self.gpu_id,
            inner: self.inner.clone(),
            precise_math: true,
        })
    }
    /// The device with ordinal `gpu_id`.
    ///
    /// Handles are shared per ordinal: while any handle (or tensor) for
    /// `gpu_id` is alive, `new(gpu_id)` returns another handle on the same
    /// logical device, so tensors created through separate `new` calls can be
    /// combined. When every handle is dropped the device and its memory are
    /// released. Use [`VulkanDevice::new_isolated`] for a private context.
    pub fn new(gpu_id: usize) -> Result<Self> {
        static SHARED: Mutex<Vec<(usize, std::sync::Weak<VulkanContext>)>> = Mutex::new(Vec::new());
        let mut shared = SHARED.lock().unwrap_or_else(|p| p.into_inner());
        shared.retain(|(_, w)| w.strong_count() > 0);
        if let Some(inner) = shared
            .iter()
            .find(|(id, _)| *id == gpu_id)
            .and_then(|(_, w)| w.upgrade())
        {
            return Ok(VulkanDevice {
                gpu_id,
                inner,
                precise_math: false,
            });
        }
        let dev = init_vulkan(gpu_id, None)?;
        shared.push((gpu_id, Arc::downgrade(&dev.inner)));
        Ok(dev)
    }

    /// A new, private logical device for `gpu_id` (not shared with [`VulkanDevice::new`]).
    pub fn new_isolated(gpu_id: usize) -> Result<Self> {
        init_vulkan(gpu_id, None)
    }

    /// Explicit kernel selection for parity tests and device benchmarks.
    pub fn new_with_qgemv(gpu_id: usize, kernel: kernels::Qgemv) -> Result<Self> {
        init_vulkan(gpu_id, Some(kernel))
    }

    /// Current sub-allocator usage, including allocations awaiting batch
    /// completion. Synchronize first for a quiescent snapshot. The command
    /// ring and driver-owned pipeline memory are fixed/separate from this pool.
    pub fn memory_stats(&self) -> (u64, u64) {
        let mem = self.inner.mem.lock().unwrap_or_else(|p| p.into_inner());
        let mut reserved = 0;
        let mut used = 0;
        for b in mem.blocks.iter().flatten() {
            reserved += b.size;
            used += b.size - b.free.iter().map(|(_, size)| size).sum::<u64>();
        }
        (reserved, used)
    }

    pub fn new_with_stream(gpu_id: usize) -> Result<Self> {
        Self::new(gpu_id)
    }

    pub fn id(&self) -> DeviceId {
        DeviceId(self.gpu_id)
    }

    /// `VkPhysicalDeviceProperties::deviceName`.
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// Driver, API version, memory and capability details of this device.
    pub fn adapter_info(&self) -> &VulkanAdapterInfo {
        &self.inner.info
    }

    /// Whether tensor memory is unified (device-local and host-visible).
    pub fn host_unified_memory(&self) -> bool {
        self.inner.limits.host_unified
    }

    pub fn limits(&self) -> Limits {
        self.inner.limits
    }

    pub(crate) fn ash(&self) -> &ash::Device {
        &self.inner.device
    }

    pub(crate) fn exec(&self) -> MutexGuard<'_, kernels::Exec> {
        self.inner.exec.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// The device's fault word, bound last by every indexing kernel.
    pub(crate) fn fault_buf(&self) -> kernels::Buf {
        self.exec().fault()
    }

    /// Report (and clear) an out-of-range id an indexing kernel launched by
    /// this thread flagged in the batches completed so far.  Called at
    /// every host read-back and `synchronize`, the points where the CPU
    /// backend's error for the same input would have been observed.
    pub fn check_fault(&self) -> Result<()> {
        if self.exec().take_fault() {
            return Err(Error::Msg(
                "vulkan: an index_select / gather / scatter / index_add / embedding id was out of range for the indexed dimension (reported at the next host read-back)".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn pipes(&self) -> MutexGuard<'_, kernels::PipeMap> {
        self.inner.pipes.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub(crate) fn layouts(&self, nbuf: u32) -> Result<Arc<kernels::Layouts>> {
        let mut map = self.inner.layouts.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(l) = map.get(&nbuf) {
            return Ok(l.clone());
        }
        let l = Arc::new(kernels::create_layouts(&self.inner.device, nbuf)?);
        map.insert(nbuf, l.clone());
        Ok(l)
    }

    /// Submit and wait for the pending batch, then release the buffers it
    /// referenced.
    pub fn flush(&self) -> Result<()> {
        // Exclude another dequantize/GEMM pair until this batch has completed
        // and its scratch has been retired. Ordinary decode never creates it.
        let mut scratch = self
            .inner
            .quant_gemm_scratch
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let mut exec = self.exec();
        let mut garbage = exec.flush()?;
        exec.batch_limit = kernels::MAX_LAUNCHES;
        drop(exec);
        if let Some(s) = scratch.take() {
            garbage.push(kernels::Garbage {
                buffer: s.buffer,
                alloc: s.alloc,
            });
        }
        self.free_garbage(garbage);
        Ok(())
    }

    pub(crate) fn free_garbage(&self, garbage: Vec<kernels::Garbage>) {
        if garbage.is_empty() {
            return;
        }
        let mut mem = self.inner.mem.lock().unwrap_or_else(|p| p.into_inner());
        for g in garbage {
            unsafe { self.inner.device.destroy_buffer(g.buffer, None) };
            mem.free(&self.inner.device, g.alloc);
        }
    }

    /// Release a storage's buffer: at once when nothing is recorded, else
    /// after the open batch completes.
    fn release(&self, buffer: vk::Buffer, alloc: Alloc) {
        let mut exec = self.exec();
        if exec.recording {
            exec.garbage.push(kernels::Garbage { buffer, alloc });
            return;
        }
        drop(exec);
        let mut mem = self.inner.mem.lock().unwrap_or_else(|p| p.into_inner());
        unsafe { self.inner.device.destroy_buffer(buffer, None) };
        mem.free(&self.inner.device, alloc);
    }

    /// Allocate an uninitialised buffer for `numel` elements of `dtype`.
    /// The capacity is rounded up to whole words so the packed 1/2-byte
    /// kernels can own every word they write.
    pub fn alloc(&self, dtype: DType, numel: usize) -> Result<VulkanStorage> {
        let bytes = dtype
            .size_in_bytes()
            .checked_mul(numel)
            .ok_or_else(|| Error::Msg("vulkan: overflow in storage size".into()))?;
        let capacity = bytes.div_ceil(4).max(1) * 4;
        let (buffer, alloc, mapped) = self.alloc_raw(capacity as u64)?;
        Ok(VulkanStorage {
            buffer,
            alloc,
            mapped,
            capacity_bytes: capacity,
            dtype,
            numel,
            device: self.clone(),
        })
    }

    fn alloc_raw(&self, capacity: u64) -> Result<(vk::Buffer, Alloc, *mut u8)> {
        let dev = &self.inner.device;
        let buffer = unsafe {
            dev.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(capacity)
                    .usage(TENSOR_USAGE)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }
        .map_err(|e| Error::Msg(format!("vulkan create_buffer failed: {e:?}")))?;
        let req = unsafe { dev.get_buffer_memory_requirements(buffer) };
        let mut mem = self.inner.mem.lock().unwrap_or_else(|p| p.into_inner());
        if req.memory_type_bits & (1 << mem.mem_type) == 0 {
            unsafe { dev.destroy_buffer(buffer, None) };
            return Err(Error::Msg(format!(
                "vulkan: a {capacity}-byte tensor buffer cannot use memory type {} (memoryTypeBits {:#x})",
                mem.mem_type, req.memory_type_bits
            )));
        }
        let (alloc, mapped) =
            match mem.alloc(dev, req.size, req.alignment.max(self.inner.buffer_align)) {
                Ok(a) => a,
                Err(e) => {
                    unsafe { dev.destroy_buffer(buffer, None) };
                    return Err(e);
                }
            };
        let memory = mem.blocks[alloc.block]
            .as_ref()
            .map(|b| b.memory)
            .unwrap_or(vk::DeviceMemory::null());
        if let Err(e) = unsafe { dev.bind_buffer_memory(buffer, memory, alloc.offset) } {
            unsafe { dev.destroy_buffer(buffer, None) };
            mem.free(dev, alloc);
            return Err(Error::Msg(format!(
                "vulkan bind_buffer_memory failed: {e:?}"
            )));
        }
        Ok((buffer, alloc, mapped))
    }
}

// ─── Storage ─────────────────────────────────────────────────────────────────

/// A Vulkan storage: a `VkBuffer` (+ its mapped host pointer) + dtype +
/// element count.
pub struct VulkanStorage {
    pub buffer: vk::Buffer,
    alloc: Alloc,
    pub mapped: *mut u8,
    pub capacity_bytes: usize,
    pub dtype: DType,
    pub numel: usize,
    pub device: VulkanDevice,
}

impl std::fmt::Debug for VulkanStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanStorage")
            .field("capacity_bytes", &self.capacity_bytes)
            .field("dtype", &self.dtype)
            .field("numel", &self.numel)
            .field("device", &self.device)
            .finish()
    }
}

unsafe impl Send for VulkanStorage {}
unsafe impl Sync for VulkanStorage {}

impl Drop for VulkanStorage {
    fn drop(&mut self) {
        self.device.release(self.buffer, self.alloc);
    }
}

impl VulkanStorage {
    pub fn transfer_to_device(&self, dst: &VulkanDevice) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        dst.storage_from_cpu_storage(&cpu)
    }

    pub fn from_vec<T: crate::WithDType>(slice: Vec<T>, device: &VulkanDevice) -> Result<Self> {
        Self::from_slice(&slice, device)
    }

    fn from_slice<T: crate::WithDType>(slice: &[T], device: &VulkanDevice) -> Result<Self> {
        let dtype = T::DTYPE;
        let storage = device.alloc(dtype, slice.len())?;
        let bytes = dtype.size_in_bytes() * slice.len();
        // # Safety: `slice` holds `bytes` valid bytes.
        let data = unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, bytes) };
        unsafe { storage.set_bytes(data) }?;
        Ok(storage)
    }

    /// The buffer as a kernel argument.
    pub fn buf(&self) -> Buf {
        Buf {
            buffer: self.buffer,
            bytes: self.capacity_bytes as u64,
        }
    }

    fn elem_size(&self) -> usize {
        self.dtype.size_in_bytes()
    }

    /// Copy `src` into the mapped buffer (a fresh storage: nothing recorded
    /// reads it yet, and submission makes host writes visible).
    ///
    /// # Safety
    /// `src` must be at most `self.capacity_bytes` long.
    pub(crate) unsafe fn set_bytes(&self, src: &[u8]) -> Result<()> {
        if src.len() > self.capacity_bytes {
            return Err(Error::Msg(format!(
                "vulkan set_bytes: {} bytes exceed capacity {}",
                src.len(),
                self.capacity_bytes
            )));
        }
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), self.mapped, src.len()) };
        kernels::note_h2d(src.len());
        Ok(())
    }

    /// Read `bytes` from the buffer after completing the pending batch.
    pub(crate) fn get_bytes(&self, bytes: usize) -> Result<Vec<u8>> {
        self.device.flush()?;
        self.device.check_fault()?;
        let n = bytes.min(self.capacity_bytes);
        let mut out = vec![0u8; n];
        unsafe { std::ptr::copy_nonoverlapping(self.mapped, out.as_mut_ptr(), n) };
        kernels::note_d2h(n);
        Ok(out)
    }

    /// A contiguous copy of the elements addressed by `l`, on the device.
    fn contiguous_copy(&self, l: &Layout) -> Result<Self> {
        let n = l.shape().elem_count();
        let out = self.device.alloc(self.dtype, n)?;
        kernels::run_copy_strided(
            &self.device,
            self.elem_size(),
            self.buf(),
            out.buf(),
            n,
            l,
            0,
        )?;
        Ok(out)
    }

    /// `(storage, offset)`: the storage itself when `l` is contiguous
    /// (with its start offset), otherwise a contiguous copy at offset 0.
    fn as_contiguous(&self, l: &Layout) -> Result<(std::borrow::Cow<'_, Self>, usize)> {
        match l.contiguous_offsets() {
            Some((o1, _)) => Ok((std::borrow::Cow::Borrowed(self), o1)),
            None => Ok((std::borrow::Cow::Owned(self.contiguous_copy(l)?), 0)),
        }
    }

    /// u32 ids from a u32 / i64 / u8 id storage (cast on the device when
    /// needed); returns the storage and the offset of its first element.
    fn ids_u32(&self, l: &Layout) -> Result<(std::borrow::Cow<'_, Self>, usize)> {
        match self.dtype {
            DType::U32 => self.as_contiguous(l),
            DType::I64 | DType::U8 => {
                let n = l.shape().elem_count();
                let out = self.device.alloc(DType::U32, n)?;
                // i64 ids saturate (negative / beyond u32 → u32::MAX) so the
                // consuming kernel's bounds check faults instead of a
                // truncated id selecting the wrong element.
                if self.dtype == DType::I64 {
                    kernels::run_ids_i64(&self.device, self.buf(), out.buf(), n, l)?;
                } else {
                    kernels::run_cast(
                        &self.device,
                        self.dtype,
                        DType::U32,
                        self.buf(),
                        out.buf(),
                        n,
                        l,
                    )?;
                }
                Ok((std::borrow::Cow::Owned(out), 0))
            }
            d => Err(Error::Msg(format!("vulkan: unsupported index dtype {d:?}"))),
        }
    }
}

impl Clone for VulkanStorage {
    /// Whole-buffer device-to-device copy.
    fn clone(&self) -> Self {
        self.try_clone(&Layout::contiguous(self.numel))
            .expect("vulkan: device buffer copy failed")
    }
}

fn transmute_bytes<T: Copy>(raw: &[u8], numel: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(numel);
    let size = std::mem::size_of::<T>();
    for i in 0..numel {
        // # Safety: raw holds numel * size bytes, T is plain data.
        let v = unsafe { std::ptr::read_unaligned(raw.as_ptr().add(i * size) as *const T) };
        out.push(v);
    }
    out
}

fn native() -> bool {
    kernels::native_enabled()
}

/// Strides of a matmul operand `[batch..., rows, cols]` under `l`, when the
/// batch dims collapse to one stride.  `None` means the operand must be
/// materialised.
fn mat_strides(l: &Layout, batch: usize) -> Option<MatStrides> {
    let dims = l.dims();
    let st = l.stride();
    let r = dims.len();
    if r < 2 {
        return None;
    }
    let row = st[r - 2];
    let col = st[r - 1];
    let batch_stride = if r == 2 {
        0
    } else {
        let bdims = &dims[..r - 2];
        let bst = &st[..r - 2];
        if bdims.iter().product::<usize>() != batch {
            return None;
        }
        if bst.iter().all(|&s| s == 0) {
            0
        } else {
            let inner = bst[bdims.len() - 1];
            for i in 0..bdims.len() - 1 {
                if bdims[i] > 1 && bst[i] != bdims[i + 1] * bst[i + 1] {
                    return None;
                }
            }
            inner
        }
    };
    Some(MatStrides {
        row,
        col,
        offset: l.start_offset(),
        batch: batch_stride,
    })
}

fn scalar_bits(s: crate::scalar::Scalar) -> Option<u64> {
    use crate::scalar::Scalar::*;
    Some(match s {
        U8(v) => v as u64,
        U32(v) => v as u64,
        I16(v) => v as u16 as u64,
        I32(v) => v as u32 as u64,
        I64(v) => v as u64,
        BF16(v) => v.to_bits() as u64,
        F16(v) => v.to_bits() as u64,
        F32(v) => v.to_bits() as u64,
        F64(v) => v.to_bits(),
        _ => return None,
    })
}

impl BackendStorage for VulkanStorage {
    type Device = VulkanDevice;

    fn try_clone(&self, _layout: &Layout) -> Result<Self> {
        let out = self.device.alloc(self.dtype, self.numel)?;
        if self.numel > 0 {
            // Copy whole words (the capacity is word-padded).
            let words = self.capacity_bytes / 4;
            let l = Layout::contiguous(words);
            kernels::run_copy_strided(&self.device, 4, self.buf(), out.buf(), words, &l, 0)?;
        }
        Ok(out)
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn const_set(&mut self, s: crate::scalar::Scalar, layout: &Layout) -> Result<()> {
        if native() {
            if let Some(bits) = scalar_bits(s) {
                let n = layout.shape().elem_count();
                match kernels::run_fill(&self.device, self.elem_size(), self.buf(), n, layout, bits)
                {
                    Ok(()) => return Ok(()),
                    Err(e) => kernels::note_fallback("const_set", Some(&e)),
                }
            } else {
                kernels::note_fallback("const_set", None);
            }
        }
        let mut cpu = self.to_cpu_storage()?;
        cpu.const_set(s, layout)?;
        let dev = self.device.clone();
        *self = dev.storage_from_cpu_storage(&cpu)?;
        Ok(())
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        let bytes = self.elem_size() * self.numel;
        let raw = self.get_bytes(bytes)?;
        Ok(match self.dtype {
            DType::U8 => CpuStorage::U8(raw),
            DType::U32 => CpuStorage::U32(transmute_bytes(&raw, self.numel)),
            DType::I16 => CpuStorage::I16(transmute_bytes(&raw, self.numel)),
            DType::I32 => CpuStorage::I32(transmute_bytes(&raw, self.numel)),
            DType::I64 => CpuStorage::I64(transmute_bytes(&raw, self.numel)),
            DType::F32 => CpuStorage::F32(transmute_bytes(&raw, self.numel)),
            DType::F64 => CpuStorage::F64(transmute_bytes(&raw, self.numel)),
            DType::F16 => CpuStorage::F16(
                transmute_bytes::<u16>(&raw, self.numel)
                    .into_iter()
                    .map(half::f16::from_bits)
                    .collect(),
            ),
            DType::BF16 => CpuStorage::BF16(
                transmute_bytes::<u16>(&raw, self.numel)
                    .into_iter()
                    .map(half::bf16::from_bits)
                    .collect(),
            ),
            other => {
                return Err(Error::Msg(format!(
                    "vulkan to_cpu_storage: dtype {other:?} not supported"
                )))
            }
        })
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        if native() && self.dtype == DType::F32 {
            let n = layout.shape().elem_count();
            let out = self.device.alloc(DType::F32, n)?;
            match kernels::run_affine(
                &self.device,
                self.buf(),
                out.buf(),
                n,
                layout,
                mul as f32,
                add as f32,
            ) {
                Ok(()) => return Ok(out),
                Err(e) => kernels::note_fallback("affine", Some(&e)),
            }
        } else if native() {
            kernels::note_fallback("affine", None);
        }
        let cpu = self.to_cpu_storage()?;
        let out = cpu.affine(layout, mul, add)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        if native() && self.dtype == DType::F32 {
            let n = layout.shape().elem_count();
            let out = self.device.alloc(DType::F32, n)?;
            match kernels::run_powf(&self.device, self.buf(), out.buf(), n, layout, e as f32) {
                Ok(()) => return Ok(out),
                Err(err) => kernels::note_fallback("powf", Some(&err)),
            }
        } else if native() {
            kernels::note_fallback("powf", None);
        }
        let cpu = self.to_cpu_storage()?;
        let out = cpu.powf(layout, e)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        if native() && self.dtype == DType::F32 {
            let n = layout.shape().elem_count();
            let out = self.device.alloc(DType::F32, n)?;
            match kernels::run_elu(&self.device, self.buf(), out.buf(), n, layout, alpha as f32) {
                Ok(()) => return Ok(out),
                Err(err) => kernels::note_fallback("elu", Some(&err)),
            }
        } else if native() {
            kernels::note_fallback("elu", None);
        }
        let cpu = self.to_cpu_storage()?;
        let out = cpu.elu(layout, alpha)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, s: &[usize]) -> Result<Self> {
        if native() && self.dtype == DType::F32 {
            match self.reduce_native(op, layout, s) {
                Ok(Some(out)) => return Ok(out),
                Ok(None) => kernels::note_fallback("reduce_op", None),
                Err(e) => kernels::note_fallback("reduce_op", Some(&e)),
            }
        } else if native() {
            kernels::note_fallback("reduce_op", None);
        }
        let cpu = self.to_cpu_storage()?;
        let out = cpu.reduce_op(op, layout, s)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        if native() && self.dtype == rhs.dtype && matches!(self.dtype, DType::F32 | DType::U32) {
            let n = lhs_l.shape().elem_count();
            let out = self.device.alloc(DType::U8, n)?;
            match kernels::run_cmp(
                &self.device,
                kernels::cmp_code(op),
                self.buf(),
                rhs.buf(),
                out.buf(),
                n,
                lhs_l,
                rhs_l,
                self.dtype == DType::U32,
            ) {
                Ok(()) => return Ok(out),
                Err(e) => kernels::note_fallback("cmp", Some(&e)),
            }
        } else if native() {
            kernels::note_fallback("cmp", None);
        }
        let lhs = self.to_cpu_storage()?;
        let rhs = rhs.to_cpu_storage()?;
        let out = lhs.cmp(op, &rhs, lhs_l, rhs_l)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        if native() {
            if dtype == self.dtype {
                return self.contiguous_copy(layout);
            }
            if kernels::cast_supported(self.dtype, dtype) {
                let n = layout.shape().elem_count();
                let out = self.device.alloc(dtype, n)?;
                match kernels::run_cast(
                    &self.device,
                    self.dtype,
                    dtype,
                    self.buf(),
                    out.buf(),
                    n,
                    layout,
                ) {
                    Ok(()) => return Ok(out),
                    Err(e) => kernels::note_fallback("to_dtype", Some(&e)),
                }
            } else {
                kernels::note_fallback("to_dtype", None);
            }
        }
        let cpu = self.to_cpu_storage()?;
        let out = cpu.to_dtype(layout, dtype)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        if native() && self.dtype == DType::F32 {
            if let Some(code) = kernels::unary_code(B::NAME) {
                let n = layout.shape().elem_count();
                let out = self.device.alloc(DType::F32, n)?;
                match kernels::run_unary(&self.device, code, self.buf(), out.buf(), n, layout) {
                    Ok(()) => return Ok(out),
                    Err(e) => kernels::note_fallback(B::NAME, Some(&e)),
                }
            } else {
                kernels::note_fallback(B::NAME, None);
            }
        } else if native() {
            kernels::note_fallback(B::NAME, None);
        }
        let cpu = self.to_cpu_storage()?;
        let out = cpu.unary_impl::<B>(layout)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        if native() && self.dtype == rhs.dtype && matches!(self.dtype, DType::F32 | DType::U32) {
            if let Some(code) = kernels::binary_code(B::NAME) {
                let n = lhs_l.shape().elem_count();
                let out = self.device.alloc(self.dtype, n)?;
                match kernels::run_binary(
                    &self.device,
                    code,
                    self.buf(),
                    rhs.buf(),
                    out.buf(),
                    n,
                    lhs_l,
                    rhs_l,
                    self.dtype == DType::U32,
                ) {
                    Ok(()) => return Ok(out),
                    Err(e) => kernels::note_fallback(B::NAME, Some(&e)),
                }
            } else {
                kernels::note_fallback(B::NAME, None);
            }
        } else if native() {
            kernels::note_fallback(B::NAME, None);
        }
        let lhs = self.to_cpu_storage()?;
        let rhs = rhs.to_cpu_storage()?;
        let out = lhs.binary_impl::<B>(&rhs, lhs_l, rhs_l)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        if native()
            && t.dtype == f.dtype
            && matches!(self.dtype, DType::U8 | DType::U32)
            && matches!(t.elem_size(), 4 | 8)
        {
            let n = layout.shape().elem_count();
            let out = self.device.alloc(t.dtype, n)?;
            match kernels::run_where(
                &self.device,
                self.buf(),
                t.buf(),
                f.buf(),
                out.buf(),
                n,
                layout,
                t_l,
                f_l,
                self.dtype == DType::U32,
                t.elem_size() == 8,
            ) {
                Ok(()) => return Ok(out),
                Err(e) => kernels::note_fallback("where_cond", Some(&e)),
            }
        } else if native() {
            kernels::note_fallback("where_cond", None);
        }
        let cond = self.to_cpu_storage()?;
        let t = t.to_cpu_storage()?;
        let f = f.to_cpu_storage()?;
        let out = cond.where_cond(layout, &t, t_l, &f, f_l)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        if native() {
            kernels::note_fallback("conv1d", None);
        }
        let inp = self.to_cpu_storage()?;
        let kernel = kernel.to_cpu_storage()?;
        let out = inp.conv1d(l, &kernel, kernel_l, params)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        if native() {
            kernels::note_fallback("conv_transpose1d", None);
        }
        let inp = self.to_cpu_storage()?;
        let kernel = kernel.to_cpu_storage()?;
        let out = inp.conv_transpose1d(l, &kernel, kernel_l, params)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        if native() {
            kernels::note_fallback("conv2d", None);
        }
        let inp = self.to_cpu_storage()?;
        let kernel = kernel.to_cpu_storage()?;
        let out = inp.conv2d(l, &kernel, kernel_l, params)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        if native() {
            kernels::note_fallback("conv_transpose2d", None);
        }
        let inp = self.to_cpu_storage()?;
        let kernel = kernel.to_cpu_storage()?;
        let out = inp.conv_transpose2d(l, &kernel, kernel_l, params)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn index_select(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        if native()
            && matches!(self.elem_size(), 4 | 8)
            && matches!(ids.dtype, DType::U32 | DType::I64 | DType::U8)
        {
            match self.index_select_native(ids, l, ids_l, dim) {
                Ok(out) => return Ok(out),
                Err(e) => kernels::note_fallback("index_select", Some(&e)),
            }
        } else if native() {
            kernels::note_fallback("index_select", None);
        }
        let src = self.to_cpu_storage()?;
        let ids = ids.to_cpu_storage()?;
        let out = src.index_select(&ids, l, ids_l, dim)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        if native()
            && self.elem_size() == 4
            && matches!(ids.dtype, DType::U32 | DType::I64 | DType::U8)
        {
            match self.gather_native(l, ids, ids_l, dim) {
                Ok(out) => return Ok(out),
                Err(e) => kernels::note_fallback("gather", Some(&e)),
            }
        } else if native() {
            kernels::note_fallback("gather", None);
        }
        let src = self.to_cpu_storage()?;
        let ids = ids.to_cpu_storage()?;
        let out = src.gather(l, &ids, ids_l, dim)?;
        self.device.storage_from_cpu_storage(&out)
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
        if native() && self.elem_size() == 4 && src.dtype == self.dtype && l.is_contiguous() {
            match self.scatter_native(false, l, ids, ids_l, src, src_l, dim) {
                Ok(()) => return Ok(()),
                Err(e) => kernels::note_fallback("scatter_set", Some(&e)),
            }
        } else if native() {
            kernels::note_fallback("scatter_set", None);
        }
        let mut tgt = self.to_cpu_storage()?;
        let ids = ids.to_cpu_storage()?;
        let src = src.to_cpu_storage()?;
        tgt.scatter_set(l, &ids, ids_l, &src, src_l, dim)?;
        let dev = self.device.clone();
        *self = dev.storage_from_cpu_storage(&tgt)?;
        Ok(())
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
        if native() && self.dtype == DType::F32 && src.dtype == DType::F32 && l.is_contiguous() {
            match self.scatter_native(true, l, ids, ids_l, src, src_l, dim) {
                Ok(()) => return Ok(()),
                Err(e) => kernels::note_fallback("scatter_add", Some(&e)),
            }
        } else if native() {
            kernels::note_fallback("scatter_add", None);
        }
        let mut tgt = self.to_cpu_storage()?;
        let ids = ids.to_cpu_storage()?;
        let src = src.to_cpu_storage()?;
        tgt.scatter_add_set(l, &ids, ids_l, &src, src_l, dim)?;
        let dev = self.device.clone();
        *self = dev.storage_from_cpu_storage(&tgt)?;
        Ok(())
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
        if native() && self.dtype == DType::F32 && src.dtype == DType::F32 {
            match self.index_add_native(l, ids, ids_l, src, src_l, dim) {
                Ok(out) => return Ok(out),
                Err(e) => kernels::note_fallback("index_add", Some(&e)),
            }
        } else if native() {
            kernels::note_fallback("index_add", None);
        }
        let tgt = self.to_cpu_storage()?;
        let ids = ids.to_cpu_storage()?;
        let src = src.to_cpu_storage()?;
        let out = tgt.index_add(l, &ids, ids_l, &src, src_l, dim)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        if native() && self.dtype == DType::F32 && rhs.dtype == DType::F32 {
            match self.matmul_native(rhs, bmnk, lhs_l, rhs_l) {
                Ok(out) => return Ok(out),
                Err(e) => kernels::note_fallback("matmul", Some(&e)),
            }
        } else if native() {
            kernels::note_fallback("matmul", None);
        }
        let lhs = self.to_cpu_storage()?;
        let rhs = rhs.to_cpu_storage()?;
        let out = lhs.matmul(&rhs, bmnk, lhs_l, rhs_l)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        if native() && dst.dtype == self.dtype {
            let n = src_l.shape().elem_count();
            match kernels::run_copy_strided(
                &self.device,
                self.elem_size(),
                self.buf(),
                dst.buf(),
                n,
                src_l,
                dst_offset,
            ) {
                Ok(()) => return Ok(()),
                Err(e) => kernels::note_fallback("copy_strided_src", Some(&e)),
            }
        } else if native() {
            kernels::note_fallback("copy_strided_src", None);
        }
        let src = self.to_cpu_storage()?;
        let mut dst_cpu = dst.to_cpu_storage()?;
        src.copy_strided_src(&mut dst_cpu, dst_offset, src_l)?;
        let dev = dst.device.clone();
        *dst = dev.storage_from_cpu_storage(&dst_cpu)?;
        Ok(())
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_s: usize,
        dst_s: usize,
        src_o: usize,
        dst_o: usize,
    ) -> Result<()> {
        if native() && dst.dtype == self.dtype {
            match kernels::run_copy2d(
                &self.device,
                self.elem_size(),
                self.buf(),
                dst.buf(),
                d1,
                d2,
                src_s,
                dst_s,
                src_o,
                dst_o,
            ) {
                Ok(()) => return Ok(()),
                Err(e) => kernels::note_fallback("copy2d", Some(&e)),
            }
        } else if native() {
            kernels::note_fallback("copy2d", None);
        }
        let src = self.to_cpu_storage()?;
        let mut dst_cpu = dst.to_cpu_storage()?;
        src.copy2d(&mut dst_cpu, d1, d2, src_s, dst_s, src_o, dst_o)?;
        let dev = dst.device.clone();
        *dst = dev.storage_from_cpu_storage(&dst_cpu)?;
        Ok(())
    }

    fn avg_pool2d(&self, layout: &Layout, k: (usize, usize), s: (usize, usize)) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        let out = cpu.avg_pool2d(layout, k, s)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn max_pool2d(&self, layout: &Layout, k: (usize, usize), s: (usize, usize)) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        let out = cpu.max_pool2d(layout, k, s)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn upsample_nearest1d(&self, layout: &Layout, sz: usize) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        let out = cpu.upsample_nearest1d(layout, sz)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn upsample_nearest2d(&self, layout: &Layout, h: usize, w: usize) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        let out = cpu.upsample_nearest2d(layout, h, w)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn upsample_bilinear2d(
        &self,
        layout: &Layout,
        h: usize,
        w: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        let out = cpu.upsample_bilinear2d(layout, h, w, align_corners, scale_h, scale_w)?;
        self.device.storage_from_cpu_storage(&out)
    }
}

// ─── Native op bodies ────────────────────────────────────────────────────────

impl VulkanStorage {
    fn reduce_native(&self, op: ReduceOp, layout: &Layout, s: &[usize]) -> Result<Option<Self>> {
        let dims = layout.dims();
        let rank = dims.len();
        let code = match op {
            ReduceOp::Sum => kernels::RED_SUM,
            ReduceOp::Max => kernels::RED_MAX,
            ReduceOp::Min => kernels::RED_MIN,
            ReduceOp::ArgMax | ReduceOp::ArgMin => {
                if rank == 0 || s != [rank - 1] {
                    return Ok(None);
                }
                let Some((o1, _)) = layout.contiguous_offsets() else {
                    return Ok(None);
                };
                let cols = dims[rank - 1];
                let rows = dims[..rank - 1].iter().product::<usize>();
                if cols == 0 {
                    return Ok(None);
                }
                let out = self.device.alloc(DType::U32, rows)?;
                kernels::run_arg_last(
                    &self.device,
                    matches!(op, ReduceOp::ArgMax),
                    self.buf(),
                    out.buf(),
                    rows,
                    cols,
                    o1,
                )?;
                return Ok(Some(out));
            }
        };
        let mut reduced = s.to_vec();
        reduced.sort_unstable();
        reduced.dedup();
        if reduced.iter().any(|&d| d >= rank) {
            return Ok(None);
        }
        let out_dims: Vec<usize> = dims
            .iter()
            .enumerate()
            .map(|(i, &d)| if reduced.contains(&i) { 1 } else { d })
            .collect();
        let n_out = out_dims.iter().product::<usize>();
        if let Some((o1, _)) = layout.contiguous_offsets() {
            let n_trailing = reduced.len();
            let trailing_ok = reduced
                .iter()
                .enumerate()
                .all(|(i, &d)| d == rank - n_trailing + i);
            if trailing_ok && n_trailing > 0 {
                let cols = dims[rank - n_trailing..].iter().product::<usize>();
                let rows = n_out;
                if cols == 0 && code != kernels::RED_SUM {
                    return Ok(None);
                }
                let out = self.device.alloc(DType::F32, rows)?;
                if cols == 0 {
                    kernels::run_fill(
                        &self.device,
                        4,
                        out.buf(),
                        rows,
                        &Layout::contiguous(rows),
                        0,
                    )?;
                } else {
                    kernels::run_reduce_last(
                        &self.device,
                        code,
                        self.buf(),
                        out.buf(),
                        rows,
                        cols,
                        o1,
                    )?;
                }
                return Ok(Some(out));
            }
        }
        let count = reduced.iter().map(|&d| dims[d]).product::<usize>();
        if count == 0 && code != kernels::RED_SUM {
            return Ok(None);
        }
        let st = layout.stride();
        let out_strides: Vec<usize> = st
            .iter()
            .enumerate()
            .map(|(i, &v)| if reduced.contains(&i) { 0 } else { v })
            .collect();
        let ix = Idx::new(&out_dims)?.with_strides(0, &out_strides, layout.start_offset())?;
        let rdims: Vec<usize> = reduced.iter().map(|&d| dims[d]).collect();
        let rstrides: Vec<usize> = reduced.iter().map(|&d| st[d]).collect();
        let rd = if rdims.is_empty() {
            Idx::unit()
        } else {
            Idx::new(&rdims)?.with_strides(0, &rstrides, 0)?
        };
        let out = self.device.alloc(DType::F32, n_out)?;
        kernels::run_reduce_generic(
            &self.device,
            code,
            self.buf(),
            out.buf(),
            n_out,
            ix,
            rd,
            count.max(if rdims.is_empty() { 1 } else { 0 }),
        )?;
        Ok(Some(out))
    }

    fn index_select_native(
        &self,
        ids: &Self,
        l: &Layout,
        ids_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        let dims = l.dims();
        if dim >= dims.len() {
            return Err(Error::Msg("vulkan index_select: bad dim".into()));
        }
        let (src, src_off) = self.as_contiguous(l)?;
        let n_ids = ids_l.shape().elem_count();
        let left = dims[..dim].iter().product::<usize>();
        let dim_size = dims[dim];
        let right = dims[dim + 1..].iter().product::<usize>();
        let n = left * n_ids * right;
        let out = self.device.alloc(self.dtype, n)?;
        let elem8 = self.elem_size() == 8;
        let (ids_s, ids_off, i64_ids) = match ids.dtype {
            DType::I64 => {
                let (s, o) = ids.as_contiguous(ids_l)?;
                (s, o, true)
            }
            _ => {
                let (s, o) = ids.ids_u32(ids_l)?;
                (s, o, false)
            }
        };
        kernels::run_index_select(
            &self.device,
            elem8,
            i64_ids,
            src.buf(),
            ids_s.buf(),
            out.buf(),
            n,
            left,
            n_ids,
            right,
            dim_size,
            src_off,
            ids_off,
        )?;
        Ok(out)
    }

    fn gather_native(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        let dims = l.dims();
        if dim >= dims.len() || ids_l.dims().len() != dims.len() {
            return Err(Error::Msg("vulkan gather: bad dim or rank".into()));
        }
        let (ids_s, ids_off) = ids.ids_u32(ids_l)?;
        let ids_layout = Layout::new(ids_l.shape().clone(), ids_l.stride().to_vec(), ids_off);
        let n = ids_l.shape().elem_count();
        let mut src_strides = l.stride().to_vec();
        let dim_stride = src_strides[dim];
        src_strides[dim] = 0;
        let ix = Idx::new(ids_l.dims())?
            .with_layout(
                0,
                if ids_s.buffer == ids.buffer {
                    ids_l
                } else {
                    &ids_layout
                },
            )?
            .with_strides(1, &src_strides, l.start_offset())?;
        let out = self.device.alloc(self.dtype, n)?;
        kernels::run_gather(
            &self.device,
            self.buf(),
            ids_s.buf(),
            out.buf(),
            n,
            ix,
            dim_stride,
            dims[dim],
        )?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn scatter_native(
        &mut self,
        add: bool,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        let dims = l.dims();
        if dim >= dims.len() || ids_l.dims().len() != dims.len() || src_l.dims().len() != dims.len()
        {
            return Err(Error::Msg("vulkan scatter: bad dim or rank".into()));
        }
        let (ids_s, ids_off) = ids.ids_u32(ids_l)?;
        let ids_layout = Layout::new(ids_l.shape().clone(), ids_l.stride().to_vec(), ids_off);
        let ids_l = if ids_s.buffer == ids.buffer {
            ids_l
        } else {
            &ids_layout
        };
        // Enumerate the ids space with `dim` collapsed; the kernel walks
        // `dim` itself, in order.
        let mut cdims = ids_l.dims().to_vec();
        let n_j = cdims[dim];
        cdims[dim] = 1;
        let n = cdims.iter().product::<usize>();
        let ix = Idx::new(&cdims)?
            .with_layout(0, ids_l)?
            .with_layout(1, src_l)?
            .with_layout(2, l)?;
        kernels::run_scatter(
            &self.device,
            add,
            self.buf(),
            ids_s.buf(),
            src.buf(),
            n,
            ix,
            n_j,
            ids_l.stride()[dim],
            src_l.stride()[dim],
            l.stride()[dim],
            dims[dim],
        )
    }

    fn index_add_native(
        &self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        let dims = l.dims();
        if dim >= dims.len() {
            return Err(Error::Msg("vulkan index_add: bad dim".into()));
        }
        let dst = self.contiguous_copy(l)?;
        let (src_c, src_off) = src.as_contiguous(src_l)?;
        let (ids_s, ids_off) = ids.ids_u32(ids_l)?;
        let n_ids = ids_l.shape().elem_count();
        let left = dims[..dim].iter().product::<usize>();
        let right = dims[dim + 1..].iter().product::<usize>();
        kernels::run_index_add(
            &self.device,
            dst.buf(),
            ids_s.buf(),
            src_c.buf(),
            left,
            n_ids,
            right,
            dims[dim],
            src_off,
            ids_off,
        )?;
        Ok(dst)
    }

    fn matmul_native(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let (batch, m, n, k) = bmnk;
        let out = self.device.alloc(DType::F32, batch * m * n)?;
        let (lhs_buf, sa, _lhs_keep) = match mat_strides(lhs_l, batch) {
            Some(s) => (self.buf(), s, None),
            None => {
                let c = self.contiguous_copy(lhs_l)?;
                (
                    c.buf(),
                    MatStrides {
                        row: k,
                        col: 1,
                        offset: 0,
                        batch: m * k,
                    },
                    Some(c),
                )
            }
        };
        let (rhs_buf, sb, _rhs_keep) = match mat_strides(rhs_l, batch) {
            Some(s) => (rhs.buf(), s, None),
            None => {
                let c = rhs.contiguous_copy(rhs_l)?;
                (
                    c.buf(),
                    MatStrides {
                        row: n,
                        col: 1,
                        offset: 0,
                        batch: k * n,
                    },
                    Some(c),
                )
            }
        };
        kernels::run_matmul(
            &self.device,
            lhs_buf,
            rhs_buf,
            out.buf(),
            (batch, m, n, k),
            sa,
            sb,
        )?;
        Ok(out)
    }
}

impl BackendDevice for VulkanDevice {
    type Storage = VulkanStorage;

    fn new(gpu_id: usize) -> Result<Self> {
        Self::new(gpu_id)
    }

    fn set_seed(&self, _seed: u64) -> Result<()> {
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        Ok(0)
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Vulkan {
            gpu_id: self.gpu_id,
        }
    }

    fn same_device(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let numel = shape.elem_count();
        let storage = self.alloc(dtype, numel)?;
        // A fresh buffer: the host can clear it directly.
        unsafe { std::ptr::write_bytes(storage.mapped, 0, storage.capacity_bytes) };
        Ok(storage)
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        self.alloc(dtype, shape.elem_count())
    }

    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> Result<Self::Storage> {
        VulkanStorage::from_slice(s, self)
    }

    fn storage_from_cpu_storage(&self, cpu: &CpuStorage) -> Result<Self::Storage> {
        match cpu {
            CpuStorage::U8(v) => VulkanStorage::from_slice(v, self),
            CpuStorage::U32(v) => VulkanStorage::from_slice(v, self),
            CpuStorage::I16(v) => VulkanStorage::from_slice(v, self),
            CpuStorage::I32(v) => VulkanStorage::from_slice(v, self),
            CpuStorage::I64(v) => VulkanStorage::from_slice(v, self),
            CpuStorage::F32(v) => VulkanStorage::from_slice(v, self),
            CpuStorage::F64(v) => VulkanStorage::from_slice(v, self),
            CpuStorage::F16(v) => VulkanStorage::from_slice(v, self),
            CpuStorage::BF16(v) => VulkanStorage::from_slice(v, self),
            other => Err(Error::Msg(format!(
                "vulkan storage_from_cpu_storage: dtype {:?} not supported",
                other.dtype()
            ))),
        }
    }

    fn storage_from_cpu_storage_owned(&self, cpu: CpuStorage) -> Result<Self::Storage> {
        self.storage_from_cpu_storage(&cpu)
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, hi: f64) -> Result<Self::Storage> {
        let cpu = crate::cpu_backend::CpuDevice.rand_uniform(shape, dtype, lo, hi)?;
        self.storage_from_cpu_storage(&cpu)
    }

    fn rand_normal(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<Self::Storage> {
        let cpu = crate::cpu_backend::CpuDevice.rand_normal(shape, dtype, mean, std)?;
        self.storage_from_cpu_storage(&cpu)
    }

    fn synchronize(&self) -> Result<()> {
        self.flush()?;
        self.check_fault()
    }
}

// ─── Block-quantized weights on the device ───────────────────────────────────

/// Rows below which a quantized matmul dequantizes inside the GEMV kernel;
/// larger inputs (prefill) dequantize the weight to a scratch f32 buffer
/// once and run the tiled GEMM.
const QGEMV_MAX_ROWS: usize = 16;

/// A GGUF-block-quantized tensor held on the device in its on-disk format.
pub struct QVulkanStorage {
    inner: VulkanStorage,
    dtype: GgmlDType,
    elem_count: usize,
}

impl std::fmt::Debug for QVulkanStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "QVulkanStorage({:?}, {} elems)",
            self.dtype, self.elem_count
        )
    }
}

impl QVulkanStorage {
    fn bytes_for(dtype: GgmlDType, elem_count: usize) -> Result<usize> {
        let bs = dtype.block_size();
        if !elem_count.is_multiple_of(bs) {
            return Err(Error::Msg(format!(
                "vulkan: {elem_count} elements is not a whole number of {dtype:?} blocks"
            )));
        }
        Ok(elem_count / bs * dtype.type_size())
    }

    pub fn zeros(device: &VulkanDevice, elem_count: usize, dtype: GgmlDType) -> Result<Self> {
        let bytes = Self::bytes_for(dtype, elem_count)?;
        let inner = device.zeros_impl(&Shape::from(bytes), DType::U8)?;
        Ok(Self {
            inner,
            dtype,
            elem_count,
        })
    }

    /// Upload raw block bytes.
    pub fn from_bytes(
        device: &VulkanDevice,
        dtype: GgmlDType,
        elem_count: usize,
        data: &[u8],
    ) -> Result<Self> {
        let bytes = Self::bytes_for(dtype, elem_count)?;
        if data.len() < bytes {
            return Err(Error::Msg(format!(
                "vulkan: {} bytes given for a {dtype:?} tensor needing {bytes}",
                data.len()
            )));
        }
        let inner = device.alloc(DType::U8, bytes)?;
        unsafe { inner.set_bytes(&data[..bytes]) }?;
        Ok(Self {
            inner,
            dtype,
            elem_count,
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &VulkanDevice {
        &self.inner.device
    }

    pub fn elem_count(&self) -> usize {
        self.elem_count
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        Self::bytes_for(self.dtype, self.elem_count).unwrap_or(0)
    }

    /// Read the block bytes back to the host.
    pub fn data(&self) -> Result<Vec<u8>> {
        self.inner.get_bytes(self.storage_size_in_bytes())
    }

    /// Dequantize to an f32 storage on the device.
    pub fn dequantize(&self, elem_count: usize) -> Result<VulkanStorage> {
        kernels::note_dequantize(self.dtype, elem_count);
        let out = self.inner.device.alloc(DType::F32, elem_count)?;
        match self.dtype {
            GgmlDType::F32 => {
                let l = Layout::contiguous(elem_count);
                kernels::run_copy_strided(
                    &self.inner.device,
                    4,
                    self.inner.buf(),
                    out.buf(),
                    elem_count,
                    &l,
                    0,
                )?;
            }
            _ => kernels::run_dequant(
                &self.inner.device,
                self.dtype,
                self.inner.buf(),
                out.buf(),
                elem_count,
            )?,
        }
        Ok(out)
    }

    pub fn grouped_q8_eligible(&self, groups: usize, n: usize, k: usize) -> bool {
        self.dtype == GgmlDType::Q8_0 && kernels::grouped_q8_eligible(self.device(), groups, n, k)
    }

    pub fn grouped_q8_fwd(
        &self,
        shape: &Shape,
        storage: &VulkanStorage,
        layout: &Layout,
        groups: usize,
    ) -> Result<(VulkanStorage, Shape)> {
        let (n, k) = shape.dims2()?;
        let dev = self.device();
        let rows = match layout.dims() {
            [g, width] if (*g, *width) == (groups, k) => 1,
            [rows, g, width]
                if (*g, *width) == (groups, k)
                    && kernels::precise_prefill_optimized(dev, *rows) =>
            {
                *rows
            }
            _ => return Err(Error::Msg("vulkan grouped Q8: incompatible shape".into())),
        };
        if storage.dtype != DType::F32
            || !dev.same_device(&storage.device)
            || !self.grouped_q8_eligible(groups, n, k)
        {
            return Err(Error::Msg("vulkan grouped Q8: incompatible input".into()));
        }
        let Some((offset, _)) = layout.contiguous_offsets() else {
            return Err(Error::Msg("vulkan grouped Q8: noncontiguous input".into()));
        };
        let out = dev.alloc(DType::F32, rows * n)?;
        if rows == 1 {
            kernels::run_grouped_q8(
                dev,
                storage.buf(),
                self.inner.buf(),
                out.buf(),
                groups,
                n,
                k,
                offset,
            )?;
        } else {
            kernels::run_grouped_q8_prefill(
                dev,
                storage.buf(),
                self.inner.buf(),
                out.buf(),
                rows,
                groups,
                n,
                k,
                offset,
            )?;
        }
        Ok((out, Shape::from((rows, n))))
    }

    /// `x @ W^T` for `x` on the device (f32, contiguous) and this `[n, k]` weight.
    pub fn fwd(
        &self,
        self_shape: &Shape,
        storage: &VulkanStorage,
        layout: &Layout,
    ) -> Result<(VulkanStorage, Shape)> {
        if storage.dtype != DType::F32 {
            return Err(Error::Msg(format!(
                "vulkan qmatmul: input must be f32, got {:?}",
                storage.dtype
            )));
        }
        let Some((o1, _)) = layout.contiguous_offsets() else {
            return Err(Error::Msg(format!(
                "vulkan qmatmul: input tensor is not contiguous {layout:?}"
            )));
        };
        let (n, k) = self_shape.dims2()?;
        let src_shape = layout.shape();
        if src_shape.rank() < 2 {
            return Err(Error::Msg(format!(
                "vulkan qmatmul: input has only one dimension {layout:?}"
            )));
        }
        let mut dst_dims = src_shape.dims().to_vec();
        let last_k = dst_dims.pop().unwrap();
        if last_k != k {
            return Err(Error::Msg(format!(
                "vulkan qmatmul: input {layout:?} incompatible with {self_shape:?}"
            )));
        }
        dst_dims.push(n);
        let dst_shape = Shape::from(dst_dims);
        let m = src_shape.elem_count() / k;
        let dev = &self.inner.device;
        let out = dev.alloc(DType::F32, m * n)?;
        let w = self.inner.buf();
        let wide_dot = kernels::f64_dot_enabled_on(dev)
            && (matches!(self.dtype, GgmlDType::Q8_0 | GgmlDType::Q4K)
                || (self.dtype == GgmlDType::Q2K && n == 24 && k == 20480));
        if wide_dot {
            if m > QGEMV_MAX_ROWS {
                dev.exec().batch_limit = kernels::PREFILL_MAX_LAUNCHES;
            }
            kernels::run_qgemv_f64(dev, self.dtype, storage.buf(), w, out.buf(), m, n, k, o1, 1)?;
            return Ok((out, dst_shape));
        }
        match self.dtype {
            GgmlDType::F32 => {
                let sa = MatStrides {
                    row: k,
                    col: 1,
                    offset: o1,
                    batch: 0,
                };
                let sb = MatStrides {
                    row: 1,
                    col: k,
                    offset: 0,
                    batch: 0,
                };
                kernels::run_matmul_with_precision(
                    dev,
                    storage.buf(),
                    w,
                    out.buf(),
                    (1, m, n, k),
                    sa,
                    sb,
                    kernels::f64_dot_enabled_on(dev),
                )?;
            }
            GgmlDType::F16 | GgmlDType::BF16 if m <= QGEMV_MAX_ROWS => {
                kernels::run_hgemv(
                    dev,
                    self.dtype == GgmlDType::BF16,
                    storage.buf(),
                    w,
                    out.buf(),
                    m,
                    n,
                    k,
                    o1,
                )?;
            }
            _ if m <= QGEMV_MAX_ROWS => {
                kernels::run_qgemv(dev, self.dtype, storage.buf(), w, out.buf(), m, n, k, o1)?;
            }
            _ => {
                let bytes = n
                    .checked_mul(k)
                    .and_then(|v| v.checked_mul(4))
                    .ok_or_else(|| Error::Msg("vulkan qgemm scratch size overflow".into()))?
                    .max(4) as u64;
                let mut scratch = dev
                    .inner
                    .quant_gemm_scratch
                    .lock()
                    .unwrap_or_else(|p| p.into_inner());
                // Bound queued activations as well as expanded weights. Keep
                // submission in the recorder, after complete dispatch recording.
                dev.exec().batch_limit = kernels::PREFILL_MAX_LAUNCHES;
                if scratch.as_ref().is_none_or(|s| s.capacity < bytes) {
                    let (buffer, alloc, _) = dev.alloc_raw(bytes)?;
                    if let Some(old) = scratch.replace(QuantGemmScratch {
                        buffer,
                        alloc,
                        capacity: bytes,
                    }) {
                        // The previous capacity may still be read by queued work.
                        dev.release(old.buffer, old.alloc);
                    }
                }
                let s = scratch.as_ref().unwrap();
                let wf = Buf {
                    buffer: s.buffer,
                    bytes: s.capacity,
                };
                // The existing compute->compute barrier orders the previous
                // GEMM's reads before this overwrite, then makes the new
                // dequantized weights visible to GEMM. The recorder still governs submission; no additional copy.
                kernels::run_dequant(dev, self.dtype, w, wf, n * k)?;
                let sa = MatStrides {
                    row: k,
                    col: 1,
                    offset: o1,
                    batch: 0,
                };
                let sb = MatStrides {
                    row: 1,
                    col: k,
                    offset: 0,
                    batch: 0,
                };
                kernels::run_matmul_with_precision(
                    dev,
                    storage.buf(),
                    wf,
                    out.buf(),
                    (1, m, n, k),
                    sa,
                    sb,
                    wide_dot,
                )?;
            }
        }
        Ok((out, dst_shape))
    }

    /// Gather rows `ids` of this `[rows, hidden]` table as f32 `[n_ids, hidden]`.
    pub fn embedding(
        &self,
        rows: usize,
        hidden: usize,
        ids: &VulkanStorage,
        ids_l: &Layout,
    ) -> Result<VulkanStorage> {
        let (ids_s, ids_off) = ids.ids_u32(ids_l)?;
        let n_ids = ids_l.shape().elem_count();
        let dev = &self.inner.device;
        let out = dev.alloc(DType::F32, n_ids * hidden)?;
        match self.dtype {
            GgmlDType::F32 => {
                kernels::run_index_select(
                    dev,
                    false,
                    false,
                    self.inner.buf(),
                    ids_s.buf(),
                    out.buf(),
                    n_ids * hidden,
                    1,
                    n_ids,
                    hidden,
                    rows,
                    0,
                    ids_off,
                )?;
            }
            GgmlDType::F16 | GgmlDType::BF16 => {
                kernels::run_hembed(
                    dev,
                    self.dtype == GgmlDType::BF16,
                    self.inner.buf(),
                    ids_s.buf(),
                    out.buf(),
                    n_ids,
                    hidden,
                    rows,
                    ids_off,
                )?;
            }
            _ => kernels::run_qembed(
                dev,
                self.dtype,
                self.inner.buf(),
                ids_s.buf(),
                out.buf(),
                n_ids,
                hidden,
                rows,
                ids_off,
            )?,
        }
        Ok(out)
    }
}

#[cfg(all(test, feature = "vulkan"))]
mod tests {
    use super::*;
    use crate::{Device, Tensor};

    /// One device shared by every test in the module (they run on parallel
    /// threads; a context per test is not a real configuration).
    fn device() -> Option<Device> {
        static DEVICE: std::sync::OnceLock<Option<Device>> = std::sync::OnceLock::new();
        DEVICE
            .get_or_init(|| match Device::new_vulkan(0) {
                Ok(d) => Some(d),
                Err(e) => {
                    eprintln!("skipping vulkan test: {e}");
                    None
                }
            })
            .clone()
    }

    fn close(a: &Tensor, b: &Tensor, tol: f32, what: &str) {
        let a: Vec<f32> = a
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let b: Vec<f32> = b.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(a.len(), b.len(), "{what}: length");
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            let d = (x - y).abs();
            let scale = x.abs().max(y.abs()).max(1.0);
            assert!(
                d <= tol * scale,
                "{what}: element {i} differs: device {x} vs cpu {y}"
            );
        }
    }

    #[test]
    fn vulkan_model_scoped_precision_policy() -> crate::Result<()> {
        let Some(Device::Vulkan(fast)) = device() else {
            return Ok(());
        };
        if !fast.inner.shader_float64 {
            return Ok(());
        }
        let accurate = fast.with_precise_math()?;
        assert!(!fast.precise_math());
        assert!(accurate.precise_math());
        assert!(Arc::ptr_eq(&fast.inner, &accurate.inner));
        if std::env::var_os("CANDLE_VULKAN_PREFILL_OPT").is_none() && kernels::native_enabled() {
            // An automatic compact-prefill choice belongs to this model's
            // handle; sharing its queue/allocator must not enable callers.
            assert!(!kernels::precise_prefill_optimized(&fast, 32));
            for rows in [2, 25, 32] {
                assert!(kernels::precise_prefill_optimized(&accurate, rows));
            }
            for rows in [0, 1, 33] {
                assert!(!kernels::precise_prefill_optimized(&accurate, rows));
            }
        }
        let fast_device = Device::Vulkan(fast.clone());
        let accurate_device = Device::Vulkan(accurate);
        assert!(fast_device.same_device(&accurate_device));
        let values: Vec<f32> = (0..64).map(|i| i as f32 / 31.).collect();
        let table = Tensor::from_vec(values, (2, 32), &accurate_device)?;
        let table = crate::quantized::QTensor::quantize(&table, GgmlDType::Q8_0)?;
        // Caller IDs may use the original device handle. The gathered
        // activation must inherit the owning model's policy from its storage.
        let ids = Tensor::new(&[1u32], &fast_device)?;
        let rows = table.embedding(&ids)?;
        let Device::Vulkan(rows_device) = rows.device() else {
            panic!("device transfer")
        };
        assert!(rows_device.precise_math());
        assert!(!fast.precise_math());
        Ok(())
    }

    /// Write f32 to a device buffer and read it back.
    #[test]
    fn f32_roundtrip_via_tensor() -> crate::Result<()> {
        let Some(dev) = device() else { return Ok(()) };
        let v: Vec<f32> = (0..1000).map(|i| i as f32 * 0.25 - 3.0).collect();
        let t = Tensor::from_vec(v.clone(), 1000, &dev)?;
        let back: Vec<f32> = t.to_device(&Device::Cpu)?.to_vec1()?;
        assert_eq!(v, back);
        Ok(())
    }

    /// Every native kernel agrees with the CPU backend on contiguous,
    /// broadcast, transposed and narrowed views.
    #[test]
    fn vulkan_parity_native() -> crate::Result<()> {
        let Some(dev) = device() else { return Ok(()) };
        let cpu = Device::Cpu;
        let before = kernels::native_exec_count();
        let x = Tensor::arange(0f32, 96f32, &cpu)?
            .reshape((4, 24))?
            .affine(0.05, -1.7)?;
        let y = Tensor::arange(0f32, 96f32, &cpu)?
            .reshape((4, 24))?
            .affine(-0.03, 0.9)?;
        let xd = x.to_device(&dev)?;
        let yd = y.to_device(&dev)?;

        close(
            &xd.affine(1.5, -0.25)?,
            &x.affine(1.5, -0.25)?,
            1e-6,
            "affine",
        );
        close(&xd.exp()?, &x.exp()?, 1e-5, "exp");
        close(
            &xd.t()?.contiguous()?,
            &x.t()?.contiguous()?,
            0.0,
            "transpose copy",
        );
        close(&xd.silu()?, &x.silu()?, 1e-5, "silu");
        close(&xd.gelu()?, &x.gelu()?, 1e-5, "gelu");
        close(&xd.gelu_erf()?, &x.gelu_erf()?, 1e-5, "gelu_erf");
        close(&(&xd + &yd)?, &(&x + &y)?, 1e-6, "add");
        close(
            &xd.broadcast_mul(&yd.narrow(0, 0, 1)?)?,
            &x.broadcast_mul(&y.narrow(0, 0, 1)?)?,
            1e-6,
            "broadcast mul",
        );
        close(
            &xd.t()?.broadcast_add(&yd.narrow(1, 3, 1)?.t()?)?,
            &x.t()?.broadcast_add(&y.narrow(1, 3, 1)?.t()?)?,
            1e-6,
            "strided broadcast add",
        );
        close(
            &xd.narrow(1, 5, 7)?.sqr()?,
            &x.narrow(1, 5, 7)?.sqr()?,
            1e-6,
            "narrow sqr",
        );
        close(&xd.abs()?.powf(2.5)?, &x.abs()?.powf(2.5)?, 1e-5, "powf");
        close(
            &xd.clamp(-0.5f32, 0.5f32)?,
            &x.clamp(-0.5f32, 0.5f32)?,
            0.0,
            "clamp",
        );
        close(&xd.round()?, &x.round()?, 0.0, "round");

        close(&xd.sum_keepdim(1)?, &x.sum_keepdim(1)?, 1e-5, "sum last");
        close(&xd.max_keepdim(1)?, &x.max_keepdim(1)?, 0.0, "max last");
        close(&xd.min_keepdim(1)?, &x.min_keepdim(1)?, 0.0, "min last");
        close(
            &xd.sum_keepdim(0)?,
            &x.sum_keepdim(0)?,
            1e-5,
            "sum dim0 (generic)",
        );
        close(&xd.sum_all()?, &x.sum_all()?, 1e-4, "sum all");
        close(
            &xd.t()?.sum_keepdim(1)?,
            &x.t()?.sum_keepdim(1)?,
            1e-5,
            "sum over strided",
        );
        let am: Vec<u32> = xd.argmax(1)?.to_device(&cpu)?.to_vec1()?;
        assert_eq!(am, x.argmax(1)?.to_vec1::<u32>()?, "argmax");

        let mask = xd.gt(&yd)?;
        assert_eq!(
            mask.to_device(&cpu)?.to_vec2::<u8>()?,
            x.gt(&y)?.to_vec2::<u8>()?,
            "gt"
        );
        close(
            &mask.where_cond(&xd, &yd)?,
            &x.gt(&y)?.where_cond(&x, &y)?,
            0.0,
            "where",
        );
        let u: Vec<u32> = xd
            .abs()?
            .to_dtype(DType::U32)?
            .to_device(&cpu)?
            .flatten_all()?
            .to_vec1()?;
        assert_eq!(
            u,
            x.abs()?
                .to_dtype(DType::U32)?
                .flatten_all()?
                .to_vec1::<u32>()?,
            "cast u32"
        );
        close(
            &xd.to_dtype(DType::F16)?.to_dtype(DType::F32)?,
            &x.to_dtype(DType::F16)?.to_dtype(DType::F32)?,
            0.0,
            "f16 round trip",
        );
        close(
            &xd.to_dtype(DType::BF16)?.to_dtype(DType::F32)?,
            &x.to_dtype(DType::BF16)?.to_dtype(DType::F32)?,
            0.0,
            "bf16 round trip",
        );
        close(
            &mask.to_dtype(DType::F32)?,
            &x.gt(&y)?.to_dtype(DType::F32)?,
            0.0,
            "u8 -> f32",
        );
        let mi: Vec<i64> = xd
            .to_dtype(DType::I64)?
            .to_device(&cpu)?
            .flatten_all()?
            .to_vec1()?;
        assert_eq!(
            mi,
            x.to_dtype(DType::I64)?.flatten_all()?.to_vec1::<i64>()?,
            "cast i64"
        );

        close(
            &Tensor::cat(&[&xd, &yd], 1)?,
            &Tensor::cat(&[&x, &y], 1)?,
            0.0,
            "cat dim1",
        );
        close(
            &Tensor::cat(&[&xd.narrow(0, 1, 2)?, &yd.narrow(0, 0, 1)?], 0)?,
            &Tensor::cat(&[&x.narrow(0, 1, 2)?, &y.narrow(0, 0, 1)?], 0)?,
            0.0,
            "cat dim0",
        );
        let m1 = x.gt(&y)?;
        let m2 = x.lt(&y)?;
        let mcat = Tensor::cat(&[&mask.narrow(1, 1, 5)?, &xd.lt(&yd)?.narrow(1, 0, 3)?], 1)?;
        assert_eq!(
            mcat.to_device(&cpu)?.to_vec2::<u8>()?,
            Tensor::cat(&[&m1.narrow(1, 1, 5)?, &m2.narrow(1, 0, 3)?], 1)?.to_vec2::<u8>()?,
            "u8 cat (packed copy)"
        );

        let ids = Tensor::new(&[3u32, 0, 2, 3], &cpu)?;
        close(
            &xd.index_select(&ids.to_device(&dev)?, 0)?,
            &x.index_select(&ids, 0)?,
            0.0,
            "index_select dim0",
        );
        close(
            &xd.index_select(&ids.to_device(&dev)?, 1)?,
            &x.index_select(&ids, 1)?,
            0.0,
            "index_select dim1",
        );
        let ids64 = Tensor::new(&[1i64, 1, 0], &cpu)?;
        close(
            &xd.index_select(&ids64.to_device(&dev)?, 0)?,
            &x.index_select(&ids64, 0)?,
            0.0,
            "index_select i64",
        );
        let gids = Tensor::new(&[[0u32, 5, 23], [1, 1, 2], [22, 0, 7], [3, 4, 5]], &cpu)?;
        close(
            &xd.gather(&gids.to_device(&dev)?, 1)?,
            &x.gather(&gids, 1)?,
            0.0,
            "gather",
        );
        let add_ids = Tensor::new(&[1u32, 3, 1], &cpu)?;
        let src = Tensor::arange(0f32, 72f32, &cpu)?.reshape((3, 24))?;
        close(
            &xd.index_add(&add_ids.to_device(&dev)?, &src.to_device(&dev)?, 0)?,
            &x.index_add(&add_ids, &src, 0)?,
            1e-6,
            "index_add",
        );
        let sc_ids = Tensor::new(&[[0u32, 1, 2], [3, 2, 1]], &cpu)?;
        let sc_src = Tensor::new(&[[10f32, 11., 12.], [13., 14., 15.]], &cpu)?;
        let base = Tensor::zeros((4, 3), DType::F32, &cpu)?;
        close(
            &base.to_device(&dev)?.scatter_add(
                &sc_ids.to_device(&dev)?,
                &sc_src.to_device(&dev)?,
                0,
            )?,
            &base.scatter_add(&sc_ids, &sc_src, 0)?,
            0.0,
            "scatter_add",
        );
        close(
            &base.to_device(&dev)?.scatter(
                &sc_ids.to_device(&dev)?,
                &sc_src.to_device(&dev)?,
                0,
            )?,
            &base.scatter(&sc_ids, &sc_src, 0)?,
            0.0,
            "scatter",
        );
        // Duplicate ids resolve like the sequential CPU loop (last write wins).
        let dup_ids = Tensor::new(&[[0u32, 1, 1], [0, 1, 1]], &cpu)?;
        close(
            &base.to_device(&dev)?.scatter(
                &dup_ids.to_device(&dev)?,
                &sc_src.to_device(&dev)?,
                0,
            )?,
            &base.scatter(&dup_ids, &sc_src, 0)?,
            0.0,
            "scatter duplicate ids",
        );
        close(
            &base.to_device(&dev)?.scatter_add(
                &dup_ids.to_device(&dev)?,
                &sc_src.to_device(&dev)?,
                0,
            )?,
            &base.scatter_add(&dup_ids, &sc_src, 0)?,
            0.0,
            "scatter_add duplicate ids",
        );
        // An empty contraction is a zero matrix, not whatever the fresh
        // output buffer held.
        let e0 = Tensor::zeros((3, 0), DType::F32, &dev)?;
        let e1 = Tensor::zeros((0, 5), DType::F32, &dev)?;
        close(
            &e0.matmul(&e1)?,
            &Tensor::zeros((3, 5), DType::F32, &cpu)?,
            0.0,
            "matmul with K == 0",
        );
        // i64 → f32 keeps the high word.
        let big = Tensor::new(&[4_294_967_296i64, -4_294_967_297, 5, -1], &cpu)?;
        close(
            &big.to_device(&dev)?.to_dtype(DType::F32)?,
            &big.to_dtype(DType::F32)?,
            0.0,
            "cast large i64 to f32",
        );
        // An out-of-range id is reported at the next host read-back; the
        // device stays usable afterwards.
        let bad = Tensor::new(&[1u32, 9], &dev)?;
        assert!(
            xd.index_select(&bad, 0)?.to_device(&cpu).is_err(),
            "out-of-range index_select must fail"
        );
        // An i64 id beyond u32 (or negative) is out of range, not truncated.
        let big_id = Tensor::new(&[4_294_967_297i64, 1], &dev)?;
        assert!(
            xd.gather(
                &big_id
                    .reshape((2, 1))?
                    .broadcast_as((2, 24))?
                    .contiguous()?,
                0
            )
            .and_then(|t| t.to_device(&cpu))
            .is_err(),
            "i64 id beyond u32 must fail"
        );
        let neg_id = Tensor::new(&[-1i64, 1], &dev)?;
        assert!(
            xd.index_select(&neg_id, 0)?.to_device(&cpu).is_err(),
            "negative i64 id must fail"
        );
        close(
            &xd.index_select(&ids.to_device(&dev)?, 0)?,
            &x.index_select(&ids, 0)?,
            0.0,
            "index_select after a fault",
        );

        let a = Tensor::arange(0f32, 24f32 * 40f32, &cpu)?
            .reshape((24, 40))?
            .affine(1e-3, -0.4)?;
        let w = Tensor::arange(0f32, 40f32 * 17f32, &cpu)?
            .reshape((17, 40))?
            .affine(-2e-3, 0.3)?;
        let (ad, wd) = (a.to_device(&dev)?, w.to_device(&dev)?);
        close(&ad.matmul(&wd.t()?)?, &a.matmul(&w.t()?)?, 1e-4, "gemm NT");
        close(
            &ad.t()?.contiguous()?.t()?.matmul(&wd.t()?)?,
            &a.matmul(&w.t()?)?,
            1e-4,
            "gemm NT strided lhs",
        );
        close(
            &wd.matmul(&ad.t()?)?,
            &w.matmul(&a.t()?)?,
            1e-4,
            "gemm NT 2",
        );
        let b = Tensor::arange(0f32, 40f32 * 9f32, &cpu)?
            .reshape((40, 9))?
            .affine(3e-3, -0.2)?;
        close(
            &ad.matmul(&b.to_device(&dev)?)?,
            &a.matmul(&b)?,
            1e-4,
            "gemm NN",
        );
        let q = Tensor::arange(0f32, 2f32 * 3f32 * 40f32, &cpu)?
            .reshape((2, 3, 1, 40))?
            .affine(1e-2, -0.5)?;
        let kk = Tensor::arange(0f32, 2f32 * 3f32 * 5f32 * 40f32, &cpu)?
            .reshape((2, 3, 5, 40))?
            .affine(-1e-2, 0.2)?;
        close(
            &q.to_device(&dev)?
                .matmul(&kk.to_device(&dev)?.transpose(2, 3)?)?,
            &q.matmul(&kk.transpose(2, 3)?)?,
            1e-4,
            "batched decode attention QK^T",
        );
        let scores = q.matmul(&kk.transpose(2, 3)?)?;
        close(
            &scores.to_device(&dev)?.matmul(&kk.to_device(&dev)?)?,
            &scores.matmul(&kk)?,
            1e-4,
            "batched decode attention scores V",
        );
        let row = a.narrow(0, 2, 1)?;
        close(
            &row.to_device(&dev)?.matmul(&wd.t()?)?,
            &row.matmul(&w.t()?)?,
            1e-4,
            "gemv NT",
        );
        close(
            &row.to_device(&dev)?.matmul(&b.to_device(&dev)?)?,
            &row.matmul(&b)?,
            1e-4,
            "gemv NN",
        );
        let big = Tensor::arange(0f32, 130f32 * 70f32, &cpu)?
            .reshape((130, 70))?
            .affine(1e-3, -0.05)?;
        let bw = Tensor::arange(0f32, 70f32 * 131f32, &cpu)?
            .reshape((131, 70))?
            .affine(-1e-3, 0.04)?;
        close(
            &big.to_device(&dev)?.matmul(&bw.to_device(&dev)?.t()?)?,
            &big.matmul(&bw.t()?)?,
            1e-4,
            "gemm edges",
        );

        let z = Tensor::zeros((3, 5), DType::F32, &dev)?;
        close(&z, &Tensor::zeros((3, 5), DType::F32, &cpu)?, 0.0, "zeros");
        close(&xd.copy()?, &x, 0.0, "clone");
        let o = Tensor::ones((2, 7), DType::F32, &dev)?;
        close(
            &o,
            &Tensor::ones((2, 7), DType::F32, &cpu)?,
            0.0,
            "ones (fill)",
        );

        let n = kernels::native_exec_count() - before;
        eprintln!(
            "vulkan parity: {n} native kernel launches, {} fallbacks",
            kernels::fallback_count()
        );
        assert!(n >= 40, "native kernels must actually run (got {n})");
        Ok(())
    }

    /// Block-quantized weights: dequantize, GEMV, GEMM and embedding gather
    /// on the device match candle's CPU reference for every block format.
    #[test]
    fn vulkan_quantized_parity() -> crate::Result<()> {
        use crate::quantized::{GgmlDType, QMatMul, QTensor};
        let Some(dev) = device() else { return Ok(()) };
        let cpu = Device::Cpu;
        let (n, k) = (12usize, 512usize);
        let w = Tensor::arange(0f32, (n * k) as f32, &cpu)?
            .reshape((n, k))?
            .affine(7e-4, -2.1)?
            .sin()?;
        let x1 = Tensor::arange(0f32, k as f32, &cpu)?
            .reshape((1, k))?
            .affine(3e-3, -0.7)?
            .cos()?;
        let xm = Tensor::arange(0f32, (40 * k) as f32, &cpu)?
            .reshape((40, k))?
            .affine(1e-3, -0.9)?
            .sin()?;
        for dtype in [
            GgmlDType::F32,
            GgmlDType::F16,
            GgmlDType::BF16,
            GgmlDType::Q8_0,
            GgmlDType::Q8_1,
            GgmlDType::Q4_0,
            GgmlDType::Q4_1,
            GgmlDType::Q5_0,
            GgmlDType::Q5_1,
            GgmlDType::Q2K,
            GgmlDType::Q3K,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
            GgmlDType::Q8K,
        ] {
            let q_cpu = QTensor::quantize(&w, dtype)?;
            let bytes = q_cpu.data()?.into_owned();
            let q_dev = QTensor::new(
                crate::quantized::QStorage::from_data(
                    std::borrow::Cow::Borrowed(&bytes),
                    &dev,
                    dtype,
                )?,
                (n, k),
            )?;
            let what = format!("{dtype:?}");
            close(
                &q_dev.dequantize(&dev)?,
                &q_cpu.dequantize(&cpu)?,
                1e-6,
                &format!("{what} dequantize"),
            );
            assert_eq!(q_dev.data()?.into_owned(), bytes, "{what} data round trip");
            let w_ref = q_cpu.dequantize(&cpu)?;
            let mm_cpu = QMatMul::from_qtensor(q_cpu)?;
            let mm_dev = QMatMul::from_qtensor(q_dev)?;
            use crate::Module;
            let ref1 = x1.matmul(&w_ref.t()?)?;
            let refm = xm.matmul(&w_ref.t()?)?;
            let dev1 = mm_dev.forward(&x1.to_device(&dev)?)?;
            let devm = mm_dev.forward(&xm.to_device(&dev)?)?;
            close(&dev1, &ref1, 1e-4, &format!("{what} gemv vs f32"));
            close(&devm, &refm, 1e-4, &format!("{what} gemm vs f32"));
            close(
                &dev1,
                &mm_cpu.forward(&x1)?,
                2e-2,
                &format!("{what} gemv vs cpu"),
            );
            close(
                &devm,
                &mm_cpu.forward(&xm)?,
                2e-2,
                &format!("{what} gemm vs cpu"),
            );
            let ids = Tensor::new(&[3u32, 0, 11, 3], &cpu)?;
            close(
                &mm_dev.embedding(&ids.to_device(&dev)?)?,
                &mm_cpu.embedding(&ids)?,
                1e-6,
                &format!("{what} embedding"),
            );
        }
        Ok(())
    }

    /// Many small allocations and frees exercise the sub-allocator and the
    /// deferred release of buffers referenced by an open batch.
    #[test]
    fn vulkan_allocator_churn() -> crate::Result<()> {
        let Some(dev) = device() else { return Ok(()) };
        let cpu = Device::Cpu;
        let mut acc = Tensor::zeros((64, 64), DType::F32, &dev)?;
        let mut acc_cpu = Tensor::zeros((64, 64), DType::F32, &cpu)?;
        for i in 0..300 {
            let t = Tensor::full(i as f32 * 0.01, (64, 64), &dev)?;
            let tc = Tensor::full(i as f32 * 0.01, (64, 64), &cpu)?;
            acc = (acc + t.exp()?)?;
            acc_cpu = (acc_cpu + tc.exp()?)?;
            if i % 97 == 0 {
                // A large tensor forces a dedicated block.
                let big = Tensor::ones((1024, 1024), DType::F32, &dev)?;
                acc = acc.broadcast_add(&big.sum_all()?.affine(1e-6, 0.0)?)?;
                acc_cpu = acc_cpu.broadcast_add(
                    &Tensor::ones((1024, 1024), DType::F32, &cpu)?
                        .sum_all()?
                        .affine(1e-6, 0.0)?,
                )?;
            }
        }
        close(&acc, &acc_cpu, 1e-4, "allocator churn");
        Ok(())
    }

    /// Repeated prefill GEMMs must not retain one f32 weight matrix per
    /// dispatch. Alternate weights and grow scratch while old work is queued.
    #[test]
    fn vulkan_quantized_gemm_reuses_bounded_scratch() -> crate::Result<()> {
        use crate::quantized::{QMatMul, QStorage, QTensor};
        use crate::Module;
        let vk = VulkanDevice::new_isolated(0)?;
        let weak = Arc::downgrade(&vk.inner);
        {
            let dev = Device::Vulkan(vk.clone());
            let x = Tensor::arange(0f32, (192 * 512) as f32, &Device::Cpu)?
                .affine(0.003, -1.0)?
                .sin()?
                .reshape((192, 512))?
                .to_device(&dev)?;
            let mut matrices = Vec::new();
            let mut expected = Vec::new();
            for (n, scale) in [(513, 0.002), (1024, 0.005)] {
                let w = Tensor::arange(0f32, (n * 512) as f32, &Device::Cpu)?
                    .affine(scale, -0.5)?
                    .cos()?
                    .reshape((n, 512))?;
                let q = QTensor::quantize(&w, GgmlDType::Q8_0)?;
                let q = QTensor::new(
                    QStorage::from_data(q.data()?, &dev, GgmlDType::Q8_0)?,
                    (n, 512),
                )?;
                expected.push(
                    x.matmul(&q.dequantize(&dev)?.t()?)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                );
                matrices.push(QMatMul::from_qtensor(q)?);
            }
            vk.flush()?;
            let before = vk.memory_stats();
            let mut retained = Vec::new();
            // 512 dispatches: exercise the bounded prefill submission limit.
            // The old path retains 384 MiB of expanded matrices here.
            for i in 0..256 {
                let out = matrices[i % 2].forward(&x)?;
                if i < 2 || i >= 254 {
                    retained.push((i % 2, out));
                }
            }
            let queued = vk.memory_stats();
            assert_eq!(vk.exec().batch_limit, kernels::PREFILL_MAX_LAUNCHES);
            assert!(
                queued.0 <= before.0 + CHUNK_BYTES,
                "unbounded reservations: {before:?} -> {queued:?}"
            );
            assert!(
                queued.1 <= before.1 + (192 << 20),
                "expanded matrices retained: {before:?} -> {queued:?}"
            );
            for (which, out) in retained {
                let actual = out.flatten_all()?.to_vec1::<f32>()?;
                assert_eq!(
                    actual, expected[which],
                    "scratch overwrite changed queued output"
                );
            }
            vk.flush()?;
            assert_eq!(
                before,
                vk.memory_stats(),
                "scratch must retire at host readback"
            );
            assert_eq!(
                vk.exec().batch_limit,
                kernels::MAX_LAUNCHES,
                "readback must restore decode batching"
            );
            // Leave scratch queued again: dropping the device must not leak a
            // self-owning VulkanStorage through the cache.
            drop(matrices[0].forward(&x)?);
        }
        drop(vk);
        assert!(
            weak.upgrade().is_none(),
            "scratch owns its device in an Arc cycle"
        );
        Ok(())
    }

    #[test]
    #[ignore = "asserts a device's private recording state, which the per-thread fault drain \
                flushes whenever another test thread exits; run with --ignored --test-threads=1"]
    fn vulkan_allocation_keeps_pending_arguments_alive() -> crate::Result<()> {
        let vk = VulkanDevice::new_isolated(0)?;
        let temporary = vk.alloc(DType::F32, 1)?;
        kernels::run_fill(&vk, 4, temporary.buf(), 1, &Layout::contiguous(1), 0)?;
        drop(temporary);
        // Allocation can happen after raw arguments have been prepared. It
        // must not submit and destroy their deferred owners at chunk pressure.
        for _ in 0..33 {
            drop(vk.alloc(DType::U8, 4 << 20)?);
        }
        assert!(
            vk.exec().recording,
            "allocation submitted pending arguments"
        );
        assert_eq!(
            vk.exec().garbage.len(),
            34,
            "allocation retired prepared buffers"
        );
        vk.flush()?;
        Ok(())
    }

    #[test]
    fn vulkan_devices_are_shared_per_ordinal() -> crate::Result<()> {
        let a = VulkanDevice::new(0)?;
        let b = VulkanDevice::new(0)?;
        assert!(
            a.same_device(&b),
            "new(0) must return the same logical device"
        );
        let c = VulkanDevice::new_isolated(0)?;
        assert!(!a.same_device(&c), "new_isolated must not share");
        let x = Tensor::new(&[1f32, 2., 3.], &Device::Vulkan(a))?;
        let y = Tensor::new(&[4f32, 5., 6.], &Device::Vulkan(b))?;
        assert_eq!((x + y)?.to_vec1::<f32>()?, vec![5., 7., 9.]);
        Ok(())
    }

    #[test]
    fn vulkan_quantized_gemm_scratch_is_shared_safely_between_threads() -> crate::Result<()> {
        use crate::quantized::{QMatMul, QStorage, QTensor};
        use crate::Module;
        let dev = Device::Vulkan(VulkanDevice::new(0)?);
        let x = Tensor::arange(0f32, (32 * 512) as f32, &Device::Cpu)?
            .affine(0.001, -0.5)?
            .sin()?
            .reshape((32, 512))?
            .to_device(&dev)?;
        let mut jobs = Vec::new();
        let barrier = Arc::new(std::sync::Barrier::new(2));
        for scale in [0.002, 0.005] {
            let w = Tensor::arange(0f32, (512 * 512) as f32, &Device::Cpu)?
                .affine(scale, -1.0)?
                .cos()?
                .reshape((512, 512))?;
            let q = QTensor::quantize(&w, GgmlDType::Q8_0)?;
            let q = QTensor::new(
                QStorage::from_data(q.data()?, &dev, GgmlDType::Q8_0)?,
                (512, 512),
            )?;
            let expected = x
                .matmul(&q.dequantize(&dev)?.t()?)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            jobs.push((QMatMul::from_qtensor(q)?, expected));
        }
        let threads: Vec<_> = jobs
            .into_iter()
            .map(|(mm, expected)| {
                let x = x.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || -> crate::Result<()> {
                    barrier.wait();
                    for _ in 0..32 {
                        let out = mm.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;
                        assert_eq!(out, expected, "concurrent projection overwrote scratch");
                    }
                    Ok(())
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap()?;
        }
        Ok(())
    }

    /// Importance-weighted quantization on the device yields the CPU's blocks.
    #[test]
    fn vulkan_imatrix_quantize() -> crate::Result<()> {
        let Some(dev) = device() else { return Ok(()) };
        let cpu = Device::Cpu;
        let x = Tensor::arange(0f32, 512f32, &cpu)?
            .affine(0.01, -2.0)?
            .reshape((2, 256))?;
        let w: Vec<f32> = (0..256).map(|i| 1.0 + (i % 7) as f32).collect();
        for dtype in [GgmlDType::Q4K, GgmlDType::Q6K] {
            let q_cpu = crate::quantized::QTensor::quantize_imatrix(&x, &w, dtype)?;
            let q_dev =
                crate::quantized::QTensor::quantize_imatrix(&x.to_device(&dev)?, &w, dtype)?;
            assert_eq!(
                q_cpu.data()?.as_ref(),
                q_dev.data()?.as_ref(),
                "{dtype:?}: imatrix blocks"
            );
            let q_onto = crate::quantized::QTensor::quantize_imatrix_onto(&x, &w, dtype, &dev)?;
            assert_eq!(
                q_cpu.data()?.as_ref(),
                q_onto.data()?.as_ref(),
                "{dtype:?}: imatrix blocks (onto)"
            );
        }
        Ok(())
    }

    /// A fault raised by one thread's launch is reported to that thread only:
    /// a thread doing valid work on the same device never sees it.
    #[test]
    fn vulkan_fault_is_per_thread() -> crate::Result<()> {
        let Some(dev) = device() else { return Ok(()) };
        let cpu = Device::Cpu;
        let x = Tensor::arange(0f32, 32f32, &cpu)?
            .reshape((4, 8))?
            .to_device(&dev)?;
        let bad = Tensor::new(&[1u32, 9], &dev)?;
        let good = Tensor::new(&[1u32, 3], &dev)?;
        let faulty = std::thread::spawn({
            let (x, bad, cpu) = (x.clone(), bad.clone(), cpu.clone());
            move || {
                (0..20).all(|_| {
                    x.index_select(&bad, 0)
                        .and_then(|t| t.to_device(&cpu))
                        .is_err()
                })
            }
        });
        let clean = std::thread::spawn({
            let (x, good, cpu) = (x.clone(), good.clone(), cpu.clone());
            move || {
                (0..200).all(|_| {
                    x.index_select(&good, 0)
                        .and_then(|t| t.to_device(&cpu))
                        .is_ok()
                })
            }
        });
        assert!(
            faulty.join().unwrap(),
            "the faulting thread must see its error every time"
        );
        assert!(
            clean.join().unwrap(),
            "the clean thread must never see another thread's fault"
        );
        Ok(())
    }

    /// A thread that exits with a faulting launch still in flight does not
    /// hand its fault to the next owner of its slot.
    #[test]
    fn vulkan_fault_slot_is_drained_on_thread_exit() -> crate::Result<()> {
        let Some(dev) = device() else { return Ok(()) };
        let cpu = Device::Cpu;
        let x = Tensor::arange(0f32, 32f32, &cpu)?
            .reshape((4, 8))?
            .to_device(&dev)?;
        let bad = Tensor::new(&[1u32, 9], &dev)?;
        let good = Tensor::new(&[1u32, 3], &dev)?;
        for _ in 0..4 {
            std::thread::spawn({
                let (x, bad) = (x.clone(), bad.clone());
                // Launch and drop without any read-back, then exit.
                move || drop(x.index_select(&bad, 0))
            })
            .join()
            .unwrap();
            let clean = std::thread::spawn({
                let (x, good, cpu) = (x.clone(), good.clone(), cpu.clone());
                move || {
                    (0..20).all(|_| {
                        x.index_select(&good, 0)
                            .and_then(|t| t.to_device(&cpu))
                            .is_ok()
                    })
                }
            });
            assert!(
                clean.join().unwrap(),
                "a recycled slot must come back clean"
            );
        }
        Ok(())
    }

    /// A quantized matmul with a small, block-aligned inner dimension runs
    /// `xs` quantized per 32 values as `k_quant_x8` does (llama.cpp's
    /// `quantize_row_q8_0`) and dequantized again: `d · round(x / d)`.
    fn q8_round_trip(xs: &[f32]) -> Vec<f32> {
        xs.chunks(32)
            .flat_map(|b| {
                let amax = b.iter().fold(0f32, |a, v| a.max(v.abs()));
                let d = amax / 127.0;
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                b.iter()
                    .map(move |v| d * (v * id).round())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Every quantized-GEMV kernel — the generic one, the Mali one and its
    /// int8 form (with the hardware dot product when the device has it, and
    /// the polyfill), forced here whatever the device — matches a CPU reference for every
    /// block format, with `N` off the Mali kernel's 16-column tiles and
    /// every `M` up to the GEMV limit (partial 4-row slices included).
    #[test]
    fn vulkan_qgemv_kernels_match_cpu() -> crate::Result<()> {
        use crate::backend::{BackendDevice, BackendStorage};
        use crate::quantized::{GgmlDType, QTensor};
        let Some(dev) = device() else { return Ok(()) };
        let Device::Vulkan(vd) = &dev else {
            unreachable!()
        };
        let cpu = Device::Cpu;
        let (n, k) = (37usize, 512usize);
        let w = Tensor::arange(0f32, (n * k) as f32, &cpu)?
            .reshape((n, k))?
            .affine(7e-4, -2.1)?
            .sin()?;
        let x = Tensor::arange(0f32, ((QGEMV_MAX_ROWS + 1) * k) as f32, &cpu)?
            .affine(1.3e-3, -0.9)?
            .cos()?;
        let xv: Vec<f32> = x.to_vec1()?;
        for dtype in [
            GgmlDType::Q8_0,
            GgmlDType::Q8_1,
            GgmlDType::Q4_0,
            GgmlDType::Q4_1,
            GgmlDType::Q5_0,
            GgmlDType::Q5_1,
            GgmlDType::Q2K,
            GgmlDType::Q3K,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
            GgmlDType::Q8K,
        ] {
            let q = QTensor::quantize(&w, dtype)?;
            let wref = q.dequantize(&cpu)?;
            let bytes = q.data()?.into_owned();
            let wbuf = vd.storage_from_slice(&bytes)?;
            let xbuf = vd.storage_from_slice(&xv)?;
            let mut variants = vec![
                kernels::Qgemv::Generic,
                kernels::Qgemv::CubeCl,
                kernels::Qgemv::CubeClExpanded,
                kernels::Qgemv::Mali,
                kernels::Qgemv::MaliInt8 { hw_dot: false },
            ];
            if vd.limits().int_dot {
                variants.push(kernels::Qgemv::MaliInt8 { hw_dot: true });
            }
            let wv: Vec<f32> = wref.flatten_all()?.to_vec1()?;
            for kernel in variants {
                for rows in [1usize, 2, 3, 4, 5, 7, 8, QGEMV_MAX_ROWS] {
                    // Offset the activations by one row to exercise `xoff`.
                    let out = vd.alloc(DType::F32, rows * n)?;
                    kernels::run_qgemv_with(
                        vd,
                        kernel,
                        dtype,
                        xbuf.buf(),
                        wbuf.buf(),
                        out.buf(),
                        rows,
                        n,
                        k,
                        k,
                    )?;
                    let got = match out.to_cpu_storage()? {
                        crate::CpuStorage::F32(v) => Tensor::from_vec(v, (rows, n), &cpu)?,
                        _ => unreachable!(),
                    };
                    let xr = x.narrow(0, k, rows * k)?;
                    let want = match kernel {
                        // The int8 kernel sees the activations as llama.cpp's
                        // Q8_0 rounding leaves them: compare with exactly that.
                        kernels::Qgemv::MaliInt8 { .. } => {
                            let xq = q8_round_trip(&xr.to_vec1()?);
                            let mut c = vec![0f32; rows * n];
                            for r in 0..rows {
                                for j in 0..n {
                                    c[r * n + j] = (0..k)
                                        .map(|i| xq[r * k + i] as f64 * wv[j * k + i] as f64)
                                        .sum::<f64>()
                                        as f32;
                                }
                            }
                            Tensor::from_vec(c, (rows, n), &cpu)?
                        }
                        _ => xr.reshape((rows, k))?.matmul(&wref.t()?)?,
                    };
                    // The int8 path is checked against its own exact
                    // reference, so it is held tighter (it agrees within 2e-6).
                    let tol = if matches!(kernel, kernels::Qgemv::MaliInt8 { .. }) {
                        1e-5
                    } else {
                        1e-4
                    };
                    close(&got, &want, tol, &format!("{kernel:?} {dtype:?} m={rows}"));
                }
            }
        }
        Ok(())
    }

    /// the fused GEMV (m <= 16) and the dequantize + GEMM branch (m > 16)
    /// and matches the CPU on both.
    #[test]
    fn vulkan_quantized_matmul_small_k() -> crate::Result<()> {
        use crate::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
        use crate::Module;
        let Some(dev) = device() else { return Ok(()) };
        let cpu = Device::Cpu;
        let (n, k) = (5usize, 96usize);
        let w = Tensor::arange(0f32, (n * k) as f32, &cpu)?
            .affine(0.01, -0.3)?
            .reshape((n, k))?;
        let q_cpu = QTensor::quantize(&w, GgmlDType::Q8_0)?;
        let q_dev = QTensor::new(
            QStorage::from_data(q_cpu.data()?, &dev, GgmlDType::Q8_0)?,
            (n, k),
        )?;
        let (mm_cpu, mm_dev) = (QMatMul::from_qtensor(q_cpu)?, QMatMul::from_qtensor(q_dev)?);
        for m in [1usize, 20] {
            let x = Tensor::arange(0f32, (m * k) as f32, &cpu)?
                .affine(0.002, -0.1)?
                .reshape((m, k))?;
            close(
                &mm_dev.forward(&x.to_device(&dev)?)?,
                &mm_cpu.forward(&x)?,
                2e-3,
                "quantized matmul k=96",
            );
        }
        Ok(())
    }
}
