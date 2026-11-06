//! Native Vulkan kernels: pipeline cache, command batching, argument
//! marshalling and one launch helper per kernel in [`super::glsl`].
//!
//! Every kernel is compiled once per device (GLSL → SPIR-V through `naga`,
//! then a compute pipeline) and cached by name.  Launches are *recorded*
//! into one open command buffer per device and submitted lazily: the only
//! synchronisation points are host read-backs (`to_cpu_storage`), an
//! explicit `synchronize`, and the batch filling up.  Each dispatch is
//! followed by a compute→compute memory barrier, so consecutive kernels see
//! each other's writes in order; device buffers dropped while a batch is
//! open are released only after that batch has completed.
//!
//! Native execution is on by default; `CANDLE_VULKAN_NATIVE=0` forces every
//! op through the CPU round-trip (a correctness reference), and
//! `CANDLE_VULKAN_TRACE=1` logs each op that still falls back.
// Kernel launchers mirror their kernel's argument list one-to-one.
#![allow(clippy::too_many_arguments)]

use super::glsl;
use super::{Alloc, VulkanDevice};
use crate::{Error, Layout, Result};
use ash::vk;
use std::collections::HashMap;

// ─── Gating / diagnostics ────────────────────────────────────────────────────

static NATIVE_EXEC: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static FALLBACKS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Whether native kernels run (default) or every op takes the CPU
/// round-trip (`CANDLE_VULKAN_NATIVE=0`).
pub fn native_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("CANDLE_VULKAN_NATIVE") {
        Ok(s) => !(s == "0" || s.eq_ignore_ascii_case("false") || s.eq_ignore_ascii_case("off")),
        Err(_) => true,
    })
}

fn trace_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("CANDLE_VULKAN_TRACE"), Ok(s) if s == "1"))
}

/// Number of native kernel dispatches so far.
pub fn native_exec_count() -> usize {
    NATIVE_EXEC.load(std::sync::atomic::Ordering::Relaxed)
}

pub(crate) fn note_native_exec() {
    NATIVE_EXEC.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Count of ops that went through the CPU round-trip.
pub fn fallback_count() -> usize {
    FALLBACKS.load(std::sync::atomic::Ordering::Relaxed)
}

static H2D_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static D2H_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SUBMITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static WAIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DEQUANTIZE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FALLBACK_OPS: std::sync::Mutex<Option<HashMap<String, usize>>> = std::sync::Mutex::new(None);

pub(crate) fn note_h2d(bytes: usize) {
    H2D_BYTES.fetch_add(bytes as u64, std::sync::atomic::Ordering::Relaxed);
}
pub(crate) fn note_d2h(bytes: usize) {
    D2H_BYTES.fetch_add(bytes as u64, std::sync::atomic::Ordering::Relaxed);
}
pub(crate) fn note_submit(wait: std::time::Duration) {
    SUBMITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    WAIT_NS.fetch_add(wait.as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
}
pub(crate) fn note_dequantize(dtype: crate::quantized::GgmlDType, elem_count: usize) {
    DEQUANTIZE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if trace_enabled() {
        eprintln!("[vulkan] full dequantization of a {dtype:?} tensor ({elem_count} elements)");
    }
}

/// A snapshot of the process-wide Vulkan execution counters (all devices).
///
/// These make CPU fallback and host/device traffic measurable: a model that
/// "runs on Vulkan" but shows many fallbacks or large transfers per token is
/// mostly running on the CPU.
#[derive(Clone, Debug, Default)]
pub struct VulkanStats {
    /// Native compute dispatches recorded.
    pub native_dispatches: usize,
    /// Ops that went through the CPU round-trip (download, CPU kernel, upload).
    pub cpu_fallbacks: usize,
    /// CPU fallbacks by op name, most frequent first.
    pub cpu_fallbacks_by_op: Vec<(String, usize)>,
    /// Bytes written from the host into device buffers (uploads).
    pub host_to_device_bytes: u64,
    /// Bytes read back from device buffers to the host (downloads).
    pub device_to_host_bytes: u64,
    /// Command-buffer submissions that the host waited on (host/device syncs).
    pub syncs: u64,
    /// Total host time spent waiting for submitted GPU work, in nanoseconds.
    pub gpu_wait_ns: u64,
    /// Full-tensor dequantizations of quantized weights on the device.
    pub dequantizations: u64,
}

impl std::fmt::Display for VulkanStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "native dispatches {}, CPU fallbacks {}, H2D {:.1} MiB, D2H {:.1} MiB, syncs {}, GPU wait {:.1} ms, dequantizations {}",
            self.native_dispatches,
            self.cpu_fallbacks,
            self.host_to_device_bytes as f64 / (1u64 << 20) as f64,
            self.device_to_host_bytes as f64 / (1u64 << 20) as f64,
            self.syncs,
            self.gpu_wait_ns as f64 / 1e6,
            self.dequantizations,
        )?;
        if !self.cpu_fallbacks_by_op.is_empty() {
            let ops: Vec<String> = self
                .cpu_fallbacks_by_op
                .iter()
                .map(|(op, n)| format!("{op}={n}"))
                .collect();
            write!(f, " [{}]", ops.join(", "))?;
        }
        Ok(())
    }
}

/// The current process-wide Vulkan counters.
pub fn stats() -> VulkanStats {
    use std::sync::atomic::Ordering::Relaxed;
    let mut by_op: Vec<(String, usize)> = FALLBACK_OPS
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .as_ref()
        .map(|m| m.iter().map(|(k, v)| (k.clone(), *v)).collect())
        .unwrap_or_default();
    by_op.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    VulkanStats {
        native_dispatches: NATIVE_EXEC.load(Relaxed),
        cpu_fallbacks: FALLBACKS.load(Relaxed),
        cpu_fallbacks_by_op: by_op,
        host_to_device_bytes: H2D_BYTES.load(Relaxed),
        device_to_host_bytes: D2H_BYTES.load(Relaxed),
        syncs: SUBMITS.load(Relaxed),
        gpu_wait_ns: WAIT_NS.load(Relaxed),
        dequantizations: DEQUANTIZE.load(Relaxed),
    }
}

/// Reset every process-wide Vulkan counter to zero (e.g. after model load,
/// to measure steady-state inference only).
pub fn reset_stats() {
    use std::sync::atomic::Ordering::Relaxed;
    NATIVE_EXEC.store(0, Relaxed);
    FALLBACKS.store(0, Relaxed);
    H2D_BYTES.store(0, Relaxed);
    D2H_BYTES.store(0, Relaxed);
    SUBMITS.store(0, Relaxed);
    WAIT_NS.store(0, Relaxed);
    DEQUANTIZE.store(0, Relaxed);
    *FALLBACK_OPS.lock().unwrap_or_else(|p| p.into_inner()) = None;
}

/// Record (and, under `CANDLE_VULKAN_TRACE`, log) an op that fell back to
/// the CPU path.  `why` is `None` when no native kernel exists for the case
/// and `Some(err)` when the kernel failed.
pub fn note_fallback(op: &str, why: Option<&Error>) {
    FALLBACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    *FALLBACK_OPS
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .get_or_insert_with(HashMap::new)
        .entry(op.to_string())
        .or_insert(0) += 1;
    if trace_enabled() {
        match why {
            Some(e) => eprintln!("[vulkan] {op}: native kernel failed, using CPU round-trip: {e}"),
            None => {
                eprintln!("[vulkan] {op}: no native kernel for this case, using CPU round-trip")
            }
        }
    }
}

// ─── Index descriptor ────────────────────────────────────────────────────────

pub const MAXD: usize = glsl::MAXD;

/// Output shape plus up to three input stride sets, written to the
/// parameter buffer of a launch.  Layout must match the `Idx` struct in the
/// GLSL prelude (std430: tightly packed ints).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Idx {
    pub nd: i32,
    pub dims: [i32; MAXD],
    pub s0: [i32; MAXD],
    pub s1: [i32; MAXD],
    pub s2: [i32; MAXD],
    pub o0: i32,
    pub o1: i32,
    pub o2: i32,
}

impl Idx {
    pub fn unit() -> Self {
        Idx {
            nd: 1,
            dims: [1; MAXD],
            s0: [0; MAXD],
            s1: [0; MAXD],
            s2: [0; MAXD],
            o0: 0,
            o1: 0,
            o2: 0,
        }
    }

    pub fn new(dims: &[usize]) -> Result<Self> {
        if dims.len() > MAXD {
            return Err(Error::Msg(format!(
                "vulkan: rank {} exceeds the kernel limit of {MAXD}",
                dims.len()
            )));
        }
        let mut ix = Idx {
            nd: dims.len().max(1) as i32,
            ..Idx::unit()
        };
        for (i, &d) in dims.iter().enumerate() {
            ix.dims[i] = to_i32(d)?;
        }
        Ok(ix)
    }

    /// Fill stride set `which` (0..3) from a layout whose shape is this
    /// descriptor's output shape.
    pub fn with_layout(self, which: usize, l: &Layout) -> Result<Self> {
        if l.dims().len() > MAXD {
            return Err(Error::Msg(
                "vulkan: layout rank exceeds the kernel limit".into(),
            ));
        }
        self.with_strides(which, l.stride(), l.start_offset())
    }

    /// Stride set `which` from explicit strides and offset.
    pub fn with_strides(mut self, which: usize, strides: &[usize], offset: usize) -> Result<Self> {
        let (s, o) = match which {
            0 => (&mut self.s0, &mut self.o0),
            1 => (&mut self.s1, &mut self.o1),
            _ => (&mut self.s2, &mut self.o2),
        };
        for (i, &st) in strides.iter().enumerate() {
            s[i] = to_i32(st)?;
        }
        *o = to_i32(offset)?;
        Ok(self)
    }

    fn bytes(&self) -> &[u8] {
        // # Safety: repr(C) plain-old-data.
        unsafe {
            std::slice::from_raw_parts(self as *const Idx as *const u8, std::mem::size_of::<Idx>())
        }
    }
}

pub fn to_i32(v: usize) -> Result<i32> {
    i32::try_from(v).map_err(|_| {
        Error::Msg(format!(
            "vulkan: dimension {v} exceeds the 32-bit kernel index range"
        ))
    })
}

// ─── Pipelines ───────────────────────────────────────────────────────────────

/// Push-constant range every pipeline layout declares (the guaranteed
/// minimum `maxPushConstantsSize`).
const PUSH_BYTES: u32 = 128;

/// A compiled compute pipeline.
pub struct Pipe {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub dsl: vk::DescriptorSetLayout,
    pub nbuf: u32,
}

/// Naga 27 lowers GLSL `barrier()` to `Barrier::all()`, which includes
/// SUB_GROUP. Its SPIR-V writer interprets that flag as *subgroup execution
/// scope*, even when WORK_GROUP is also set. That leaves different subgroups
/// racing on shared reduction/tile memory (#81). GLSL barrier() in a compute
/// shader must rendezvous the whole workgroup. Correct the execution scope
/// before validation/emission, retaining the other memory-ordering flags.
fn fix_glsl_barrier_scope(block: &mut naga::Block) {
    use naga::Statement;
    for statement in block.iter_mut() {
        match statement {
            Statement::ControlBarrier(flags) if flags.contains(naga::Barrier::WORK_GROUP) => {
                flags.remove(naga::Barrier::SUB_GROUP);
            }
            Statement::Block(body) => fix_glsl_barrier_scope(body),
            Statement::If { accept, reject, .. } => {
                fix_glsl_barrier_scope(accept);
                fix_glsl_barrier_scope(reject);
            }
            Statement::Switch { cases, .. } => {
                for case in cases {
                    fix_glsl_barrier_scope(&mut case.body);
                }
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                fix_glsl_barrier_scope(body);
                fix_glsl_barrier_scope(continuing);
            }
            _ => {}
        }
    }
}

fn glsl_to_spirv(source: &str) -> Result<Vec<u32>> {
    use naga::back::spv;
    use naga::front::glsl::{Frontend, Options};
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    use naga::ShaderStage;

    let options = Options::from(ShaderStage::Compute);
    let mut frontend = Frontend::default();
    let mut module = frontend.parse(&options, source).map_err(|e| {
        Error::Msg(format!(
            "vulkan shader parse failed: {}\n--- source ---\n{source}",
            e.emit_to_string(source)
        ))
    })?;
    for (_, function) in module.functions.iter_mut() {
        fix_glsl_barrier_scope(&mut function.body);
    }
    for entry in &mut module.entry_points {
        fix_glsl_barrier_scope(&mut entry.function.body);
    }
    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    let info = validator.validate(&module).map_err(|e| {
        Error::Msg(format!(
            "vulkan shader validation failed: {e:?}\n--- source ---\n{source}"
        ))
    })?;
    let mut writer = spv::Writer::new(&spv::Options::default())
        .map_err(|e| Error::Msg(format!("vulkan spv writer init failed: {e:?}")))?;
    let mut words: Vec<u32> = Vec::new();
    let pipeline_options = spv::PipelineOptions {
        shader_stage: ShaderStage::Compute,
        entry_point: "main".to_string(),
    };
    writer
        .write(&module, &info, Some(&pipeline_options), &None, &mut words)
        .map_err(|e| Error::Msg(format!("vulkan spv write failed: {e:?}")))?;
    if source.contains("// JOSHUA_NO_CONTRACTION") {
        words = no_contraction(words);
    }
    Ok(words)
}

// Naga's GLSL front end currently drops `precise`. Error-free transforms
// require the individual rounded adds/multiplies to survive driver fusion.
// SPIR-V annotations may refer forward to result IDs; no IDs are allocated.
fn no_contraction(words: Vec<u32>) -> Vec<u32> {
    let mut decorations = Vec::new();
    let mut types = words.len();
    let mut pos = 5;
    while pos < words.len() {
        let count = (words[pos] >> 16) as usize;
        let op = words[pos] & 0xffff;
        if (19..=39).contains(&op) {
            types = types.min(pos);
        }
        // OpFAdd, OpFSub, OpFMul, OpFDiv. The result ID is operand two.
        if matches!(op, 129 | 131 | 133 | 136) {
            decorations.extend_from_slice(&[(3 << 16) | 71, words[pos + 2], 42]);
        }
        pos += count;
    }
    let mut out = Vec::with_capacity(words.len() + decorations.len());
    out.extend_from_slice(&words[..types]);
    out.extend(decorations);
    out.extend_from_slice(&words[types..]);
    out
}

/// The descriptor-set + pipeline layouts for `nbuf` tensor bindings plus
/// the parameter buffer, shared by every pipeline with that binding count.
pub struct Layouts {
    pub dsl: vk::DescriptorSetLayout,
    pub layout: vk::PipelineLayout,
}

pub(super) fn create_layouts(device: &ash::Device, nbuf: u32) -> Result<Layouts> {
    let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..=nbuf)
        .map(|b| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    let dsl = unsafe {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )
    }
    .map_err(|e| Error::Msg(format!("vulkan create_descriptor_set_layout failed: {e:?}")))?;
    let push = [vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        offset: 0,
        size: PUSH_BYTES,
    }];
    let set_layouts = [dsl];
    let layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts)
                .push_constant_ranges(&push),
            None,
        )
    }
    .map_err(|e| {
        unsafe { device.destroy_descriptor_set_layout(dsl, None) };
        Error::Msg(format!("vulkan create_pipeline_layout failed: {e:?}"))
    })?;
    Ok(Layouts { dsl, layout })
}

/// Compile a WGSL compute shader (entry point `main`) to SPIR-V — for the
/// kernels that need what naga's GLSL front end lacks, such as the packed
/// int8 dot product (`dot4I8Packed`, emitted as `OpSDot`).
pub(super) fn wgsl_to_spirv(source: &str) -> Result<Vec<u32>> {
    use naga::back::spv;
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    let module = naga::front::wgsl::parse_str(source).map_err(|e| {
        Error::Msg(format!(
            "vulkan shader parse failed: {}\n--- source ---\n{source}",
            e.emit_to_string(source)
        ))
    })?;
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .map_err(|e| {
            Error::Msg(format!(
                "vulkan shader validation failed: {e:?}\n--- source ---\n{source}"
            ))
        })?;
    let mut words = Vec::new();
    let pipeline_options = spv::PipelineOptions {
        shader_stage: naga::ShaderStage::Compute,
        entry_point: "main".to_string(),
    };
    spv::Writer::new(&spv::Options::default())
        .map_err(|e| Error::Msg(format!("vulkan spv writer init failed: {e:?}")))?
        .write(&module, &info, Some(&pipeline_options), &None, &mut words)
        .map_err(|e| Error::Msg(format!("vulkan spv write failed: {e:?}")))?;
    Ok(words)
}

pub(super) fn create_pipe(
    device: &ash::Device,
    layouts: &Layouts,
    nbuf: u32,
    spirv: &[u32],
) -> Result<Pipe> {
    let module = unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spirv), None)
    }
    .map_err(|e| Error::Msg(format!("vulkan create_shader_module failed: {e:?}")))?;
    let entry = std::ffi::CString::new("main").unwrap();
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(&entry);
    let info = [vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(layouts.layout)];
    let pipelines =
        unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &info, None) };
    unsafe { device.destroy_shader_module(module, None) };
    let pipeline = match pipelines {
        Ok(p) => p[0],
        Err((_, e)) => {
            return Err(Error::Msg(format!(
                "vulkan create_compute_pipelines failed: {e:?}"
            )))
        }
    };
    Ok(Pipe {
        pipeline,
        layout: layouts.layout,
        dsl: layouts.dsl,
        nbuf,
    })
}

// ─── Command batching ────────────────────────────────────────────────────────

/// Bytes of parameter-buffer ring per batch (each launch takes one
/// [`PARAM_SLOT`]-byte slot).
const PARAMS_BYTES: usize = 2 << 20;
/// Slot size: two `Idx` descriptors, rounded to the largest
/// `minStorageBufferOffsetAlignment` (256).
const PARAM_SLOT: usize = 512;
/// Launches recorded before a batch is submitted on its own.
pub(super) const MAX_LAUNCHES: usize = 1024;
// Expanded-weight scratch bounds weight memory; this bounds queued prefill
// activations. Submission stays at the existing post-dispatch boundary.
pub(super) const PREFILL_MAX_LAUNCHES: usize = 64;

/// A device buffer that must outlive the batch that references it.
pub struct Garbage {
    pub buffer: vk::Buffer,
    pub alloc: Alloc,
}

/// The recording state of one device: a command buffer, its descriptor
/// pool, the parameter ring and the buffers awaiting release.
pub struct Exec {
    device: ash::Device,
    queue: vk::Queue,
    cmd_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    desc_pool: vk::DescriptorPool,
    params_buf: vk::Buffer,
    params_mem: vk::DeviceMemory,
    params_map: *mut u8,
    params_off: usize,
    /// The device's fault buffer (see `glsl.rs`, one word per thread slot
    /// of `crate::fault_slot`): indexing kernels set the calling thread's
    /// slot on an out-of-range id; [`Exec::take_fault`] reads and clears it.
    fault_buf: vk::Buffer,
    fault_mem: vk::DeviceMemory,
    fault_map: *mut u8,
    pub(super) recording: bool,
    launches: usize,
    pub(super) batch_limit: usize,
    pub(super) garbage: Vec<Garbage>,
}

unsafe impl Send for Exec {}

impl Exec {
    pub(super) fn new(
        device: &ash::Device,
        queue: vk::Queue,
        queue_family: u32,
        host_mem_type: u32,
    ) -> Result<Self> {
        let cmd_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .map_err(|e| Error::Msg(format!("vulkan create_command_pool failed: {e:?}")))?;
        let cmd = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|e| Error::Msg(format!("vulkan allocate_command_buffers failed: {e:?}")))?[0];
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(|e| Error::Msg(format!("vulkan create_fence failed: {e:?}")))?;
        let sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: (MAX_LAUNCHES * 8) as u32,
        }];
        let desc_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(MAX_LAUNCHES as u32)
                    .pool_sizes(&sizes),
                None,
            )
        }
        .map_err(|e| Error::Msg(format!("vulkan create_descriptor_pool failed: {e:?}")))?;
        // The parameter ring: one host-visible buffer, bump-allocated.
        let params_buf = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(PARAMS_BYTES as u64)
                    .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }
        .map_err(|e| Error::Msg(format!("vulkan create_buffer(params) failed: {e:?}")))?;
        let req = unsafe { device.get_buffer_memory_requirements(params_buf) };
        let params_mem = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(host_mem_type),
                None,
            )
        }
        .map_err(|e| Error::Msg(format!("vulkan allocate_memory(params) failed: {e:?}")))?;
        unsafe { device.bind_buffer_memory(params_buf, params_mem, 0) }
            .map_err(|e| Error::Msg(format!("vulkan bind_buffer_memory(params) failed: {e:?}")))?;
        let params_map = unsafe {
            device.map_memory(params_mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        }
        .map_err(|e| Error::Msg(format!("vulkan map_memory(params) failed: {e:?}")))?
            as *mut u8;
        let (fault_buf, fault_mem, fault_map) =
            host_buffer(device, FAULT_BYTES as u64, host_mem_type, "fault")?;
        unsafe { std::ptr::write_bytes(fault_map, 0, FAULT_BYTES) };
        Ok(Exec {
            device: device.clone(),
            queue,
            cmd_pool,
            cmd,
            fence,
            desc_pool,
            params_buf,
            params_mem,
            params_map,
            params_off: 0,
            fault_buf,
            fault_mem,
            fault_map,
            recording: false,
            launches: 0,
            batch_limit: MAX_LAUNCHES,
            garbage: Vec::new(),
        })
    }

    fn begin(&mut self) -> Result<()> {
        if self.recording {
            return Ok(());
        }
        unsafe {
            self.device
                .reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())
                .map_err(|e| Error::Msg(format!("vulkan reset_command_buffer failed: {e:?}")))?;
            self.device
                .begin_command_buffer(
                    self.cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|e| Error::Msg(format!("vulkan begin_command_buffer failed: {e:?}")))?;
        }
        self.recording = true;
        self.launches = 0;
        self.params_off = 0;
        Ok(())
    }

    /// Submit the open batch (if any), wait for it, and release everything
    /// it referenced.  Returns the buffers to free to the caller (which
    /// holds the allocator).
    pub(super) fn flush(&mut self) -> Result<Vec<Garbage>> {
        if self.recording {
            self.recording = false;
            unsafe {
                // Make the batch's writes visible to the host.
                let barrier = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)];
                self.device.cmd_pipeline_barrier(
                    self.cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &barrier,
                    &[],
                    &[],
                );
                self.device
                    .end_command_buffer(self.cmd)
                    .map_err(|e| Error::Msg(format!("vulkan end_command_buffer failed: {e:?}")))?;
                let cmds = [self.cmd];
                let submit = [vk::SubmitInfo::default().command_buffers(&cmds)];
                let t0 = std::time::Instant::now();
                self.device
                    .queue_submit(self.queue, &submit, self.fence)
                    .map_err(|e| Error::Msg(format!("vulkan queue_submit failed: {e:?}")))?;
                self.device
                    .wait_for_fences(&[self.fence], true, u64::MAX)
                    .map_err(|e| Error::Msg(format!("vulkan wait_for_fences failed: {e:?}")))?;
                note_submit(t0.elapsed());
                self.device
                    .reset_fences(&[self.fence])
                    .map_err(|e| Error::Msg(format!("vulkan reset_fences failed: {e:?}")))?;
                self.device
                    .reset_descriptor_pool(self.desc_pool, vk::DescriptorPoolResetFlags::empty())
                    .map_err(|e| {
                        Error::Msg(format!("vulkan reset_descriptor_pool failed: {e:?}"))
                    })?;
            }
        }
        Ok(std::mem::take(&mut self.garbage))
    }

    /// Record one dispatch.  `params` is written to the ring and bound last.
    fn record(
        &mut self,
        pipe: &Pipe,
        bufs: &[Buf],
        push: &[u8],
        params: &[u8],
        groups: [u32; 3],
    ) -> Result<bool> {
        self.begin()?;
        let dev = &self.device;
        // Parameter slot.
        if self.params_off + PARAM_SLOT > PARAMS_BYTES || params.len() > PARAM_SLOT {
            return Err(Error::Msg("vulkan: parameter ring exhausted".into()));
        }
        let slot = self.params_off;
        unsafe {
            std::ptr::copy_nonoverlapping(params.as_ptr(), self.params_map.add(slot), params.len())
        };
        self.params_off += PARAM_SLOT;
        // Descriptor set.
        let set_layouts = [pipe.dsl];
        let set = unsafe {
            dev.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.desc_pool)
                    .set_layouts(&set_layouts),
            )
        }
        .map_err(|e| Error::Msg(format!("vulkan allocate_descriptor_sets failed: {e:?}")))?[0];
        let mut infos: Vec<vk::DescriptorBufferInfo> = bufs
            .iter()
            .map(|b| vk::DescriptorBufferInfo {
                buffer: b.buffer,
                offset: 0,
                range: b.bytes.max(4),
            })
            .collect();
        infos.push(vk::DescriptorBufferInfo {
            buffer: self.params_buf,
            offset: slot as u64,
            range: PARAM_SLOT as u64,
        });
        let writes: Vec<vk::WriteDescriptorSet> = infos
            .iter()
            .enumerate()
            .map(|(i, bi)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(bi))
            })
            .collect();
        unsafe {
            dev.update_descriptor_sets(&writes, &[]);
            dev.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, pipe.pipeline);
            dev.cmd_bind_descriptor_sets(
                self.cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipe.layout,
                0,
                &[set],
                &[],
            );
            let mut pc = [0u8; PUSH_BYTES as usize];
            pc[..push.len()].copy_from_slice(push);
            dev.cmd_push_constants(self.cmd, pipe.layout, vk::ShaderStageFlags::COMPUTE, 0, &pc);
            dev.cmd_dispatch(self.cmd, groups[0], groups[1], groups[2]);
            let barrier = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
            dev.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &barrier,
                &[],
                &[],
            );
        }
        self.launches += 1;
        note_native_exec();
        Ok(self.launches >= self.batch_limit)
    }
}

impl Exec {
    /// The fault word as a kernel argument.
    pub(super) fn fault(&self) -> Buf {
        Buf {
            buffer: self.fault_buf,
            bytes: FAULT_BYTES as u64,
        }
    }

    /// Clear fault `slot` (after a flush).
    pub(super) fn clear_fault(&mut self, slot: usize) {
        unsafe { std::ptr::write_volatile((self.fault_map as *mut u32).add(slot), 0) };
    }

    /// Whether an indexing kernel launched by this thread flagged an
    /// out-of-range id since the last call (the flag is cleared).
    /// Meaningful after a flush.
    pub(super) fn take_fault(&mut self) -> bool {
        let p = unsafe { (self.fault_map as *mut u32).add(crate::fault_slot::current()) };
        let v = unsafe { std::ptr::read_volatile(p) };
        if v != 0 {
            unsafe { std::ptr::write_volatile(p, 0) };
        }
        v != 0
    }
}

/// Size of the fault buffer (one word per thread slot).
const FAULT_BYTES: usize = crate::fault_slot::BYTES;

/// A small host-visible buffer bound to its own allocation and mapped.
fn host_buffer(
    device: &ash::Device,
    size: u64,
    mem_type: u32,
    what: &str,
) -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8)> {
    let buf = unsafe {
        device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )
    }
    .map_err(|e| Error::Msg(format!("vulkan create_buffer({what}) failed: {e:?}")))?;
    let req = unsafe { device.get_buffer_memory_requirements(buf) };
    if req.memory_type_bits & (1 << mem_type) == 0 {
        unsafe { device.destroy_buffer(buf, None) };
        return Err(Error::Msg(format!(
            "vulkan: the {what} buffer cannot use memory type {mem_type} (memoryTypeBits {:#x})",
            req.memory_type_bits
        )));
    }
    let mem = unsafe {
        device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mem_type),
            None,
        )
    }
    .map_err(|e| Error::Msg(format!("vulkan allocate_memory({what}) failed: {e:?}")))?;
    unsafe { device.bind_buffer_memory(buf, mem, 0) }
        .map_err(|e| Error::Msg(format!("vulkan bind_buffer_memory({what}) failed: {e:?}")))?;
    let map = unsafe { device.map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }
        .map_err(|e| Error::Msg(format!("vulkan map_memory({what}) failed: {e:?}")))?
        as *mut u8;
    Ok((buf, mem, map))
}

impl Drop for Exec {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.unmap_memory(self.fault_mem);
            self.device.destroy_buffer(self.fault_buf, None);
            self.device.free_memory(self.fault_mem, None);
            self.device.unmap_memory(self.params_mem);
            self.device.destroy_buffer(self.params_buf, None);
            self.device.free_memory(self.params_mem, None);
            self.device.destroy_descriptor_pool(self.desc_pool, None);
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.cmd_pool, None);
        }
    }
}

/// A buffer argument: the handle and the bytes the kernel may address.
#[derive(Clone, Copy)]
pub struct Buf {
    pub buffer: vk::Buffer,
    pub bytes: u64,
}

/// Push-constant builder.
#[derive(Default)]
pub struct Push(Vec<u8>);

impl Push {
    pub fn i(mut self, v: i32) -> Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn u(mut self, v: u32) -> Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn f(mut self, v: f32) -> Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn us(self, v: usize) -> Result<Self> {
        Ok(self.i(to_i32(v)?))
    }
}

/// Group counts for `n` invocations of 1-D groups of `wg`, spread over the
/// x and y grid axes (`z` is the caller's batch axis).
pub fn grid(n: usize, wg: usize, z: usize, max_x: u32) -> [u32; 3] {
    let groups = n.div_ceil(wg).max(1);
    let gx = groups.min(max_x as usize).max(1);
    let gy = groups.div_ceil(gx);
    [gx as u32, gy as u32, z.max(1) as u32]
}

/// Group counts for `rows` one-group-per-row launches.
fn row_grid(rows: usize, z: usize, max_x: u32) -> [u32; 3] {
    let gx = rows.min(max_x as usize).max(1);
    let gy = rows.div_ceil(gx);
    [gx as u32, gy as u32, z.max(1) as u32]
}

impl VulkanDevice {
    /// Launch `key` (compiling it with `src` on first use) over `groups`.
    ///
    /// Public so an operator living outside this crate — Joshua's `conv1d`
    /// and NVFP4 GEMV — can submit its own kernel through the backend's
    /// pipeline cache and descriptor ring instead of taking a CPU round trip
    /// or reimplementing the submission path.  Additive: nothing internal
    /// changes, and no existing caller's behaviour is affected.
    pub fn launch(
        &self,
        key: &str,
        src: impl FnOnce(usize) -> String,
        bufs: &[Buf],
        push: Push,
        ix: &[Idx],
        groups: [u32; 3],
    ) -> Result<()> {
        self.launch_spirv(key, |wg| glsl_to_spirv(&src(wg)), bufs, push, ix, groups)
    }

    /// [`Self::launch`] for a kernel whose `src` yields SPIR-V directly
    /// (e.g. through [`wgsl_to_spirv`]).
    pub fn launch_spirv(
        &self,
        key: &str,
        src: impl FnOnce(usize) -> Result<Vec<u32>>,
        bufs: &[Buf],
        push: Push,
        ix: &[Idx],
        groups: [u32; 3],
    ) -> Result<()> {
        if groups.contains(&0) {
            return Ok(());
        }
        let mut params = Vec::with_capacity(2 * std::mem::size_of::<Idx>());
        let unit = Idx::unit();
        params.extend_from_slice(ix.first().unwrap_or(&unit).bytes());
        params.extend_from_slice(ix.get(1).unwrap_or(&unit).bytes());
        self.launch_spirv_params(key, src, bufs, push, &params, groups)
    }

    // Custom parameter ABI in the existing ring: no additional device buffer
    // or queue submission. All barriers and deferred frees remain shared.
    fn launch_spirv_params(
        &self,
        key: &str,
        src: impl FnOnce(usize) -> Result<Vec<u32>>,
        bufs: &[Buf],
        push: Push,
        params: &[u8],
        groups: [u32; 3],
    ) -> Result<()> {
        if groups.contains(&0) {
            return Ok(());
        }
        let max_range = self.limits().max_storage_range;
        for b in bufs {
            if b.bytes > max_range {
                return Err(Error::Msg(format!(
                    "vulkan: a {}-byte buffer exceeds the device's maxStorageBufferRange ({max_range})",
                    b.bytes
                )));
            }
        }
        let nbuf = bufs.len() as u32;
        let pipe = self.pipeline(key, nbuf, src)?;
        let mut exec = self.exec();
        let full = exec.record(&pipe, bufs, &push.0, params, groups)?;
        if full {
            let garbage = exec.flush()?;
            self.free_garbage(garbage);
        }
        Ok(())
    }

    /// The cached pipeline for `key`, compiled with the device's `WG`.
    fn pipeline(
        &self,
        key: &str,
        nbuf: u32,
        src: impl FnOnce(usize) -> Result<Vec<u32>>,
    ) -> Result<std::sync::Arc<Pipe>> {
        let mut pipes = self.pipes();
        if let Some(p) = pipes.get(key) {
            return Ok(p.clone());
        }
        let wg = self.limits().wg;
        let layouts = self.layouts(nbuf)?;
        let pipe = create_pipe(self.ash(), &layouts, nbuf, &src(wg)?)?;
        let pipe = std::sync::Arc::new(pipe);
        pipes.insert(key.to_string(), pipe.clone());
        Ok(pipe)
    }
}

// ─── Op codes (must match glsl.rs) ───────────────────────────────────────────

pub fn unary_code(name: &str) -> Option<i32> {
    Some(match name {
        "exp" => 0,
        "log" => 1,
        "sin" => 2,
        "cos" => 3,
        "tanh" => 4,
        "neg" => 5,
        "recip" => 6,
        "sqr" => 7,
        "sqrt" => 8,
        "gelu" => 9,
        "gelu_erf" => 10,
        "erf" => 11,
        "silu" => 12,
        "abs" => 13,
        "ceil" => 14,
        "floor" => 15,
        "round" => 16,
        "relu" => 17,
        "sign" => 18,
        "sigmoid" => 19,
        _ => return None,
    })
}

const OP_AFFINE: i32 = 20;
const OP_POWF: i32 = 21;
const OP_ELU: i32 = 22;

pub fn binary_code(name: &str) -> Option<i32> {
    Some(match name {
        "add" => 0,
        "sub" => 1,
        "mul" => 2,
        "div" => 3,
        "minimum" => 4,
        "maximum" => 5,
        _ => return None,
    })
}

pub fn cmp_code(op: crate::op::CmpOp) -> i32 {
    use crate::op::CmpOp;
    match op {
        CmpOp::Eq => 0,
        CmpOp::Ne => 1,
        CmpOp::Le => 2,
        CmpOp::Ge => 3,
        CmpOp::Lt => 4,
        CmpOp::Gt => 5,
    }
}

pub const RED_SUM: i32 = 0;
pub const RED_MAX: i32 = 1;
pub const RED_MIN: i32 = 2;

// ─── Elementwise ─────────────────────────────────────────────────────────────

fn unary_launch(
    d: &VulkanDevice,
    op: i32,
    f0: f32,
    f1: f32,
    x: Buf,
    out: Buf,
    n: usize,
    l: &Layout,
) -> Result<()> {
    let (contig, off, ix) = match l.contiguous_offsets() {
        Some((o1, _)) => (1, o1, Idx::unit()),
        None => (0, 0, Idx::new(l.dims())?.with_layout(0, l)?),
    };
    let push = Push::default().us(n)?.i(contig).us(off)?.i(op).f(f0).f(f1);
    let lim = d.limits();
    if f64_dot_enabled_on(d) {
        return d.launch(
            "unary_precise",
            glsl::k_unary_precise,
            &[x, out],
            push,
            &[ix],
            grid(n, lim.wg, 1, lim.max_groups_x),
        );
    }
    d.launch(
        "unary",
        glsl::k_unary,
        &[x, out],
        push,
        &[ix],
        grid(n, lim.wg, 1, lim.max_groups_x),
    )
}

pub fn run_unary(d: &VulkanDevice, op: i32, x: Buf, out: Buf, n: usize, l: &Layout) -> Result<()> {
    unary_launch(d, op, 0.0, 0.0, x, out, n, l)
}

pub fn run_affine(
    d: &VulkanDevice,
    x: Buf,
    out: Buf,
    n: usize,
    l: &Layout,
    mul: f32,
    add: f32,
) -> Result<()> {
    unary_launch(d, OP_AFFINE, mul, add, x, out, n, l)
}

pub fn run_powf(d: &VulkanDevice, x: Buf, out: Buf, n: usize, l: &Layout, e: f32) -> Result<()> {
    unary_launch(d, OP_POWF, e, 0.0, x, out, n, l)
}

pub fn run_elu(d: &VulkanDevice, x: Buf, out: Buf, n: usize, l: &Layout, alpha: f32) -> Result<()> {
    unary_launch(d, OP_ELU, alpha, 0.0, x, out, n, l)
}

/// f32 binary op; `u32` selects the integer kernel.
pub fn run_binary(
    d: &VulkanDevice,
    op: i32,
    a: Buf,
    b: Buf,
    out: Buf,
    n: usize,
    la: &Layout,
    lb: &Layout,
    u32: bool,
) -> Result<()> {
    let (contig, oa, ob, ix) = match (la.contiguous_offsets(), lb.contiguous_offsets()) {
        (Some((oa, _)), Some((ob, _))) => (1, oa, ob, Idx::unit()),
        _ => (
            0,
            0,
            0,
            Idx::new(la.dims())?
                .with_layout(0, la)?
                .with_layout(1, lb)?,
        ),
    };
    let push = Push::default().us(n)?.i(contig).us(oa)?.us(ob)?.i(op);
    let lim = d.limits();
    if !u32 && f64_dot_enabled_on(d) {
        return d.launch(
            "binary_precise",
            glsl::k_binary_precise,
            &[a, b, out],
            push,
            &[ix],
            grid(n, lim.wg, 1, lim.max_groups_x),
        );
    }
    let key = if u32 { "binary_u32" } else { "binary_f32" };
    d.launch(
        key,
        |wg| glsl::k_binary(wg, u32),
        &[a, b, out],
        push,
        &[ix],
        grid(n, lim.wg, 1, lim.max_groups_x),
    )
}

pub fn run_cmp(
    d: &VulkanDevice,
    op: i32,
    a: Buf,
    b: Buf,
    out: Buf,
    n: usize,
    la: &Layout,
    lb: &Layout,
    u32: bool,
) -> Result<()> {
    let ix = Idx::new(la.dims())?
        .with_layout(0, la)?
        .with_layout(1, lb)?;
    let push = Push::default().us(n)?.i(op);
    let lim = d.limits();
    let key = if u32 { "cmp_u32" } else { "cmp_f32" };
    d.launch(
        key,
        |wg| glsl::k_cmp(wg, u32),
        &[a, b, out],
        push,
        &[ix],
        grid(n.div_ceil(4), lim.wg, 1, lim.max_groups_x),
    )
}

/// `where_cond` with a u8 or u32 condition over 4- or 8-byte payloads.
pub fn run_where(
    d: &VulkanDevice,
    cond: Buf,
    t: Buf,
    f: Buf,
    out: Buf,
    n: usize,
    lc: &Layout,
    lt: &Layout,
    lf: &Layout,
    cond_u32: bool,
    elem8: bool,
) -> Result<()> {
    let ix = Idx::new(lc.dims())?
        .with_layout(0, lc)?
        .with_layout(1, lt)?
        .with_layout(2, lf)?;
    let push = Push::default().us(n)?;
    let lim = d.limits();
    let key = format!(
        "where_{}_{}",
        if cond_u32 { "u32" } else { "u8" },
        if elem8 { 8 } else { 4 }
    );
    d.launch(
        &key,
        |wg| glsl::k_where(wg, cond_u32, elem8),
        &[cond, t, f, out],
        push,
        &[ix],
        grid(n, lim.wg, 1, lim.max_groups_x),
    )
}

/// Strided copy of `n` elements (`elem` bytes each) described by `l` into
/// `dst` at element offset `dst_off`, contiguous.
pub fn run_copy_strided(
    d: &VulkanDevice,
    elem: usize,
    src: Buf,
    dst: Buf,
    n: usize,
    l: &Layout,
    dst_off: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let ix = Idx::new(l.dims())?.with_layout(0, l)?;
    let lim = d.limits();
    match elem {
        4 | 8 => {
            let elem8 = elem == 8;
            let push = Push::default().us(n)?.us(dst_off)?;
            let key = if elem8 { "copy_s8" } else { "copy_s4" };
            d.launch(
                key,
                |wg| glsl::k_copy_wide(wg, elem8),
                &[src, dst],
                push,
                &[ix],
                grid(n, lim.wg, 1, lim.max_groups_x),
            )
        }
        1 | 2 => {
            let per = 4 / elem;
            let w0 = dst_off / per;
            let nw = (dst_off + n).div_ceil(per) - w0;
            let push = Push::default().us(n)?.us(dst_off)?.us(w0)?.us(nw)?;
            let key = if elem == 1 { "copy_s1" } else { "copy_s2" };
            d.launch(
                key,
                |wg| glsl::k_copy_packed(wg, elem),
                &[src, dst],
                push,
                &[ix],
                grid(nw, lim.wg, 1, lim.max_groups_x),
            )
        }
        _ => Err(Error::Msg(format!(
            "vulkan: no copy kernel for {elem}-byte elements"
        ))),
    }
}

pub fn run_copy2d(
    d: &VulkanDevice,
    elem: usize,
    src: Buf,
    dst: Buf,
    d1: usize,
    d2: usize,
    src_s: usize,
    dst_s: usize,
    src_o: usize,
    dst_o: usize,
) -> Result<()> {
    if d1 == 0 || d2 == 0 {
        return Ok(());
    }
    let lim = d.limits();
    match elem {
        4 | 8 => {
            let elem8 = elem == 8;
            let push = Push::default()
                .us(d1)?
                .us(d2)?
                .us(src_s)?
                .us(dst_s)?
                .us(src_o)?
                .us(dst_o)?;
            let key = if elem8 { "copy2d_8" } else { "copy2d_4" };
            d.launch(
                key,
                |wg| glsl::k_copy2d_wide(wg, elem8),
                &[src, dst],
                push,
                &[],
                grid(d1 * d2, lim.wg, 1, lim.max_groups_x),
            )
        }
        1 | 2 => {
            let per = 4 / elem;
            let end = dst_o + (d1 - 1) * dst_s + d2;
            let w0 = dst_o / per;
            let nw = end.div_ceil(per) - w0;
            let push = Push::default()
                .us(d1)?
                .us(d2)?
                .us(src_s)?
                .us(dst_s)?
                .us(src_o)?
                .us(dst_o)?
                .us(w0)?
                .us(nw)?;
            let key = if elem == 1 { "copy2d_1" } else { "copy2d_2" };
            d.launch(
                key,
                |wg| glsl::k_copy2d_packed(wg, elem),
                &[src, dst],
                push,
                &[],
                grid(nw, lim.wg, 1, lim.max_groups_x),
            )
        }
        _ => Err(Error::Msg(format!(
            "vulkan: no copy2d kernel for {elem}-byte elements"
        ))),
    }
}

/// Fill the elements addressed by `l` with the bit pattern `bits` (element
/// size `elem`).  1- and 2-byte fills need a contiguous layout.
pub fn run_fill(
    d: &VulkanDevice,
    elem: usize,
    dst: Buf,
    n: usize,
    l: &Layout,
    bits: u64,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let lim = d.limits();
    match elem {
        4 | 8 => {
            let elem8 = elem == 8;
            let ix = Idx::new(l.dims())?.with_layout(0, l)?;
            let push = Push::default().us(n)?.u(bits as u32).u((bits >> 32) as u32);
            let key = if elem8 { "fill_8" } else { "fill_4" };
            d.launch(
                key,
                |wg| glsl::k_fill_wide(wg, elem8),
                &[dst],
                push,
                &[ix],
                grid(n, lim.wg, 1, lim.max_groups_x),
            )
        }
        1 | 2 => {
            let Some((off, _)) = l.contiguous_offsets() else {
                return Err(Error::Msg(
                    "vulkan: strided fill of 1/2-byte elements has no kernel".into(),
                ));
            };
            let per = 4 / elem;
            let w0 = off / per;
            let nw = (off + n).div_ceil(per) - w0;
            let push = Push::default()
                .us(n)?
                .us(off)?
                .us(w0)?
                .us(nw)?
                .u(bits as u32);
            let key = if elem == 1 { "fill_1" } else { "fill_2" };
            d.launch(
                key,
                |wg| glsl::k_fill_packed(wg, elem),
                &[dst],
                push,
                &[],
                grid(nw, lim.wg, 1, lim.max_groups_x),
            )
        }
        _ => Err(Error::Msg(format!(
            "vulkan: no fill kernel for {elem}-byte elements"
        ))),
    }
}

/// The storage class of a dtype for the cast kernel.
pub fn cls_of(dtype: crate::DType) -> Option<glsl::Cls> {
    use crate::DType::*;
    Some(match dtype {
        F32 => glsl::Cls::F32,
        U32 => glsl::Cls::U32,
        I32 => glsl::Cls::I32,
        U8 => glsl::Cls::U8,
        F16 => glsl::Cls::F16,
        BF16 => glsl::Cls::BF16,
        I64 => glsl::Cls::I64,
        _ => return None,
    })
}

/// Whether a (from, to) cast has a kernel.
pub fn cast_supported(from: crate::DType, to: crate::DType) -> bool {
    match (cls_of(from), cls_of(to)) {
        (Some(f), Some(t)) => glsl::k_cast(64, f, t).is_some(),
        _ => false,
    }
}

/// i64 ids → u32 with out-of-range values saturated (see `glsl::k_ids_i64`).
pub fn run_ids_i64(d: &VulkanDevice, x: Buf, out: Buf, n: usize, l: &Layout) -> Result<()> {
    let ix = Idx::new(l.dims())?.with_layout(0, l)?;
    let push = Push::default().us(n)?;
    let lim = d.limits();
    d.launch(
        "ids_i64",
        glsl::k_ids_i64,
        &[x, out],
        push,
        &[ix],
        grid(n, lim.wg, 1, lim.max_groups_x),
    )
}

pub fn run_cast(
    d: &VulkanDevice,
    from: crate::DType,
    to: crate::DType,
    x: Buf,
    out: Buf,
    n: usize,
    l: &Layout,
) -> Result<()> {
    let (Some(f), Some(t)) = (cls_of(from), cls_of(to)) else {
        return Err(Error::Msg(format!(
            "vulkan: no cast kernel {from:?} -> {to:?}"
        )));
    };
    let ix = Idx::new(l.dims())?.with_layout(0, l)?;
    let per = match to.size_in_bytes() {
        1 => 4,
        2 => 2,
        _ => 1,
    };
    let nw = n.div_ceil(per);
    let push = Push::default().us(n)?.us(nw)?;
    let lim = d.limits();
    let key = format!("cast_{from:?}_{to:?}");
    d.launch(
        &key,
        |wg| glsl::k_cast(wg, f, t).expect("cast pair checked"),
        &[x, out],
        push,
        &[ix],
        grid(nw, lim.wg, 1, lim.max_groups_x),
    )
}

// ─── Reductions ──────────────────────────────────────────────────────────────

pub fn run_reduce_last(
    d: &VulkanDevice,
    op: i32,
    x: Buf,
    out: Buf,
    rows: usize,
    cols: usize,
    off: usize,
) -> Result<()> {
    let push = Push::default().us(rows)?.us(cols)?.us(off)?.i(op);
    let lim = d.limits();
    d.launch(
        "reduce_last",
        glsl::k_reduce_last,
        &[x, out],
        push,
        &[],
        row_grid(rows, 1, lim.max_groups_x),
    )
}

pub fn run_arg_last(
    d: &VulkanDevice,
    is_max: bool,
    x: Buf,
    out: Buf,
    rows: usize,
    cols: usize,
    off: usize,
) -> Result<()> {
    let push = Push::default()
        .us(rows)?
        .us(cols)?
        .us(off)?
        .i(is_max as i32);
    let lim = d.limits();
    d.launch(
        "arg_last",
        glsl::k_arg_last,
        &[x, out],
        push,
        &[],
        row_grid(rows, 1, lim.max_groups_x),
    )
}

pub fn run_reduce_generic(
    d: &VulkanDevice,
    op: i32,
    x: Buf,
    out: Buf,
    n: usize,
    ix: Idx,
    rd: Idx,
    count: usize,
) -> Result<()> {
    let push = Push::default().us(n)?.us(count)?.i(op);
    let lim = d.limits();
    d.launch(
        "reduce_generic",
        glsl::k_reduce_generic,
        &[x, out],
        push,
        &[ix, rd],
        grid(n, lim.wg, 1, lim.max_groups_x),
    )
}

// ─── Indexing ────────────────────────────────────────────────────────────────

/// The calling thread's fault slot, as a push constant.
fn fslot() -> i32 {
    crate::fault_slot::current() as i32
}

pub fn run_index_select(
    d: &VulkanDevice,
    elem8: bool,
    ids_i64: bool,
    src: Buf,
    ids: Buf,
    out: Buf,
    n: usize,
    left: usize,
    n_ids: usize,
    right: usize,
    dim_size: usize,
    src_off: usize,
    ids_off: usize,
) -> Result<()> {
    let push = Push::default()
        .us(n)?
        .us(left)?
        .us(n_ids)?
        .us(right)?
        .us(dim_size)?
        .us(src_off)?
        .us(ids_off)?
        .i(fslot());
    let lim = d.limits();
    let key = format!(
        "index_select_{}_{}",
        if ids_i64 { "i64" } else { "u32" },
        if elem8 { 8 } else { 4 }
    );
    d.launch(
        &key,
        |wg| glsl::k_index_select(wg, elem8, ids_i64),
        &[src, ids, out, d.fault_buf()],
        push,
        &[],
        grid(n, lim.wg, 1, lim.max_groups_x),
    )
}

pub fn run_gather(
    d: &VulkanDevice,
    src: Buf,
    ids: Buf,
    out: Buf,
    n: usize,
    ix: Idx,
    src_dim_stride: usize,
    dim_size: usize,
) -> Result<()> {
    let push = Push::default()
        .us(n)?
        .us(src_dim_stride)?
        .us(dim_size)?
        .i(fslot());
    let lim = d.limits();
    d.launch(
        "gather_4",
        glsl::k_gather,
        &[src, ids, out, d.fault_buf()],
        push,
        &[ix],
        grid(n, lim.wg, 1, lim.max_groups_x),
    )
}

/// scatter set / add: `ix` enumerates the ids space with `dim` collapsed
/// (s0 ids, s1 src, s2 dst with the dim stride zeroed); the `*_ds` are the
/// strides along `dim`.
pub fn run_scatter(
    d: &VulkanDevice,
    add: bool,
    dst: Buf,
    ids: Buf,
    src: Buf,
    n: usize,
    ix: Idx,
    n_j: usize,
    ids_ds: usize,
    src_ds: usize,
    dst_ds: usize,
    dim_size: usize,
) -> Result<()> {
    let push = Push::default()
        .us(n)?
        .us(n_j)?
        .us(ids_ds)?
        .us(src_ds)?
        .us(dst_ds)?
        .us(dim_size)?
        .i(fslot());
    let lim = d.limits();
    let key = if add {
        "scatter_add_f32"
    } else {
        "scatter_set_4"
    };
    d.launch(
        key,
        |wg| glsl::k_scatter(wg, add),
        &[dst, ids, src, d.fault_buf()],
        push,
        &[ix],
        grid(n, lim.wg, 1, lim.max_groups_x),
    )
}

pub fn run_index_add(
    d: &VulkanDevice,
    dst: Buf,
    ids: Buf,
    src: Buf,
    left: usize,
    n_ids: usize,
    right: usize,
    dim_size: usize,
    src_off: usize,
    ids_off: usize,
) -> Result<()> {
    let n_lr = left * right;
    let push = Push::default()
        .us(n_lr)?
        .us(left)?
        .us(n_ids)?
        .us(right)?
        .us(dim_size)?
        .us(src_off)?
        .us(ids_off)?
        .i(fslot());
    let lim = d.limits();
    d.launch(
        "index_add_f32",
        glsl::k_index_add,
        &[dst, ids, src, d.fault_buf()],
        push,
        &[],
        grid(n_lr, lim.wg, 1, lim.max_groups_x),
    )
}

// ─── Dense GEMM / GEMV ───────────────────────────────────────────────────────

/// Strides of one operand of a (batched) matmul.
#[derive(Clone, Copy, Debug)]
pub struct MatStrides {
    pub row: usize,
    pub col: usize,
    pub offset: usize,
    pub batch: usize,
}

/// `C[bz] = A[bz] @ B[bz]` for `batch` matrices of `(m, k) @ (k, n)`; C is
/// written contiguous `[batch, m, n]`.  Picks the GEMV kernels for `m == 1`.
pub fn run_matmul(
    d: &VulkanDevice,
    a: Buf,
    b: Buf,
    out: Buf,
    dims: (usize, usize, usize, usize),
    sa: MatStrides,
    sb: MatStrides,
) -> Result<()> {
    run_matmul_with_precision(d, a, b, out, dims, sa, sb, false)
}

pub fn run_matmul_with_precision(
    d: &VulkanDevice,
    a: Buf,
    b: Buf,
    out: Buf,
    (batch, m, n, k): (usize, usize, usize, usize),
    sa: MatStrides,
    sb: MatStrides,
    wide: bool,
) -> Result<()> {
    if batch == 0 || m == 0 || n == 0 {
        return Ok(());
    }
    if k == 0 {
        // An empty contraction is a zero matrix on every backend; the freshly
        // allocated output holds whatever the device had there, so fill it.
        let n_out = batch * m * n;
        return run_fill(d, 4, out, n_out, &Layout::contiguous(n_out), 0);
    }
    let lim = d.limits();
    if !wide && m == 1 && sa.col == 1 {
        let push = |sbx: usize| -> Result<Push> {
            Push::default()
                .us(n)?
                .us(k)?
                .us(sbx)?
                .us(sa.offset)?
                .us(sb.offset)?
                .i(0)
                .us(sa.batch)?
                .us(sb.batch)?
                .us(n)
        };
        if sb.row == 1 {
            return d.launch(
                "gemv_nt",
                glsl::k_gemv_nt,
                &[a, b, out],
                push(sb.col)?,
                &[],
                row_grid(n, batch, lim.max_groups_x),
            );
        }
        if sb.col == 1 {
            return d.launch(
                "gemv_nn",
                glsl::k_gemv_nn,
                &[a, b, out],
                push(sb.row)?,
                &[],
                grid(n, lim.wg, batch, lim.max_groups_x),
            );
        }
    }
    let tx = lim.gemm_tx;
    let tile = 4 * tx;
    let push = Push::default()
        .us(m)?
        .us(n)?
        .us(k)?
        .us(sa.row)?
        .us(sa.col)?
        .us(sb.row)?
        .us(sb.col)?
        .us(sa.offset)?
        .us(sb.offset)?
        .i(0)
        .us(sa.batch)?
        .us(sb.batch)?
        .us(m * n)?
        .i((sb.row == 1) as i32);
    let gx = n.div_ceil(tile) as u32;
    let gy = m.div_ceil(tile) as u32;
    if wide {
        if m <= 4 {
            if k >= 256 {
                return d.launch(
                    "small_matmul_reduce_f64",
                    glsl::k_small_matmul_reduce_f64,
                    &[a, b, out],
                    push,
                    &[],
                    row_grid(m * n, batch, lim.max_groups_x),
                );
            }
            return d.launch(
                "small_matmul_f64",
                glsl::k_small_matmul_f64,
                &[a, b, out],
                push,
                &[],
                grid(m * n, lim.wg, batch, lim.max_groups_x),
            );
        }
        return d.launch(
            "gemm_f64",
            |wg| glsl::k_gemm_f64(wg, tx),
            &[a, b, out],
            push,
            &[],
            [gx, gy, batch as u32],
        );
    }
    d.launch(
        "gemm",
        |wg| glsl::k_gemm(wg, tx),
        &[a, b, out],
        push,
        &[],
        [gx, gy, batch as u32],
    )
}

// ─── Fused attention-path ops ────────────────────────────────────────────────

pub fn run_softmax_last(
    d: &VulkanDevice,
    x: Buf,
    out: Buf,
    rows: usize,
    cols: usize,
    off: usize,
) -> Result<()> {
    let push = Push::default().us(rows)?.us(cols)?.us(off)?;
    let lim = d.limits();
    d.launch(
        "softmax_last",
        glsl::k_softmax_last,
        &[x, out],
        push,
        &[],
        row_grid(rows, 1, lim.max_groups_x),
    )
}

pub fn f64_rms_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CANDLE_VULKAN_RMS_F64").is_ok_and(|v| v == "1"))
}
pub fn f64_rms_enabled_on(d: &VulkanDevice) -> bool {
    f64_rms_enabled() || d.precise_math()
}

pub fn run_rms_rows_f64(
    d: &VulkanDevice,
    x: Buf,
    out: Buf,
    rows: usize,
    cols: usize,
    off: usize,
    eps: f32,
) -> Result<()> {
    let push = Push::default().us(rows)?.us(cols)?.us(off)?.i(0).f(eps);
    let lim = d.limits();
    d.launch(
        "rms_rows_f64",
        |_| glsl::k_rmsnorm_f64(64, false),
        &[x, x, out],
        push,
        &[],
        row_grid(rows, 1, lim.max_groups_x),
    )
}

pub fn run_rmsnorm(
    d: &VulkanDevice,
    x: Buf,
    alpha: Buf,
    out: Buf,
    rows: usize,
    cols: usize,
    off: usize,
    aoff: usize,
    eps: f32,
) -> Result<()> {
    let push = Push::default()
        .us(rows)?
        .us(cols)?
        .us(off)?
        .us(aoff)?
        .f(eps);
    let lim = d.limits();
    if f64_rms_enabled_on(d) {
        return d.launch(
            "rmsnorm_f64",
            |_| glsl::k_rmsnorm_f64(64, true),
            &[x, alpha, out],
            push,
            &[],
            row_grid(rows, 1, lim.max_groups_x),
        );
    }
    d.launch(
        "rmsnorm",
        glsl::k_rmsnorm,
        &[x, alpha, out],
        push,
        &[],
        row_grid(rows, 1, lim.max_groups_x),
    )
}

/// RoPE over `x [b, h, t, d]` with `cos`/`sin` `[t, d/2]` (or `[b, t, d/2]`
/// when `cs_batched`); `interleaved` selects candle's `rope_i` pairing.
pub fn run_rope(
    d: &VulkanDevice,
    interleaved: bool,
    x: Buf,
    cos: Buf,
    sin: Buf,
    out: Buf,
    (b, h, t, dd): (usize, usize, usize, usize),
    xoff: usize,
    coff: usize,
    soff: usize,
    cs_batched: bool,
) -> Result<()> {
    let n_pairs = b * h * t * (dd / 2);
    let push = Push::default()
        .us(n_pairs)?
        .us(h)?
        .us(t)?
        .us(dd)?
        .us(xoff)?
        .us(coff)?
        .us(soff)?
        .i(cs_batched as i32);
    let lim = d.limits();
    if interleaved && f64_dot_enabled_on(d) {
        return d.launch(
            "rope_i_f64",
            glsl::k_rope_i_f64,
            &[x, cos, sin, out],
            push,
            &[],
            grid(n_pairs, lim.wg, 1, lim.max_groups_x),
        );
    }
    let key = if interleaved { "rope_i" } else { "rope" };
    d.launch(
        key,
        |wg| glsl::k_rope(wg, interleaved),
        &[x, cos, sin, out],
        push,
        &[],
        grid(n_pairs, lim.wg, 1, lim.max_groups_x),
    )
}

// ─── Quantized weights ───────────────────────────────────────────────────────

/// GGUF dtype code the kernels switch on (the on-disk ggml ids).
pub fn qtype_code(dtype: crate::quantized::GgmlDType) -> i32 {
    use crate::quantized::GgmlDType::*;
    match dtype {
        F32 => 0,
        F16 => 1,
        Q4_0 => 2,
        Q4_1 => 3,
        Q5_0 => 6,
        Q5_1 => 7,
        Q8_0 => 8,
        Q8_1 => 9,
        Q2K => 10,
        Q3K => 11,
        Q4K => 12,
        Q5K => 13,
        Q6K => 14,
        Q8K => 15,
        BF16 => 30,
    }
}

/// Which quantized-GEMV kernel a device runs (see [`glsl::k_qgemv`],
/// [`glsl::k_qgemv_mali`] and [`super::wgsl::k_qgemv_mali_i8`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qgemv {
    /// One wide work-group per output element.
    Generic,
    /// Q4_K-only compatibility selection, with generic accumulation order.
    CubeCl,
    /// Validated native Q4_K/Q8_0/Q2_K GEMVs with generic accumulation order.
    /// Previous B50 dispatch, retained for matched baseline comparisons.
    CubeClExpanded,
    /// Packed formats including Q6_K, without grouped projection (A/B reference).
    CubeClPacked,
    /// B50 default: packed formats and the validated DeepSeek grouped Q8 shape.
    CubeClProduction,
    /// Opt-in packed formats plus selected-group Q8. Consult the pinned
    /// full-model benchmark reports before changing automatic dispatch.
    CubeClNext,
    /// Diagnostic ablation: use CubeCL for exactly one weight format.
    CubeClFormat(crate::quantized::GgmlDType),
    /// Arm Mali / Immortalis: small teams, several rows per decode.
    Mali,
    /// [`Qgemv::Mali`]'s shape on int8 dot products over activations
    /// quantized per 32 (llama.cpp's Q8_0 rounding); `hw_dot` uses the
    /// device's `OpSDot`, else a bit-identical polyfill.
    MaliInt8 { hw_dot: bool },
}

impl Qgemv {
    /// Arm's PCI vendor id, as `VkPhysicalDeviceProperties::vendorID`.
    pub const ARM_VENDOR_ID: u32 = 0x13B5;

    /// Default to the Rust-authored kernel only on the adapter validated by
    /// tiny-model parity, decode timing and memory checks. Other adapters keep
    /// their existing selection; an explicit environment override wins.
    pub fn for_adapter(vendor_id: u32, device_id: u32, int_dot: bool) -> Self {
        if vendor_id == 0x8086
            && device_id == 0xE212
            && std::env::var_os("CANDLE_VULKAN_QGEMV").is_none()
        {
            Self::CubeClProduction
        } else {
            Self::for_device(vendor_id, int_dot)
        }
    }

    /// The kernel for a device with `int_dot` (the `shaderIntegerDotProduct`
    /// feature enabled): `CANDLE_VULKAN_QGEMV=cubecl-next|cubecl-packed|cubecl-expanded|cubecl|mali-int8|mali|generic` when
    /// set (to force one on any device), else on Arm GPUs the int8 kernel
    /// when the device has the dot-product feature and the float one when
    /// not.
    pub fn for_device(vendor_id: u32, int_dot: bool) -> Self {
        match std::env::var("CANDLE_VULKAN_QGEMV").as_deref() {
            Ok("mali-int8") => Self::MaliInt8 { hw_dot: int_dot },
            Ok("mali") => Self::Mali,
            Ok("generic") => Self::Generic,
            Ok("cubecl") => Self::CubeCl,
            Ok("cubecl-expanded") => Self::CubeClExpanded,
            Ok("cubecl-next") => Self::CubeClNext,
            Ok("cubecl-packed") => Self::CubeClPacked,
            _ if vendor_id == Self::ARM_VENDOR_ID && int_dot => Self::MaliInt8 { hw_dot: true },
            _ if vendor_id == Self::ARM_VENDOR_ID => Self::Mali,
            _ => Self::Generic,
        }
    }
}

/// `C[m, n] = sum_k X[m, k] * W[n, k]` over block-quantized `W` (`[N, K]`),
/// with the device's [`Qgemv`] kernel.
pub fn f64_dot_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CANDLE_VULKAN_DOT_F64").is_ok_and(|v| v == "1"))
}
pub fn f64_dot_enabled_on(d: &VulkanDevice) -> bool {
    f64_dot_enabled() || d.precise_math()
}

pub fn compensated_dot_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CANDLE_VULKAN_COMPENSATED_DOT").is_ok_and(|v| v == "1"))
}
pub fn compensated_dot_enabled_on(d: &VulkanDevice) -> bool {
    compensated_dot_enabled() || d.precise_math()
}

pub fn fused_mhc_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CANDLE_VULKAN_FUSED_MHC").is_ok_and(|v| v == "1"))
}
pub fn fused_mhc_enabled_on(d: &VulkanDevice) -> bool {
    fused_mhc_enabled() || d.precise_math()
}

pub fn run_mhc_collapse(
    d: &VulkanDevice,
    x: Buf,
    pre: Buf,
    out: Buf,
    rows: usize,
    width: usize,
    xoff: usize,
    poff: usize,
) -> Result<()> {
    let len = rows
        .checked_mul(width)
        .ok_or_else(|| Error::Msg("HC collapse size overflow".into()))?;
    let p = Push::default().us(len)?.us(width)?.us(xoff)?.us(poff)?;
    d.launch(
        "mhc_collapse_f64",
        glsl::k_mhc_collapse,
        &[x, pre, out],
        p,
        &[],
        grid(len, d.limits().wg, 1, d.limits().max_groups_x),
    )
}

pub fn run_fused_mhc(
    d: &VulkanDevice,
    x: Buf,
    base: Buf,
    out: Buf,
    rows: usize,
    xoff: usize,
    boff: usize,
    iters: usize,
    eps: f32,
    scale: [f32; 3],
) -> Result<()> {
    let p = Push::default()
        .us(rows)?
        .us(xoff)?
        .us(boff)?
        .us(iters)?
        .f(eps)
        .f(scale[0])
        .f(scale[1])
        .f(scale[2]);
    let lim = d.limits();
    if rows == 1 {
        return d.launch(
            "fused_mhc_f64_serial",
            glsl::k_fused_mhc_serial,
            &[x, base, out],
            p,
            &[],
            grid(rows, 64, 1, lim.max_groups_x),
        );
    }
    d.launch(
        "fused_mhc_f64_parallel",
        glsl::k_fused_mhc,
        &[x, base, out],
        p,
        &[],
        row_grid(rows, 1, lim.max_groups_x),
    )
}

pub fn run_qgemv_f64(
    d: &VulkanDevice,
    dtype: crate::quantized::GgmlDType,
    x: Buf,
    w: Buf,
    out: Buf,
    m: usize,
    n: usize,
    k: usize,
    off: usize,
    groups: usize,
) -> Result<()> {
    let p = Push::default()
        .us(n)?
        .us(k)?
        .i(qtype_code(dtype))
        .us(dtype.block_size())?
        .us(dtype.type_size())?
        .us(off)?
        .i(0)
        .us(m)?;
    let lim = d.limits();
    let rows = if m > 1 { 4 } else { 1 };
    if compensated_dot_enabled_on(d) {
        // B50 measurements favor 32 lanes for Q2/Q4 and four for Q8.
        // Wide FP64 is both accurate and competitive for the former, while
        // selected-group Q8 decode retains its exact four-lane FP64 reduction.
        // Batched selected groups use the same compensated streams as the
        // original full prefill projection, skipping only unused outputs.
        let q8 = dtype == crate::quantized::GgmlDType::Q8_0;
        let lanes = if q8 { 4 } else { 32 };
        let comp = q8 && (groups == 1 || m > 1);
        let name = match (lanes, comp, groups, rows) {
            (4, true, 1, 1) => std::borrow::Cow::Borrowed("q8_precise_compensated"),
            (4, true, 1, 4) => std::borrow::Cow::Borrowed("q8_precise_compensated_r4"),
            (4, false, 8, 1) => std::borrow::Cow::Borrowed("grouped_q8_precise_f64_g8"),
            (32, false, 1, 1) => std::borrow::Cow::Borrowed("qgemv_f64_l32_fast"),
            (32, false, 1, 4) => std::borrow::Cow::Borrowed("qgemv_f64_l32_r4_fast"),
            _ => {
                std::borrow::Cow::Owned(format!("qgemv_fast_l{lanes}_g{groups}_r{rows}_comp{comp}"))
            }
        };
        return d.launch(
            &name,
            |_| {
                if q8 {
                    glsl::k_q8_precise(64, groups, rows, comp)
                } else {
                    glsl::k_qgemv_f64(64, groups, lanes, rows)
                }
            },
            &[x, w, out],
            p,
            &[],
            row_grid(n.div_ceil(64 / lanes), m.div_ceil(rows), lim.max_groups_x),
        );
    }
    if dtype == crate::quantized::GgmlDType::Q8_0 {
        let name = match (groups, rows) {
            (1, 1) => std::borrow::Cow::Borrowed("q8_precise_f64"),
            (1, 4) => std::borrow::Cow::Borrowed("q8_precise_f64_r4"),
            (8, 1) => std::borrow::Cow::Borrowed("grouped_q8_precise_f64_g8"),
            _ => std::borrow::Cow::Owned(format!("q8_precise_f64_g{groups}_r{rows}")),
        };
        return d.launch(
            &name,
            |_| glsl::k_q8_precise(64, groups, rows, false),
            &[x, w, out],
            p,
            &[],
            row_grid(n.div_ceil(16), m.div_ceil(rows), lim.max_groups_x),
        );
    }
    let lanes = 8;
    let name = match (lanes, groups) {
        (4, 1) => std::borrow::Cow::Borrowed("qgemv_f64_l4"),
        (8, 1) => std::borrow::Cow::Borrowed("qgemv_f64_l8"),
        (4, 8) => std::borrow::Cow::Borrowed("grouped_q8_f64_l4_g8"),
        _ => std::borrow::Cow::Owned(format!("qgemv_f64_l{lanes}_g{groups}")),
    };
    // Reuse compact block decoding across four prefill rows. The per-output
    // lane assignment and K-order are unchanged, including partial tails.
    let name = if rows == 1 {
        name
    } else {
        std::borrow::Cow::Owned(format!("{name}_r{rows}"))
    };
    d.launch(
        &name,
        |_| glsl::k_qgemv_f64(64, groups, lanes, rows),
        &[x, w, out],
        p,
        &[],
        row_grid(n.div_ceil(64 / lanes), m.div_ceil(rows), lim.max_groups_x),
    )
}

pub fn run_qgemv(
    d: &VulkanDevice,
    dtype: crate::quantized::GgmlDType,
    x: Buf,
    w: Buf,
    out: Buf,
    m: usize,
    n: usize,
    k: usize,
    xoff: usize,
) -> Result<()> {
    run_qgemv_with(d, d.limits().qgemv, dtype, x, w, out, m, n, k, xoff)
}

/// [`run_qgemv`] with an explicit kernel.
#[allow(clippy::too_many_arguments)]
pub fn run_qgemv_with(
    d: &VulkanDevice,
    kernel: Qgemv,
    dtype: crate::quantized::GgmlDType,
    x: Buf,
    w: Buf,
    out: Buf,
    m: usize,
    n: usize,
    k: usize,
    xoff: usize,
) -> Result<()> {
    let lim = d.limits();
    let cubecl = match dtype {
        crate::quantized::GgmlDType::Q4K => Some((
            256,
            144,
            "q4k_cubecl_256",
            include_str!("q4k_cubecl.wgsl"),
            false,
        )),
        crate::quantized::GgmlDType::Q8_0 => Some((
            32,
            34,
            "q8_0_cubecl_256",
            include_str!("q8_0_cubecl.wgsl"),
            true,
        )),
        crate::quantized::GgmlDType::Q2K => Some((
            256,
            84,
            "q2k_cubecl_256",
            include_str!("q2k_cubecl.wgsl"),
            true,
        )),
        crate::quantized::GgmlDType::Q6K => Some((
            256,
            210,
            "q6k_cubecl_256",
            include_str!("q6k_cubecl.wgsl"),
            true,
        )),
        _ => None,
    };
    let selected = match kernel {
        Qgemv::CubeCl => dtype == crate::quantized::GgmlDType::Q4K,
        Qgemv::CubeClExpanded => dtype != crate::quantized::GgmlDType::Q6K,
        Qgemv::CubeClPacked | Qgemv::CubeClProduction | Qgemv::CubeClNext => true,
        Qgemv::CubeClFormat(format) => dtype == format,
        _ => false,
    };
    if let Some((block, bytes, name, source, tiled)) = cubecl.filter(|_| selected) {
        // Packed shaders use one workgroup per output row. Q8_0/Q2_K also
        // guard the final tiled row, so the large DeepSeek output head fits.
        // Keep both lane assignment and reduction order equal to generic.
        if m == 1 && lim.wg == 256 && n > 0 && k > 0 && k % block == 0
            // Small hyper-connection projections benchmark faster with generic.
            && (dtype != crate::quantized::GgmlDType::Q2K || n >= 64)
            && (tiled || n <= lim.max_groups_x as usize)
            && n <= (lim.max_groups_x as usize).saturating_mul(lim.max_groups_x as usize)
            && (bytes % 4 == 0 || (k / block) % 2 == 0 || n % 2 == 0)
        {
            let weight_bytes = n.checked_mul(k / block).and_then(|v| v.checked_mul(bytes));
            let input_bytes = xoff.checked_add(k).and_then(|v| v.checked_mul(4));
            if weight_bytes.is_none_or(|v| v as u64 > w.bytes || v > u32::MAX as usize)
                || input_bytes.is_none_or(|v| v as u64 > x.bytes || v > u32::MAX as usize)
                || n as u64 * 4 > out.bytes
            {
                return Err(Error::Msg(
                    "vulkan CubeCL GEMV: invalid buffer range".into(),
                ));
            }
            let params = Push::default().us(k)?.us(n)?.us(xoff)?.i(0);
            return d.launch_spirv_params(
                name,
                |_| wgsl_to_spirv(source),
                &[w, x, out],
                Push::default(),
                &params.0,
                row_grid(n, 1, lim.max_groups_x),
            );
        }
    }
    let push = Push::default()
        .us(n)?
        .us(k)?
        .i(qtype_code(dtype))
        .us(dtype.block_size())?
        .us(dtype.type_size())?
        .us(xoff)?
        .i(0)
        .us(m)?;
    match kernel {
        Qgemv::CubeCl
        | Qgemv::CubeClExpanded
        | Qgemv::CubeClPacked
        | Qgemv::CubeClProduction
        | Qgemv::CubeClNext
        | Qgemv::CubeClFormat(_)
        | Qgemv::Generic => d.launch(
            "qgemv",
            glsl::k_qgemv,
            &[x, w, out],
            push,
            &[],
            row_grid(n, m, lim.max_groups_x),
        ),
        Qgemv::Mali => {
            let groups = row_grid(
                n.div_ceil(glsl::QGEMV_MALI_COLS),
                m.div_ceil(glsl::QGEMV_MALI_ROWS),
                lim.max_groups_x,
            );
            d.launch(
                "qgemv_mali",
                glsl::k_qgemv_mali,
                &[x, w, out],
                push,
                &[],
                groups,
            )
        }
        Qgemv::MaliInt8 { hw_dot } => {
            // Quantize the activations once, then the int8 GEMV.
            let nblk = m * k / 32;
            let xq = d.alloc(crate::DType::U32, m * k / 4)?;
            let xs = d.alloc(crate::DType::F32, nblk)?;
            let xsum = d.alloc(crate::DType::U32, 2 * nblk)?;
            let qpush = Push::default().us(nblk)?.us(xoff)?;
            d.launch(
                "quant_x8",
                glsl::k_quant_x8,
                &[x, xq.buf(), xs.buf(), xsum.buf()],
                qpush,
                &[],
                grid(nblk, lim.wg, 1, lim.max_groups_x),
            )?;
            let push = Push::default()
                .us(n)?
                .us(k)?
                .i(qtype_code(dtype))
                .us(dtype.block_size())?
                .us(dtype.type_size())?
                .i(0)
                .us(m)?;
            let groups = row_grid(
                n.div_ceil(glsl::QGEMV_MALI_COLS),
                m.div_ceil(glsl::QGEMV_MALI_ROWS),
                lim.max_groups_x,
            );
            let key = if hw_dot {
                "qgemv_mali_i8_hw"
            } else {
                "qgemv_mali_i8"
            };
            d.launch_spirv(
                key,
                |_| wgsl_to_spirv(&super::wgsl::k_qgemv_mali_i8(hw_dot)),
                &[xq.buf(), xs.buf(), xsum.buf(), w, out],
                push,
                &[],
                groups,
            )
        }
    }
}

/// Selected diagonal blocks of `X[groups,K] @ W[N,K]^T` in group order.
/// Same 256-lane reduction as the generic multi-row GEMV, without off-diagonals.
#[allow(clippy::too_many_arguments)]
pub fn run_grouped_q8(
    d: &VulkanDevice,
    x: Buf,
    w: Buf,
    out: Buf,
    groups: usize,
    n: usize,
    k: usize,
    xoff: usize,
) -> Result<()> {
    let lim = d.limits();
    let weights = n.checked_mul(k / 32).and_then(|v| v.checked_mul(34));
    let inputs = groups
        .checked_mul(k)
        .and_then(|v| v.checked_add(xoff))
        .and_then(|v| v.checked_mul(4));
    if !grouped_q8_eligible(d, groups, n, k)
        || weights.is_none_or(|v| v as u64 > w.bytes || v > u32::MAX as usize)
        || inputs.is_none_or(|v| v as u64 > x.bytes || v > u32::MAX as usize)
        || n as u64 * 4 > out.bytes
    {
        return Err(Error::Msg(
            "vulkan grouped Q8: invalid geometry or buffer range".into(),
        ));
    }
    if f64_dot_enabled_on(d) {
        return run_qgemv_f64(
            d,
            crate::quantized::GgmlDType::Q8_0,
            x,
            w,
            out,
            1,
            n,
            k,
            xoff,
            groups,
        );
    }
    let params = Push::default().us(k)?.us(n)?.us(xoff)?.us(n / groups)?;
    d.launch_spirv_params(
        "grouped_q8_cubecl_256",
        |_| wgsl_to_spirv(include_str!("grouped_q8_cubecl.wgsl")),
        &[w, x, out],
        Push::default(),
        &params.0,
        row_grid(n, 1, lim.max_groups_x),
    )
}

/// Compact precise prefill inherits the validated model's immutable device
/// policy. Explicit overrides retain the matched old/candidate controls.
pub fn precise_prefill_optimized(d: &VulkanDevice, rows: usize) -> bool {
    static OVERRIDE: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    let enabled = OVERRIDE
        .get_or_init(|| std::env::var_os("CANDLE_VULKAN_PREFILL_OPT").map(|v| v == "1"))
        .unwrap_or(d.precise_math());
    enabled
        && native_enabled()
        && (2..=32).contains(&rows)
        && compensated_dot_enabled_on(d)
        && f64_dot_enabled_on(d)
}

#[allow(clippy::too_many_arguments)]
pub fn run_grouped_q8_prefill(
    d: &VulkanDevice,
    x: Buf,
    w: Buf,
    out: Buf,
    rows: usize,
    groups: usize,
    n: usize,
    k: usize,
    xoff: usize,
) -> Result<()> {
    let weights = n.checked_mul(k / 32).and_then(|v| v.checked_mul(34));
    let inputs = rows
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(k))
        .and_then(|v| v.checked_add(xoff))
        .and_then(|v| v.checked_mul(4));
    let outputs = rows.checked_mul(n).and_then(|v| v.checked_mul(4));
    if !precise_prefill_optimized(d, rows)
        || !grouped_q8_eligible(d, groups, n, k)
        || weights.is_none_or(|v| v as u64 > w.bytes || v > u32::MAX as usize)
        || inputs.is_none_or(|v| v as u64 > x.bytes || v > u32::MAX as usize)
        || outputs.is_none_or(|v| v as u64 > out.bytes || v > u32::MAX as usize)
    {
        return Err(Error::Msg(
            "vulkan grouped Q8 prefill: invalid geometry or buffer range".into(),
        ));
    }
    d.exec().batch_limit = PREFILL_MAX_LAUNCHES;
    run_qgemv_f64(
        d,
        crate::quantized::GgmlDType::Q8_0,
        x,
        w,
        out,
        rows,
        n,
        k,
        xoff,
        groups,
    )
}

pub fn grouped_q8_eligible(d: &VulkanDevice, groups: usize, n: usize, k: usize) -> bool {
    let lim = d.limits();
    // Larger original batches use GEMM, whose reduction order differs.
    // Production is restricted to the geometry measured in the full-model
    // B50 gate: X[8,4096], W[8192,4096], 1024 selected outputs per group.
    // Other geometries remain available only through the explicit Next opt-in.
    let selected = match lim.qgemv {
        Qgemv::CubeClProduction => (groups, n, k) == (8, 8192, 4096),
        Qgemv::CubeClNext => true,
        _ => false,
    };
    selected
        && lim.wg == 256
        && (1..=super::QGEMV_MAX_ROWS).contains(&groups)
        && n > 0
        && n % groups == 0
        && k > 0
        && k % 32 == 0
        && (k / 32 % 2 == 0 || n % 2 == 0)
        && n <= (lim.max_groups_x as usize).saturating_mul(lim.max_groups_x as usize)
}

/// `C[m, n] = sum_k X[m, k] * W[n, k]` over f16 / bf16 `W`.
pub fn run_hgemv(
    d: &VulkanDevice,
    bf16: bool,
    x: Buf,
    w: Buf,
    out: Buf,
    m: usize,
    n: usize,
    k: usize,
    xoff: usize,
) -> Result<()> {
    let push = Push::default()
        .us(n)?
        .us(k)?
        .i(bf16 as i32)
        .us(xoff)?
        .i(0)
        .us(m)?;
    let lim = d.limits();
    d.launch(
        "hgemv",
        glsl::k_hgemv,
        &[x, w, out],
        push,
        &[],
        row_grid(n, m, lim.max_groups_x),
    )
}

/// Dequantize `elem_count` elements of blocks to f32.
pub fn run_dequant(
    d: &VulkanDevice,
    dtype: crate::quantized::GgmlDType,
    w: Buf,
    out: Buf,
    elem_count: usize,
) -> Result<()> {
    use crate::quantized::GgmlDType::*;
    let lim = d.limits();
    match dtype {
        F16 | BF16 => {
            let push = Push::default().us(elem_count)?.i((dtype == BF16) as i32);
            d.launch(
                "dequant_half",
                glsl::k_dequant_half,
                &[w, out],
                push,
                &[],
                grid(elem_count, lim.wg, 1, lim.max_groups_x),
            )
        }
        F32 => Err(Error::Msg(
            "vulkan: f32 weights need no dequantization".into(),
        )),
        _ => {
            let nsub = elem_count / 32;
            let push = Push::default()
                .us(nsub)?
                .i(qtype_code(dtype))
                .us(dtype.block_size())?
                .us(dtype.type_size())?;
            d.launch(
                "dequant",
                glsl::k_dequant,
                &[w, out],
                push,
                &[],
                grid(nsub, lim.wg, 1, lim.max_groups_x),
            )
        }
    }
}

/// Gather rows of an f16 / bf16 `[vocab, K]` table into f32 `[n_ids, K]`.
pub fn run_hembed(
    d: &VulkanDevice,
    bf16: bool,
    w: Buf,
    ids: Buf,
    out: Buf,
    n_ids: usize,
    k: usize,
    vocab: usize,
    ids_off: usize,
) -> Result<()> {
    let n = n_ids * k;
    let push = Push::default()
        .us(n)?
        .us(k)?
        .i(bf16 as i32)
        .us(ids_off)?
        .us(vocab)?
        .i(fslot());
    let lim = d.limits();
    d.launch(
        "hembed",
        glsl::k_hembed,
        &[w, ids, out, d.fault_buf()],
        push,
        &[],
        grid(n, lim.wg, 1, lim.max_groups_x),
    )
}

/// Gather rows of a block-quantized `[vocab, K]` table into f32 `[n_ids, K]`.
pub fn run_qembed(
    d: &VulkanDevice,
    dtype: crate::quantized::GgmlDType,
    w: Buf,
    ids: Buf,
    out: Buf,
    n_ids: usize,
    k: usize,
    vocab: usize,
    ids_off: usize,
) -> Result<()> {
    let push = Push::default()
        .us(n_ids)?
        .us(k)?
        .i(qtype_code(dtype))
        .us(dtype.block_size())?
        .us(dtype.type_size())?
        .us(ids_off)?
        .us(vocab)?
        .i(fslot());
    let lim = d.limits();
    d.launch(
        "qembed",
        glsl::k_qembed,
        &[w, ids, out, d.fault_buf()],
        push,
        &[],
        row_grid(n_ids, 1, lim.max_groups_x),
    )
}

/// Type alias for the pipeline cache map.
pub type PipeMap = HashMap<String, std::sync::Arc<Pipe>>;

#[cfg(test)]
mod barrier_tests {
    use super::{glsl, glsl_to_spirv};
    use std::collections::HashMap;

    // Inspect the emitted SPIR-V, so this catches a regression in either
    // Naga's frontend or writer even on CI without a Vulkan device.
    fn assert_workgroup_barriers(source: &str) {
        let words = glsl_to_spirv(source).unwrap();
        let mut constants = HashMap::new();
        let mut barriers = Vec::new();
        let mut cursor = 5; // SPIR-V module header.
        while cursor < words.len() {
            let count = (words[cursor] >> 16) as usize;
            let opcode = words[cursor] & 0xffff;
            assert!(count > 0 && cursor + count <= words.len());
            let args = &words[cursor + 1..cursor + count];
            match opcode {
                43 if args.len() == 3 => {
                    // OpConstant: type, result id, literal.
                    constants.insert(args[1], args[2]);
                }
                224 => barriers.push((args[0], args[2])), // OpControlBarrier.
                _ => {}
            }
            cursor += count;
        }
        assert!(
            !barriers.is_empty(),
            "expected at least one control barrier"
        );
        for (scope, semantics) in barriers {
            assert_eq!(
                constants[&scope], 2,
                "barrier must use Workgroup execution scope"
            );
            // AcquireRelease (0x8) and WorkgroupMemory (0x100).
            assert_eq!(constants[&semantics] & 0x108, 0x108);
        }
    }

    /// The int8 GEMV compiles through naga's WGSL front end; with `hw_dot`
    /// its dot products are `OpSDot` (opcode 4450), without it none are.
    #[test]
    fn wgsl_int8_gemv_emits_sdot_only_with_hw_dot() {
        let count = |hw: bool| {
            let words = super::wgsl_to_spirv(&super::super::wgsl::k_qgemv_mali_i8(hw)).unwrap();
            let (mut cursor, mut n) = (5, 0);
            while cursor < words.len() {
                let count = (words[cursor] >> 16) as usize;
                n += usize::from(words[cursor] & 0xffff == 4450);
                cursor += count.max(1);
            }
            n
        };
        assert!(count(true) > 0, "hw_dot must emit OpSDot");
        assert_eq!(count(false), 0, "the polyfill must not need the feature");
        glsl_to_spirv(&glsl::k_quant_x8(64)).unwrap();
    }

    #[test]
    fn glsl_barriers_synchronize_whole_workgroups() {
        for wg in [64, 128, 256] {
            for source in [
                glsl::k_softmax_last(wg),
                glsl::k_rmsnorm(wg),
                glsl::k_reduce_last(wg),
                glsl::k_arg_last(wg),
                glsl::k_gemv_nt(wg),
                glsl::k_qgemv(wg),
                glsl::k_qgemv_mali(wg),
                glsl::k_hgemv(wg),
                glsl::k_gemm(wg, 8),
                glsl::k_gemm(wg, 16),
            ] {
                assert_workgroup_barriers(&source);
            }
        }
        for source in [
            glsl::k_small_matmul_reduce_f64(64),
            glsl::k_qgemv_f64(64, 1, 4, 4),
            glsl::k_qgemv_f64(64, 1, 8, 4),
            glsl::k_qgemv_compensated(64, 1, 32, 4),
            glsl::k_q8_precise(64, 1, 4, true),
            glsl::k_q8_precise(64, 8, 1, false),
            glsl::k_q8_precise(64, 8, 4, true),
            glsl::k_fused_mhc(64),
        ] {
            assert_workgroup_barriers(&source);
        }
        // Cover helper functions and nested control flow as well as the
        // reduction loops in the actual kernels above.
        assert_workgroup_barriers(
            r#"#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform PC { int mode; } pc;
void sync_group() { barrier(); }
void main() {
    { barrier(); }
    if (pc.mode == 0) { barrier(); } else { sync_group(); }
    switch (pc.mode) {
        case 1: barrier(); break;
        default: barrier(); break;
    }
    for (int i = 0; i < 2; i++) { barrier(); }
}
"#,
        );
    }
}

pub fn run_precise_unary(
    d: &VulkanDevice,
    x: Buf,
    out: Buf,
    n: usize,
    off: usize,
    op: u32,
) -> Result<()> {
    let p = Push::default().us(n)?.us(off)?.i(op as i32);
    let lim = d.limits();
    d.launch(
        "precise_unary",
        glsl::k_precise_unary,
        &[x, out],
        p,
        &[],
        grid(n, lim.wg, 1, lim.max_groups_x),
    )
}
pub fn run_precise_rows(
    d: &VulkanDevice,
    x: Buf,
    out: Buf,
    rows: usize,
    cols: usize,
    inner: usize,
    off: usize,
    soft: bool,
) -> Result<()> {
    let p = Push::default()
        .us(rows)?
        .us(cols)?
        .us(inner)?
        .us(off)?
        .i(soft as i32);
    let lim = d.limits();
    d.launch(
        "precise_rows",
        |_| glsl::k_precise_rows(64),
        &[x, out],
        p,
        &[],
        grid(rows, 64, 1, lim.max_groups_x),
    )
}

#[cfg(test)]
mod precise_perf_tests {
    use super::*;
    use crate::{
        backend::{BackendDevice, BackendStorage},
        CpuStorage, DType, VulkanStorage,
    };
    use std::time::Instant;

    #[test]
    #[ignore = "manual native B50 kernel comparison; requires CANDLE_VULKAN_DOT_F64=1"]
    fn native_precise_kernel_comparison() -> Result<()> {
        let d = VulkanDevice::new(0)?;
        assert!(f64_dot_enabled());
        let before = fallback_count();
        let (m, n, k) = (1, 272, 5120);
        let av: Vec<f32> = (0..k)
            .map(|i| ((i * 17 % 127) as f32 / 31.7).sin())
            .collect();
        let bv: Vec<f32> = (0..n * k)
            .map(|i| ((i * 31 % 251) as f32 / 63.7).cos())
            .collect();
        let a = VulkanStorage::from_slice(&av, &d)?;
        let b = VulkanStorage::from_slice(&bv, &d)?;
        let out = d.alloc(DType::F32, m * n)?;
        let p = || -> Result<Push> {
            Ok(Push::default()
                .us(m)?
                .us(n)?
                .us(k)?
                .us(k)?
                .i(1)
                .us(n)?
                .i(1)
                .i(0)
                .i(0)
                .i(0)
                .i(0)
                .i(0)
                .us(m * n)?
                .i(0))
        };
        let lim = d.limits();
        let tile = 4 * lim.gemm_tx;
        let mut reference = None::<Vec<f32>>;
        for method in ["direct", "reduce", "tiled"] {
            let call = || -> Result<()> {
                match method {
                    "direct" => d.launch(
                        "micro_direct",
                        glsl::k_small_matmul_f64,
                        &[a.buf(), b.buf(), out.buf()],
                        p()?,
                        &[],
                        grid(m * n, lim.wg, 1, lim.max_groups_x),
                    ),
                    "reduce" => d.launch(
                        "micro_reduce",
                        glsl::k_small_matmul_reduce_f64,
                        &[a.buf(), b.buf(), out.buf()],
                        p()?,
                        &[],
                        row_grid(m * n, 1, lim.max_groups_x),
                    ),
                    _ => d.launch(
                        "micro_tiled",
                        |wg| glsl::k_gemm_f64(wg, lim.gemm_tx),
                        &[a.buf(), b.buf(), out.buf()],
                        p()?,
                        &[],
                        [n.div_ceil(tile) as u32, 1, 1],
                    ),
                }
            };
            for _ in 0..3 {
                call()?;
                d.synchronize()?;
            }
            let start = Instant::now();
            for _ in 0..30 {
                call()?;
                d.synchronize()?;
            }
            let ms = start.elapsed().as_secs_f64() * 1000. / 30.;
            let CpuStorage::F32(values) = out.to_cpu_storage()? else {
                unreachable!()
            };
            let delta = reference
                .as_ref()
                .map(|r| {
                    r.iter()
                        .zip(&values)
                        .map(|(&x, &y)| (x - y).abs())
                        .fold(0., f32::max)
                })
                .unwrap_or(0.);
            if reference.is_none() {
                reference = Some(values);
            }
            println!("MICRO router {method} ms={ms:.6} max_abs={delta:e}");
        }
        use crate::quantized::{GgmlDType, QTensor};
        for (dtype, n, k, groups) in [
            (GgmlDType::Q2K, 24, 20480, 1),
            (GgmlDType::Q4K, 2304, 5120, 1),
            (GgmlDType::Q8_0, 5120, 8192, 1),
            (GgmlDType::Q8_0, 8192, 4096, 8),
        ] {
            let weights: Vec<f32> = (0..n * k)
                .map(|i| ((i * 31 % 251) as f32 - 125.) / 63.7)
                .collect();
            let q = QTensor::quantize(
                &crate::Tensor::from_vec(weights, (n, k), &crate::Device::Cpu)?,
                dtype,
            )?;
            let data = q.data()?;
            let w = VulkanStorage::from_slice(data.as_ref(), &d)?;
            let xv: Vec<f32> = (0..groups * k)
                .map(|i| ((i * 17 % 127) as f32 / 31.7).sin())
                .collect();
            let x = VulkanStorage::from_slice(&xv, &d)?;
            let out = d.alloc(DType::F32, n)?;
            let p = || -> Result<Push> {
                Ok(Push::default()
                    .us(n)?
                    .us(k)?
                    .i(qtype_code(dtype))
                    .us(dtype.block_size())?
                    .us(dtype.type_size())?
                    .i(0)
                    .i(0)
                    .i(1))
            };
            let mut reference = None::<Vec<f32>>;
            for (method, lanes) in [
                (0, if dtype == GgmlDType::Q8_0 { 4 } else { 8 }),
                (0, 32),
                (1, if dtype == GgmlDType::Q8_0 { 4 } else { 8 }),
                (1, 32),
                (2, if dtype == GgmlDType::Q8_0 { 4 } else { 8 }),
                (2, 32),
                (3, 4),
                (4, 4),
            ] {
                if method >= 3 && dtype != GgmlDType::Q8_0 {
                    continue;
                }
                let name = format!("micro_q_{dtype:?}_{groups}_{method}_{lanes}");
                let call = || {
                    d.launch(
                        &name,
                        |_| match method {
                            0 => glsl::k_qgemv_f64(64, groups, lanes, 1),
                            1 => glsl::k_qgemv_compensated(64, groups, lanes, 1),
                            2 => glsl::k_qgemv_compensated_block(64, groups, lanes, 1),
                            _ => glsl::k_q8_precise(64, groups, 1, method == 4),
                        },
                        &[x.buf(), w.buf(), out.buf()],
                        p()?,
                        &[],
                        row_grid(n.div_ceil(64 / lanes), 1, lim.max_groups_x),
                    )
                };
                for _ in 0..3 {
                    call()?;
                    d.synchronize()?;
                }
                let start = Instant::now();
                for _ in 0..20 {
                    call()?;
                    d.synchronize()?;
                }
                let ms = start.elapsed().as_secs_f64() * 1000. / 20.;
                let CpuStorage::F32(values) = out.to_cpu_storage()? else {
                    unreachable!()
                };
                let delta = reference
                    .as_ref()
                    .map(|r| {
                        r.iter()
                            .zip(&values)
                            .map(|(&x, &y)| (x - y).abs())
                            .fold(0., f32::max)
                    })
                    .unwrap_or(0.);
                if reference.is_none() {
                    reference = Some(values);
                }
                println!("MICRO quant {dtype:?} groups={groups} method={method} lanes={lanes} ms={ms:.6} max_abs={delta:e}");
            }
        }
        for rows in [1, 32] {
            let xv: Vec<f32> = (0..rows * 24)
                .map(|i| ((i * 17 % 251) as f32 - 125.) / 11.7)
                .collect();
            let bv: Vec<f32> = (0..24)
                .map(|i| ((i * 31 % 53) as f32 - 26.) / 11.3)
                .collect();
            let x = VulkanStorage::from_slice(&xv, &d)?;
            let b = VulkanStorage::from_slice(&bv, &d)?;
            let out = d.alloc(DType::F32, rows * 24)?;
            let p = || -> Result<Push> {
                Ok(Push::default()
                    .us(rows)?
                    .i(0)
                    .i(0)
                    .i(20)
                    .f(1e-6)
                    .f(0.173)
                    .f(-0.37)
                    .f(0.61))
            };
            let mut reference = None::<Vec<f32>>;
            for parallel in [false, true] {
                let key = if parallel {
                    "micro_mhc_parallel"
                } else {
                    "micro_mhc_serial"
                };
                let call = || {
                    d.launch(
                        key,
                        |wg| {
                            if parallel {
                                glsl::k_fused_mhc(wg)
                            } else {
                                glsl::k_fused_mhc_serial(wg)
                            }
                        },
                        &[x.buf(), b.buf(), out.buf()],
                        p()?,
                        &[],
                        if parallel {
                            row_grid(rows, 1, lim.max_groups_x)
                        } else {
                            grid(rows, 64, 1, lim.max_groups_x)
                        },
                    )
                };
                for _ in 0..3 {
                    call()?;
                    d.synchronize()?;
                }
                let start = Instant::now();
                for _ in 0..30 {
                    call()?;
                    d.synchronize()?;
                }
                let ms = start.elapsed().as_secs_f64() * 1000. / 30.;
                let CpuStorage::F32(values) = out.to_cpu_storage()? else {
                    unreachable!()
                };
                if let Some(r) = reference.as_ref() {
                    assert_eq!(r, &values);
                } else {
                    reference = Some(values);
                }
                println!("MICRO mhc rows={rows} parallel={parallel} ms={ms:.6}");
            }
        }
        assert_eq!(fallback_count(), before);
        Ok(())
    }
}
