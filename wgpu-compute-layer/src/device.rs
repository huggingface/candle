use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use std::hash::Hash;

use tracing::instrument;
use wgpu::{Backends, InstanceDescriptor, InstanceFlags};

use crate::DType;
use crate::{shader_loader, wgpu_functions};

#[cfg(feature = "wgpu_debug")]
use super::debug_info::{self, MInfo, Measurements, PerformanceMeasurmentDebugInfo, ShaderInfo};

use super::cache::{
    BindGroupReference, BindgroupAlignment, BindgroupAlignmentLayout, BindgroupInputBase,
    BindgroupLayouts, BufferReferenceId, ModelCache,
};
use super::queue_buffer::{MlQueue, MlQueueDispatch, PipelineReference};
use super::util::ToU64;
use super::WgpuStorage;
use crate::queue_buffer::QueueBufferInner;

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
#[cfg_attr(
    any(feature = "wgpu_debug_serialize", feature = "wgpu_debug"),
    derive(serde::Serialize, serde::Deserialize)
)]
/// Records a single pipeline invocation for debugging and replay purposes.
///
/// This structure captures the dispatch parameters along with the
/// pipeline and bind group references used at the time of execution. When
/// debug features are enabled, it can be serialized to allow inspection,
/// logging, or offline analysis of GPU command streams.
pub struct DebugPipelineRecording {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub pipeline: PipelineReference,
    pub meta: Vec<u32>,
    pub bindgroup: BindGroupReference,
    pub count: u32,
}

pub struct InternalCounter {
    pub buffer_counter: u32,
    pub buffer_reuse_counter : u32,
    pub bindgroup_counter: u32,
    pub cached_bindgroup_use_counter: u32,
    pub unary_inplace_counter: u32,
    pub binary_inplace_counter: u32,
    pub copy_inplace_counter: u32,
}

pub enum Backend {
    /// Vulkan API (Windows, Linux, Android, MacOS via `vulkan-portability`/MoltenVK)
    Vulkan = 1,
    /// Metal API (Apple platforms)
    Metal = 2,
    /// Direct3D-12 (Windows)
    Dx12 = 3,
    /// OpenGL 3.3+ (Windows), OpenGL ES 3.0+ (Linux, Android, MacOS via Angle), and WebGL2
    Gl = 4,
    /// WebGPU in the browser
    BrowserWebGpu = 5,
}

#[derive(Debug, Clone, std::marker::Copy)]
pub struct WgpuBackends(pub u32);

impl WgpuBackends {
    pub fn vulkan() -> Self {
        WgpuBackends(1 << Backend::Vulkan as u32)
    }

    pub fn gl() -> Self {
        WgpuBackends(1 << Backend::Gl as u32)
    }

    pub fn metal() -> Self {
        WgpuBackends(1 << Backend::Metal as u32)
    }

    pub fn dx12() -> Self {
        WgpuBackends(1 << Backend::Dx12 as u32)
    }

    pub fn browser_webgpu() -> Self {
        WgpuBackends(1 << Backend::BrowserWebGpu as u32)
    }

    pub fn primary() -> Self {
        Self::vulkan() | Self::metal() | Self::dx12() | Self::browser_webgpu()
    }

    pub fn secondary() -> Self {
        Self::gl()
    }
}

impl Default for WgpuBackends {
    fn default() -> Self {
        WgpuBackends::primary() | WgpuBackends::secondary()
    }
}

impl std::ops::BitOr for WgpuBackends {
    type Output = WgpuBackends;

    fn bitor(self, rhs: Self) -> Self::Output {
        WgpuBackends(self.0 | rhs.0)
    }
}

impl std::ops::BitAnd for WgpuBackends {
    type Output = bool;

    fn bitand(self, rhs: Self) -> Self::Output {
        (self.0 & rhs.0) > 0
    }
}

#[derive(Debug)]
pub struct WgpuDeviceConfig {
    ///the size of the buffer used for storing meta information (e.g. input layouts)
    pub meta_buffer_size: u32,
    ///specifies the maximum number of floating point operations to be queued in a single command buffer.
    ///(For example, a matrix multiplication of 1000x1000 * 1000x1000 would be 1,000,000 operations,
    ///so only 2 of these multiplications can be queued in a command buffer if max_workload_size is set to 2,000,000).
    pub max_workload_size: u64,
    ///Maximum size for cached wgpu::buffers. When this size is reached, free buffers will be deleted until only 75% of this maximum size is used.
    ///if this value is too low for the desired model, performance may drop significantly (e.g. the model requires at least 2gb of data, if this value is e.g. 100mb, all free buffers will be cleared after every command).
    pub buffer_cached_max_allowed_size: u64,

    ///Whether created buffers are cached and reused.
    ///If set to false, a new wgpu::Buffer is created for each tensor used.
    pub use_cache: bool,

    ///When data is copied from the CPU to the WGPU device, all previous commands may be flushed to free up other buffers for reuse.
    ///However, on a webGPU this may not be optimal as we cannot wait for commands to finish (as this function is not asynchronous).
    pub flush_gpu_before_buffer_init: bool,

    ///The buffers used for previously flushed gpu commands are cached to improve performance when finding buffers for future calls of the same model.  
    ///buffer_mapping_size' specifies how many previously flushed gpu commands are cached.
    pub buffer_mapping_size: u32,

    ///Defines the backend to use (Vulkan, Metal, Dx12,GL or WebGpu)
    pub backend: WgpuBackends,
}

impl Default for WgpuDeviceConfig {
    fn default() -> WgpuDeviceConfig {
        WgpuDeviceConfig {
            meta_buffer_size: 10 * 1024 * 1024,
            max_workload_size: 1024u64 * 1024 * 1024 * 2,
            buffer_cached_max_allowed_size: ((1024.0 * 1024.0 * 1024.0) * (7.3)) as u64,
            use_cache: true,
            flush_gpu_before_buffer_init: true,
            buffer_mapping_size: 3,
            backend: WgpuBackends::metal()
                | WgpuBackends::vulkan()
                | WgpuBackends::browser_webgpu(), //directx shader compilation is much slower than vulkan. (like 300secs vs 5s there is a faster copmiler, but this would need additional .dlls, and with this compilations needs 30s as well)
        }
    }
}

#[derive(Debug)]
pub struct WgpuDeviceInner {
    pub device: wgpu::Device,
    pub backend: wgpu::Backend,
    pub device_limits: wgpu::Limits, //we cache the limits here, because device.limit() was relatively slow on the browser
    device_features: wgpu::Features,

    pub(crate) queue: wgpu::Queue,

    pub(crate) command_queue: Mutex<QueueBufferInner>,
    pub(crate) meta_buffer: wgpu::Buffer, //buffer for storing meta information

    pub(crate) bindgroup_layouts: BindgroupLayouts,

    pub(crate) cache: Mutex<ModelCache>, //if cache is set, all commands are not queued to the gpu, but are cached inside ModelCache, so there can be reused later on
    //debug counter
    #[cfg(feature = "wgpu_debug")]
    pub(crate) debug: PerformanceMeasurmentDebugInfo,

    pub(crate) configuration: WgpuDeviceConfig,

    extensions: Mutex<DeviceExtensions>,

    #[cfg(target_arch = "wasm32")]
    pub(crate) submission_tracker: Arc<SubmissionTracker>, //alows to wait async for a speficic wgpu submission
}

#[derive(Debug)]
#[cfg(target_arch = "wasm32")]
pub(crate) struct SubmissionTracker {
    pub(crate) next_id: std::sync::atomic::AtomicU64,
    pub(crate) completed_id: std::sync::atomic::AtomicU64,
    pub(crate) tx: flume::Sender<u64>,
    pub(crate) rx: flume::Receiver<u64>,
}

#[derive(Debug)]
struct DeviceExtensions {
    map: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl DeviceExtensions {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    fn get<T: Any>(&self) -> Option<&T> {
        self.map.get(&TypeId::of::<T>())?.downcast_ref::<T>()
    }

    fn get_mut<T: Any>(&mut self) -> Option<&mut T> {
        self.map.get_mut(&TypeId::of::<T>())?.downcast_mut::<T>()
    }

    fn insert<T: Any + Send + Sync>(&mut self, value: T) {
        self.map.insert(TypeId::of::<T>(), Box::new(value));
    }
}

#[derive(Debug, Clone)]
pub struct WgpuDevice {
    pub inner: Arc<WgpuDeviceInner>,
}

impl std::ops::Deref for WgpuDevice {
    type Target = WgpuDeviceInner;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[cfg(feature = "wgpu_debug")]
pub struct WgpuDebugInfo {
    pub duration: u64,
    pub output_size: u64,
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl WgpuDevice {
    #[instrument]
    pub async fn create(_: usize, configuration: WgpuDeviceConfig) -> crate::Result<Self> {
        let mut backend = wgpu::Backends::empty();
        if configuration.backend & WgpuBackends::vulkan() {
            backend.insert(Backends::VULKAN);
        }
        if configuration.backend & WgpuBackends::dx12() {
            backend.insert(Backends::DX12);
        }
        if configuration.backend & WgpuBackends::gl() {
            backend.insert(Backends::GL);
        }
        if configuration.backend & WgpuBackends::metal() {
            backend.insert(Backends::METAL);
        }
        if configuration.backend & WgpuBackends::browser_webgpu() {
            backend.insert(Backends::BROWSER_WEBGPU);
        }

        let instance = wgpu::Instance::new(&InstanceDescriptor {
            backends: backend,
            flags: InstanceFlags::default(),
            backend_options: wgpu::BackendOptions {
                dx12: wgpu::Dx12BackendOptions {
                    shader_compiler: wgpu::Dx12Compiler::Fxc,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        });

        // `request_adapter` instantiates the general connection to the GPU
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .expect("no adapter could be requested");

        let mut limits = wgpu::Limits::downlevel_defaults();

        #[cfg(feature = "wgpu_debug")]
        let mut features = wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        #[cfg(not(feature = "wgpu_debug"))]
        let mut features = wgpu::Features::empty();

        let adapter_features = adapter.features();
        let adatper_limits = adapter.limits();

        limits.min_storage_buffer_offset_alignment =
            adatper_limits.min_storage_buffer_offset_alignment;
        limits.max_storage_buffers_per_shader_stage = 5;
        limits.max_storage_buffer_binding_size = adatper_limits.max_storage_buffer_binding_size; //use as much as possible
        limits.max_buffer_size = adatper_limits.max_buffer_size; //use as much as possible
        if adapter_features.contains(wgpu::Features::SHADER_INT64) {
            features.insert(wgpu::Features::SHADER_INT64);
        }
        if adapter_features.contains(wgpu::Features::SHADER_F64) {
            features.insert(wgpu::Features::SHADER_F64);
        }
        if adapter_features.contains(wgpu::Features::SHADER_F16) {
            features.insert(wgpu::Features::SHADER_F16);
        }
        if adapter_features.contains(wgpu::Features::SHADER_I16) {
            features.insert(wgpu::Features::SHADER_I16);
        }

        // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
        //  `features` being the available features.
        log::debug!("Request Device");
        log::debug!("Features: {:?}", features);
        log::debug!("Limits: {:?}", limits);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: features,
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .map_err(|err| err.to_string())?;
        log::info!("Device Requested");

        #[cfg(feature = "wgpu_debug")]
        let debug_info = super::debug_info::PerformanceMeasurmentDebugInfo::new(&device);

        let meta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: configuration.meta_buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let device_limits = device.limits();
        let bindgroup_layouts = BindgroupLayouts::new(&device);

        let max_memory_size: u64 = device_limits
            .max_buffer_size
            .min(device_limits.max_storage_buffer_binding_size as u64);

        #[cfg(target_arch = "wasm32")]
        let (tx, rx) = flume::unbounded();
        #[cfg(target_arch = "wasm32")]
        let tracker = SubmissionTracker {
            next_id: std::sync::atomic::AtomicU64::new(1),
            completed_id: std::sync::atomic::AtomicU64::new(0),
            tx,
            rx,
        };

        Ok(WgpuDevice {
            inner: Arc::new(WgpuDeviceInner {
                device,
                device_limits,
                device_features: features,
                backend: adapter.get_info().backend,
                queue,
                #[cfg(feature = "wgpu_debug")]
                debug: debug_info,
                command_queue: Mutex::new(QueueBufferInner::new(configuration.meta_buffer_size)),
                meta_buffer,
                cache: Mutex::new(ModelCache::new(
                    configuration.buffer_mapping_size,
                    max_memory_size,
                )),
                bindgroup_layouts,
                extensions: Mutex::new(DeviceExtensions::new()),
                configuration,
                #[cfg(target_arch = "wasm32")]
                submission_tracker: Arc::new(tracker),
            }),
        })
    }

    //allows to load const debug info(for simulating calls)
    pub fn load_simulation_consts(&self, consts: Vec<Vec<(&'static str, f64)>>) {
        let mut queue = self.command_queue.lock().unwrap();
        queue.load_simulation_consts(consts);
    }

    pub fn simulate_command(
        &self,
        command: &DebugPipelineRecording,
        dest_buffer: &WgpuStorage,
        input1_buffer: &WgpuStorage,
        input2_buffer: &WgpuStorage,
        input3_buffer: &WgpuStorage,
    ) {
        let mut command_queue = self.get_queue();
        command_queue
            .get_meta_mut()
            .append(&mut command.meta.clone());

        let new_input = match command.bindgroup.get_input() {
            super::cache::BindgroupInputBase::Bindgroup0(alignment) => {
                super::cache::BindgroupInputBase::Bindgroup0(*alignment)
            }
            super::cache::BindgroupInputBase::Bindgroup1(_, alignment) => {
                super::cache::BindgroupInputBase::Bindgroup1(input1_buffer.buffer(), *alignment)
            }
            super::cache::BindgroupInputBase::Bindgroup2(_, _, alignment) => {
                super::cache::BindgroupInputBase::Bindgroup2(
                    input1_buffer.buffer(),
                    input2_buffer.buffer(),
                    *alignment,
                )
            }
            super::cache::BindgroupInputBase::Bindgroup3(_, _, _, alignment) => {
                super::cache::BindgroupInputBase::Bindgroup3(
                    input1_buffer.buffer(),
                    input2_buffer.buffer(),
                    input3_buffer.buffer(),
                    *alignment,
                )
            }
        };

        let q = MlQueue::Dispatch(MlQueueDispatch {
            x: command.x,
            y: command.y,
            z: command.z,
            pipeline: command.pipeline.clone(),
            pipeline_cached: None,
            bindgroup: BindGroupReference::new(dest_buffer.buffer(), new_input),
            bindgroup_cached: None,
            meta: command_queue.current_meta,
            workload_size: 0_usize,
            #[cfg(feature = "wgpu_debug")]
            debug: None,
        });
        command_queue.command_queue.push(q);
    }

    pub fn is_dtype_available(&self, dtype: DType) -> bool {
        match dtype {
            DType::U32 => true,
            DType::F32 => true,
            DType::U8 => false,
            DType::I64 => self.device_features.contains(wgpu::Features::SHADER_INT64),
            DType::F64 => self.device_features.contains(wgpu::Features::SHADER_F64),
            DType::F16 => self.device_features.contains(wgpu::Features::SHADER_F16),
        }
    }

    #[instrument(skip(self, size))]
    pub fn alloc_uninit_size<T: ToU64>(&self, dtype: crate::DType, size: T) -> WgpuStorage {
        let size = size.to_u64() * dtype.size_in_bytes() as u64;
        let buffer;
        {
            let mut cache = self.cache.lock().unwrap();
            buffer = cache.create_buffer_reference(size, true);
        }
        WgpuStorage::new(buffer, self.clone(), dtype, size)
    }

    #[instrument(skip(self, data))]
    pub fn alloc_from_slice<T: bytemuck::Pod>(
        &self,
        dtype: crate::DType,
        data: &[T],
    ) -> crate::Result<WgpuStorage> {
        let data: &[u8] = bytemuck::cast_slice(data);
        self.alloc_from_bytes(dtype, data)
    }

    #[instrument(skip(self, data))]
    pub fn alloc_from_bytes(&self, dtype: crate::DType, data: &[u8]) -> crate::Result<WgpuStorage> {
        let size = data.len();
        let buffer;
        {
            if self.configuration.flush_gpu_before_buffer_init {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    if let Some(index) = wgpu_functions::flush_gpu_command(self, None)? {
                        wgpu_functions::wait_for_submission(self, index)?;
                    }
                }
            }
            let mut cache = self.cache.lock().unwrap();
            buffer = cache.create_buffer_reference_init(self, data, true);
        }
        Ok(WgpuStorage::new(buffer, self.clone(), dtype, size as u64))
    }

    /**************** Virtual Bindgroups: ****************/
    pub fn create_bind_group_input0(
        &self,
        buffer_dest: BufferReferenceId,
        alignment: BindgroupAlignment,
    ) -> BindGroupReference {
        let alignment = BindgroupAlignmentLayout::Bindgroup0(alignment);
        alignment.validate();
        BindGroupReference::new(buffer_dest, BindgroupInputBase::Bindgroup0(alignment))
    }

    pub fn create_bind_group_input1(
        &self,
        buffer_dest: BufferReferenceId,
        buffer_input1: BufferReferenceId,
        alignment: BindgroupAlignment,
    ) -> BindGroupReference {
        self.create_bind_group_input1_with_alignment(
            buffer_dest,
            buffer_input1,
            BindgroupAlignmentLayout::Bindgroup1(alignment, alignment),
        )
    }

    pub fn create_bind_group_input1_with_alignment(
        &self,
        buffer_dest: BufferReferenceId,
        buffer_input1: BufferReferenceId,
        alignment: BindgroupAlignmentLayout,
    ) -> BindGroupReference {
        alignment.validate();
        BindGroupReference::new(
            buffer_dest,
            BindgroupInputBase::Bindgroup1(buffer_input1, alignment),
        )
    }

    pub fn create_bind_group_input2(
        &self,
        buffer_dest: BufferReferenceId,
        buffer_input1: BufferReferenceId,
        buffer_input2: BufferReferenceId,
        alignment: BindgroupAlignment,
    ) -> BindGroupReference {
        self.create_bind_group_input2_with_alignment(
            buffer_dest,
            buffer_input1,
            buffer_input2,
            BindgroupAlignmentLayout::Bindgroup2(alignment, alignment, alignment),
        )
    }

    pub fn create_bind_group_input2_with_alignment(
        &self,
        buffer_dest: BufferReferenceId,
        buffer_input1: BufferReferenceId,
        buffer_input2: BufferReferenceId,
        alignment: BindgroupAlignmentLayout,
    ) -> BindGroupReference {
        alignment.validate();
        BindGroupReference::new(
            buffer_dest,
            BindgroupInputBase::Bindgroup2(buffer_input1, buffer_input2, alignment),
        )
    }

    pub fn create_bind_group_input3(
        &self,
        buffer_dest: BufferReferenceId,
        buffer_input1: BufferReferenceId,
        buffer_input2: BufferReferenceId,
        buffer_input3: BufferReferenceId,
        alignment: BindgroupAlignment,
    ) -> BindGroupReference {
        self.create_bind_group_input3_with_alignment(
            buffer_dest,
            buffer_input1,
            buffer_input2,
            buffer_input3,
            BindgroupAlignmentLayout::Bindgroup3(alignment, alignment, alignment, alignment),
        )
    }

    pub fn create_bind_group_input3_with_alignment(
        &self,
        buffer_dest: BufferReferenceId,
        buffer_input1: BufferReferenceId,
        buffer_input2: BufferReferenceId,
        buffer_input3: BufferReferenceId,
        alignment: BindgroupAlignmentLayout,
    ) -> BindGroupReference {
        alignment.validate();
        BindGroupReference::new(
            buffer_dest,
            BindgroupInputBase::Bindgroup3(buffer_input1, buffer_input2, buffer_input3, alignment),
        )
    }
}

impl WgpuDevice {
    //Full Recording
    #[cfg(feature = "wgpu_debug")]
    pub fn start_recording_commands(&self) {
        let mut model_cache = self.cache.lock().unwrap();
        model_cache.full_recording.should_record = true;
        println!("start_recording_commands");
    }

    #[cfg(feature = "wgpu_debug")]
    pub fn stop_recording_commands(&self, output_path: &str) -> crate::Result<()> {
        println!("stop_recording_commands");
        let mut model_cache = self.cache.lock().unwrap();
        model_cache.full_recording.should_record = false;
        drop(model_cache);
        debug_info::create_dispatch_zip(self, output_path)
    }

    //Durations and call Count by Pipeline
    #[cfg(feature = "wgpu_debug")]
    pub async fn get_debug_info_full(&self) -> crate::Result<Measurements> {
        use super::wgpu_functions::synchronize_async;
        synchronize_async(self).await?;
        let data =
            wgpu_functions::read_from_buffer_async::<u64>(self, &self.debug.query_set_buffer)
                .await?;

        let period = self.queue.get_timestamp_period();
        let mut result = Measurements::new(period);
        let mut last_end_time = 0u64;
        let mut shader_pipeline2 = self.debug.shader_pipeline.lock().unwrap();
        let shader_pipeline = shader_pipeline2.clone();
        let mut indexes: Vec<_> = shader_pipeline.into_iter().collect();
        indexes.sort_by_key(|f| f.0);
        for (i, p) in indexes.into_iter().enumerate() {
            let start_time = data[(p.0 / 8) as usize];
            let end_time = data[(p.0 / 8) as usize + 1];
            if end_time < start_time {
                panic!("Start Time was after End Time! startTime: {start_time}, endTime:{end_time}, i:{i}");
            }
            if end_time == 0 {
                panic!("End time was 0")
            }
            if start_time == 0 {
                panic!("End time was 0")
            }
            if end_time == start_time {
                panic!("End time == start_time == {}", end_time)
            }
            if start_time < last_end_time {
                panic!("Start Time was before last End Time! startTime: {start_time}, last endTime:{last_end_time}, i:{i}");
            }
            last_end_time = end_time;
            result.data.push(MInfo::new(
                p.1.pipeline.to_owned(),
                start_time,
                end_time,
                p.1.workload_size,
                p.1.x,
                p.1.y,
                p.1.z,
            ));
        }
        self.debug
            .counter
            .store(0u32, std::sync::atomic::Ordering::Relaxed);
        shader_pipeline2.clear();
        Ok(result)
    }

    #[cfg(feature = "wgpu_debug")]
    pub async fn get_debug_info(
        &self,
    ) -> crate::Result<std::collections::HashMap<String, Vec<WgpuDebugInfo>>> {
        let info = self.get_debug_info_full().await?;
        let mut map: std::collections::HashMap<String, Vec<WgpuDebugInfo>> =
            std::collections::HashMap::new();

        for item in info.data.iter() {
            map.entry(item.label.clone())
                .or_default()
                .push(WgpuDebugInfo {
                    duration: item.end_time - item.start_time,
                    output_size: item.output_size,
                    x: item.x,
                    y: item.y,
                    z: item.z,
                });
        }
        Ok(map)
    }

    #[cfg(feature = "wgpu_debug")]
    pub fn get_pipeline_info(&self) -> crate::Result<Vec<ShaderInfo>> {
        use super::debug_info;
        let mut cache = self.cache.lock().unwrap();
        let shaders = &mut cache.shader;

        let queue = self.command_queue.lock().unwrap();

        Ok(shaders
            .shaders
            .iter()
            .map(|(k, v)| {
                let pipelines = &v.pipelines;
                let s = debug_info::ShaderInfo {
                    name: shaders.loader_cache.get_shader_name(*k),
                    pipelines: pipelines
                        .keys()
                        .map(|pk| debug_info::PipelineInfo {
                            name: shaders.loader_cache.get_entry_point(pk.0).to_owned(),
                            consts: queue.id_to_const_array[pk.1]
                                .iter()
                                .map(|(s, x)| (s.to_string(), *x))
                                .collect(),
                        })
                        .collect(),
                };
                s
            })
            .collect())
    }

    #[cfg(feature = "wgpu_debug")]
    pub fn log_debuginfo_to_file(
        &self,
        folder: &str,
        name: &str,
        version: &str,
    ) -> crate::Result<()> {
        let info = pollster::block_on(self.get_debug_info()).unwrap();
        let map2 = debug_info::calulate_measurment(&info);
        debug_info::save_list(
            &map2,
            &format!("{folder}wgpu_{name}_test_{version}_measurements.json"),
        )
        .unwrap();

        let info: Vec<debug_info::ShaderInfo> = self.get_pipeline_info().unwrap();
        debug_info::save_list(
            &info,
            &format!("{folder}wgpu_{name}_test_{version}_shaders.json"),
        )
        .unwrap();

        let (pipelines, consts) = self.get_used_pipelines();
        std::fs::write(
            format!("{folder}wgpu_{name}_test_{version}_used_pipelines.json"),
            pipelines,
        )?;
        std::fs::write(
            format!("{folder}wgpu_{name}_test_{version}_used_consts.json"),
            consts,
        )?;

        let cache = self.cache.lock().unwrap();
        debug_info::save_list(
            &cache.debug_buffer_info,
            &format!("{folder}wgpu_{name}_test_{version}_buffers.json"),
        )
        .unwrap();
        Ok(())
    }

    #[cfg(feature = "wgpu_debug")]
    fn get_used_pipelines(&self) -> (String, String) {
        let cache = self.cache.lock().unwrap();
        let queue = self.command_queue.lock().unwrap();
        let consts = &queue.id_to_const_array;

        let debug: Vec<_> = cache.debug.values().collect();
        (
            serde_json::to_string(&debug).unwrap(),
            serde_json::to_string(consts).unwrap(),
        )
    }

    pub fn get_internal_counters(&self) -> InternalCounter {
        let cache = self.cache.lock().unwrap();
        InternalCounter {
            buffer_counter: cache.buffers.buffer_counter(),
            buffer_reuse_counter : cache.buffers.buffer_reuse_counter(),

            binary_inplace_counter: cache.binary_inplace_counter,
            unary_inplace_counter: cache.unary_inplace_counter,
            copy_inplace_counter: cache.copy_inplace_counter,

            bindgroup_counter: cache.bindgroups.bindgroup_counter(),
            cached_bindgroup_use_counter: cache.bindgroups.cached_bindgroup_use_counter(),
        }
    }
}

impl WgpuDevice {
    #[cfg(target_arch = "wasm32")]
    pub fn synchronize(&self) -> crate::Result<()> {
        panic!("Synchronize is not possible on wasm. use synchronize_async");
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn synchronize(&self) -> crate::Result<()> {
        wgpu_functions::synchronize(self)
    }

    pub async fn synchronize_async(&self) -> crate::Result<()> {
        wgpu_functions::synchronize_async(self).await
    }
}

impl WgpuDevice {
    pub fn with_extension<T: Any, R>(&self, f: impl FnOnce(&T) -> R) -> Option<R> {
        let ext = self.extensions.lock().unwrap();
        ext.get::<T>().map(f)
    }

    pub fn with_extension_mut<T: Any, R>(&self, f: impl FnOnce(&mut T) -> R) -> Option<R> {
        let mut ext = self.extensions.lock().unwrap();
        ext.get_mut::<T>().map(f)
    }

    pub fn set_extension<T: Any + Send + Sync>(&self, value: T) {
        let mut ext = self.extensions.lock().unwrap();
        ext.insert(value);
    }
}

impl WgpuDevice {
    pub fn create_buffer_reference<T: ToU64>(
        &self,
        size: T,
        referenced_by_candle_storage: bool,
    ) -> BufferReferenceId {
        self.cache
            .lock()
            .unwrap()
            .create_buffer_reference(size, referenced_by_candle_storage)
    }

    pub fn with_shader_loader_cache<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut crate::shader_loader::ShaderLoaderCache) -> R,
    {
        let mut cache = self.cache.lock().unwrap();
        f(&mut cache.shader.loader_cache)
    }

    pub fn add_wgpu_shader_loader<T: shader_loader::ShaderLoader + 'static + Send + Sync>(
        &self,
        index: shader_loader::LoaderIndex,
        shader_loader: impl Fn() -> T,
    ) {
        let mut cache = self.cache.lock().unwrap();
        cache.shader.add_wgpu_shader_loader(index, shader_loader);
    }
}
