use std::collections::HashMap;
use std::fmt;

use std::sync::{Arc, Mutex};

use std::hash::Hash;

use candle_wgpu_kernels::{Constants, EntryPoint};
use rand::SeedableRng;
use tracing::instrument;
use wgpu::{Backends, InstanceDescriptor, InstanceFlags};

use crate::backend::BackendStorage;
use crate::{notImplemented, wrongType, DType, Layout};

#[cfg(feature = "wgpu_debug")]
use super::debug_info::{DebugInfo, MInfo, Measurements, ShaderInfo};

use super::cache::{BindgroupLayouts, CachedBindgroupId, CachedBufferId, ModelCache};
use super::storage::{create_wgpu_storage, create_wgpu_storage_init};
use super::util::{ObjectToIdMapper, ToF64, ToU32};
use super::wgpu_functions::{self, unary::UnaryOperation, MetaArray};
use super::wgpu_functions::{ConstArray, KernelParameterMeta};
use super::WgpuStorage;

#[derive(Debug)]
pub(crate) enum MlQueue {
    Dispatch(MlQueueDispatch),
}

#[derive(Debug, Copy, Clone)]
pub struct OrderedFloat(pub f64);

impl PartialEq for OrderedFloat {
    fn eq(&self, other: &OrderedFloat) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for OrderedFloat {}

impl Hash for OrderedFloat {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "wgpu_debug", derive(serde::Serialize, serde::Deserialize))]
pub struct OpIsInplaceable {
    pub input1_inplaceable: bool,
    pub input2_inplaceable: bool,
}

impl OpIsInplaceable {
    pub fn new() -> Self {
        Self {
            input1_inplaceable: false,
            input2_inplaceable: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(
    any(feature = "wgpu_debug_serialize", feature = "wgpu_debug"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct PipelineType(
    pub candle_wgpu_kernels::PipelineIndex,
    pub(crate) usize,
    #[cfg_attr(
        any(feature = "wgpu_debug_serialize", feature = "wgpu_debug"),
        serde(skip)
    )]
    pub(crate) OpIsInplaceable,
);

pub(crate) type BindGroupReference = crate::wgpu_backend::cache::BindgroupReferenceFull;

#[derive(Debug)]
pub(crate) struct MlQueueDispatch {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) z: u32,
    pub(crate) pipeline: PipelineType,
    pub(crate) pipeline_cached: Option<Arc<wgpu::ComputePipeline>>,
    pub(crate) bindgroup: BindGroupReference,
    pub(crate) bindgroup_cached: Option<CachedBindgroupId>,
    pub(crate) meta: u32,
    pub(crate) workload_size: usize, //the total size needed to calculate. Needed so we do not queue to many operations at once.
    #[cfg(feature = "wgpu_debug")]
    pub(crate) debug: Option<String>,
}

//a struct, where all operations are chunked
#[derive(Debug)]
pub struct QueueBuffer {
    pub(crate) command_queue: Vec<MlQueue>,
    meta_array: MetaArray,
    const_array: ConstArray,
    const_id_map: ObjectToIdMapper<ConstArray>,
    global_command_index: u32,
    pub(crate) id_to_const_array: Vec<HashMap<String, f64>>,
    pub(crate) current_meta: u32,
    pub(crate) last_buffer: Option<CachedBufferId>, //will be used to wait for the last command queue
}

impl QueueBuffer {
    pub fn new(size: u32) -> Self {
        Self {
            command_queue: vec![],
            meta_array: MetaArray::new(size),
            current_meta: 0,
            const_array: ConstArray::new(),
            const_id_map: ObjectToIdMapper::new(),
            id_to_const_array: Vec::new(),
            last_buffer: None,
            global_command_index: 1,
        }
    }

    pub fn init(&mut self) {
        self.const_array.0.clear();
    }

    pub fn clear(&mut self) {
        self.command_queue.clear();
        self.meta_array.0.clear();
        self.init();
        self.current_meta = 0;
    }

    pub fn get_meta(&self) -> &Vec<u32> {
        &self.meta_array.0
    }

    pub fn get_meta_mut(&mut self) -> &mut Vec<u32> {
        &mut self.meta_array.0
    }

    pub fn add_layout(
        &mut self,
        layout: &Layout,
        is_contiguous: bool,
        constant_dims: Constants,
        constant_is_startofsset_zero: Constants,
        constant_is_contiguous: Constants,
    ) {
        let shape = layout.shape().dims();
        let stride = layout.stride();

        self.add_const(constant_dims, shape.len());
        if layout.start_offset() != 0 {
            self.add_const(constant_is_startofsset_zero, false);
            self.add(layout.start_offset());
        }

        if is_contiguous {
            self.add(layout.shape().elem_count());
        } else {
            self.add_const(constant_is_contiguous, false);

            self.get_meta_mut().extend(shape.iter().map(|&x| x as u32));
            self.get_meta_mut().extend(stride.iter().map(|&x| x as u32));
        }
    }

    pub fn add_layout1(&mut self, layout: &Layout) {
        self.add_layout(
            layout,
            layout.is_contiguous(),
            Constants::ConstDims1,
            Constants::ConstIsStartoffsetZero1,
            Constants::ConstIsContiguous1,
        );
    }

    pub fn add_layout2(&mut self, layout: &Layout) {
        self.add_layout(
            layout,
            layout.is_contiguous(),
            Constants::ConstDims2,
            Constants::ConstIsStartoffsetZero2,
            Constants::ConstIsContiguous2,
        );
    }

    pub fn add_layout3(&mut self, layout: &Layout) {
        self.add_layout(
            layout,
            layout.is_contiguous(),
            Constants::ConstDims3,
            Constants::ConstIsStartoffsetZero3,
            Constants::ConstIsContiguous3,
        );
    }

    //forces to write the shapes and strides
    pub fn add_layout1_non_contiguous(&mut self, layout: &Layout) {
        self.add_layout(
            layout,
            false,
            Constants::ConstDims1,
            Constants::ConstIsStartoffsetZero1,
            Constants::ConstIsContiguous1,
        );
    }

    pub fn add_layout2_non_contiguous(&mut self, layout: &Layout) {
        self.add_layout(
            layout,
            false,
            Constants::ConstDims2,
            Constants::ConstIsStartoffsetZero2,
            Constants::ConstIsContiguous2,
        );
    }

    pub fn add_layout3_non_contiguous(&mut self, layout: &Layout) {
        self.add_layout(
            layout,
            false,
            Constants::ConstDims3,
            Constants::ConstIsStartoffsetZero3,
            Constants::ConstIsContiguous3,
        );
    }

    pub fn get_pipeline(&mut self, pipeline: impl Into<candle_wgpu_kernels::PipelineIndex>) -> PipelineType {
        let (index, is_new) = self.const_id_map.get_or_insert(&self.const_array);
        if is_new {
            let hmap = HashMap::from_iter(
                self.const_array
                    .0
                    .iter()
                    .map(|(k, v)| (k.get_entry_point().to_owned(), v.to_f64())),
            );
            self.id_to_const_array.push(hmap)
        }
        self.init();
        PipelineType(pipeline.into(), index, OpIsInplaceable::new())
    }

    pub fn get_pipeline_const<T: ToU32>(
        &mut self,
        pipeline: impl Into<candle_wgpu_kernels::PipelineIndex>,
        const_vec: Vec<T>,
    ) -> PipelineType {
        for (index, v) in const_vec.into_iter().enumerate() {
            self.const_array
                .0
                .push((candle_wgpu_kernels::Constants::get_const(index), v.to_u32()));
        }

        let (index, is_new) = self.const_id_map.get_or_insert(&self.const_array);
        if is_new {
            let hmap = HashMap::from_iter(
                self.const_array
                    .0
                    .iter()
                    .map(|(k, v)| (k.get_entry_point().to_owned(), v.to_f64())),
            );
            self.id_to_const_array.push(hmap)
        }
        self.init();
        PipelineType(pipeline.into(), index, OpIsInplaceable::new())
    }
    pub fn get_pipeline_const_inplace<T: ToU32>(
        &mut self,
        pipeline: impl Into<candle_wgpu_kernels::PipelineIndex>,
        const_vec: Vec<T>,
        inplaceable: OpIsInplaceable,
    ) -> PipelineType {
        for (index, v) in const_vec.into_iter().enumerate() {
            self.const_array
                .0
                .push((candle_wgpu_kernels::Constants::get_const(index), v.to_u32()));
        }

        let (index, is_new) = self.const_id_map.get_or_insert(&self.const_array);
        if is_new {
            let hmap = HashMap::from_iter(
                self.const_array
                    .0
                    .iter()
                    .map(|(k, v)| (k.get_entry_point().to_owned(), v.to_f64())),
            );
            self.id_to_const_array.push(hmap)
        }
        self.init();
        PipelineType(pipeline.into(), index, inplaceable)
    }

    pub fn add<T: KernelParameterMeta>(&mut self, value: T) {
        self.meta_array.add(value);
    }

    pub fn add_const<T: ToU32>(&mut self, key: candle_wgpu_kernels::Constants, value: T) {
        self.const_array.insert(key, value);
    }

    pub fn global_command_index(&self) -> u32 {
        self.global_command_index
    }

    pub fn set_global_command_index(&mut self, global_command_index: u32) {
        self.global_command_index = global_command_index;
    }
}

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
#[cfg_attr(
    any(feature = "wgpu_debug_serialize", feature = "wgpu_debug"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct DebugPipelineRecording {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) z: u32,
    pub pipeline: super::device::PipelineType,
    pub(crate) meta: Vec<u32>,
    pub(crate) bindgroup: BindGroupReference,
    pub count: u32,
}

#[derive(Clone)]
pub enum MatmulAlgorithm {
    MatmulX,
    Matmul7,
    Matmul1,
    Matmul1_4,
    Matmul16_16,
    Matmul32_64,
    Matmul32_64B,
    Matmul32_32,
    Matmul64_64,
    Matmul64_64_8_8,
    Matmul64_64_4_8,
    Matmul1_64,
    Matmul1_64B,
    Matmul24_24,
    Matmul24_48,
    Matmul24_24B,
    Matmul24_48B,
}

impl fmt::Debug for MatmulAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MatmulX => write!(f, "MatmulX"),
            Self::Matmul7 => write!(f, "Matmul7"),
            Self::Matmul1 => write!(f, "Matmul1"),
            Self::Matmul1_4 => write!(f, "Matmul1_4"),
            Self::Matmul16_16 => write!(f, "Matmul5_16_16"),
            Self::Matmul32_64 => write!(f, "Matmul5_32_64"),
            Self::Matmul32_64B => write!(f, "Matmul5_32_64B"),
            Self::Matmul32_32 => write!(
                f,
                "Matmul5_32_32"
            ),
            Self::Matmul64_64 => write!(
                f,
                "Matuml5_64_64"
            ),
            Self::Matmul64_64_8_8 => write!(
                f,
                "Matmul5_64_64_8_8"
            ),
            Self::Matmul64_64_4_8 => write!(
                f,
                "Matmul5_64_64_4_8"
            ),
            Self::Matmul1_64 => write!(
                f,
                "Matmul5_1_64"
            ),
            Self::Matmul1_64B => write!(
                f,
                "Matmul5_1_64B"
            ),
            Self::Matmul24_24 => write!(
                f,
                "Matmul5_24_24"
            ),
            Self::Matmul24_48 => write!(
                f,
                "Matmul5_24_48"
            ),
            Self::Matmul24_24B => write!(
                f,
                "Matmul5_24_24B"
            ),
            Self::Matmul24_48B => write!(
                f,
                "Matmul5_24_48B"
            ),
        }
    }
}

#[derive(Debug)]
pub struct WgpuDeviceInner {
    pub device: wgpu::Device,
    pub backend: wgpu::Backend,
    pub device_limits: wgpu::Limits, //we cache the limits here, because device.limit() was relatively slow on the browser
    pub device_features: wgpu::Features,

    pub queue: wgpu::Queue,
    pub(crate) rand_state: Mutex<rand::rngs::StdRng>,

    pub(crate) command_queue: Mutex<QueueBuffer>,
    pub(crate) meta_buffer: wgpu::Buffer, //buffer for storing meta information

    pub(crate) bindgroup_layouts: BindgroupLayouts,

    pub(crate) staging_probe_buffer: wgpu::Buffer, //wait for submission is not supported on wgpu, we use a mapping to a staging buffer as a work around.

    pub(crate) cache: Mutex<ModelCache>, //if cache is set, all commands are not queued to the gpu, but are cached inside ModelCache, so there can be reused later on
    //debug counter
    #[cfg(feature = "wgpu_debug")]
    pub debug: DebugInfo,

    pub configuration: crate::WgpuDeviceConfig,

    pub matmul_alg: Mutex<MatmulAlgorithm>,
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

impl WgpuDevice {

    #[instrument]
    pub(crate) async fn create(
        _: usize,
        configuration: crate::WgpuDeviceConfig,
    ) -> crate::Result<Self> {
        let mut backend = wgpu::Backends::empty();
        if configuration.backend & crate::WgpuBackends::vulkan() {
            backend.insert(Backends::VULKAN);
        }
        if configuration.backend & crate::WgpuBackends::dx12() {
            backend.insert(Backends::DX12);
        }
        if configuration.backend & crate::WgpuBackends::gl() {
            backend.insert(Backends::GL);
        }
        if configuration.backend & crate::WgpuBackends::metal() {
            backend.insert(Backends::METAL);
        }
        if configuration.backend & crate::WgpuBackends::browser_webgpu() {
            backend.insert(Backends::BROWSER_WEBGPU);
        }

        let instance = wgpu::Instance::new(InstanceDescriptor {
            backends: backend,
            flags: InstanceFlags::default(),
            dx12_shader_compiler: wgpu::Dx12Compiler::Fxc,
            gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
        });

        // `request_adapter` instantiates the general connection to the GPU
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .unwrap();

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
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: features,
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|err| crate::Error::Wgpu(err.to_string().into()))?;
        log::info!("Device Requested");

        #[cfg(feature = "wgpu_debug")]
        let debug_info = super::debug_info::DebugInfo::new(&device);

        let meta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: configuration.meta_buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let device_limits = device.limits();
        let bindgroup_layouts = BindgroupLayouts::new(&device);

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let max_memory_size: u64 = device_limits.max_buffer_size.min(device_limits.max_storage_buffer_binding_size as u64);

        Ok(WgpuDevice {
            inner: Arc::new(WgpuDeviceInner {
                device,
                device_limits,
                device_features: features,
                backend: adapter.get_info().backend,
                queue,
                rand_state: Mutex::new(rand::rngs::StdRng::from_entropy()),
                #[cfg(feature = "wgpu_debug")]
                debug: debug_info,
                command_queue: Mutex::new(QueueBuffer::new(configuration.meta_buffer_size)),
                meta_buffer,
                cache: Mutex::new(ModelCache::new(configuration.buffer_mapping_size, max_memory_size)),
                bindgroup_layouts,
                staging_probe_buffer: staging_buffer,
                matmul_alg: Mutex::new(MatmulAlgorithm::MatmulX),
                configuration,
            }),
        })
    }

    pub fn flush_gpu_command(&self) -> crate::Result<()> {
        let mut queue = self.command_queue.lock().unwrap();
        wgpu_functions::flush_gpu_command(self, &mut queue)
    }

    pub fn add_wgpu_shader_loader<T : candle_wgpu_kernels::ShaderLoader + 'static + Send + Sync>(&self, index : candle_wgpu_kernels::LoaderIndex, shader_loader : impl Fn() -> T){
        let mut cache = self.cache.lock().unwrap();
        cache.shader.add_wgpu_shader_loader(index, shader_loader);
    }

    pub fn print_bindgroup_reuseinfo(&self) {
        let cache = self.cache.lock().unwrap();

        log::warn!(
            "Buffer: created: {}, resued : {}",
            cache.buffers.buffer_counter(),
            cache.buffers.buffer_reuse_counter()
        );
        log::warn!(
            "Bindgroup: created: {}, resued : {}",
            cache.bindgroups.bindgroup_counter(),
            cache.bindgroups.cached_bindgroup_use_counter()
        );
        log::warn!(
            "Inplace used: unary: {}, binary {}, copy: {}",
            cache.unary_inplace_counter,
            cache.binary_inplace_counter,
            cache.copy_inplace_counter
        );
    }
    pub fn print_bindgroup_reuseinfo2(&self) {
        let cache = self.cache.lock().unwrap();

        println!(
            "Buffer: created: {}, resued : {}",
            cache.buffers.buffer_counter(),
            cache.buffers.buffer_reuse_counter()
        );
        println!(
            "Bindgroup: created: {}, resued : {}",
            cache.bindgroups.bindgroup_counter(),
            cache.bindgroups.cached_bindgroup_use_counter()
        );
        println!(
            "Inplace used: unary: {}, binary {}, copy: {}",
            cache.unary_inplace_counter,
            cache.binary_inplace_counter,
            cache.copy_inplace_counter
        );
    }

    #[cfg(feature = "wgpu_debug")]
    pub fn log_debuginfo_to_file(&self, folder : &str, name : &str, version : &str) -> crate::Result<()>{
        let info = pollster::block_on(self.get_debug_info()).unwrap();
        let map2 = crate::wgpu::debug_info::calulate_measurment(&info);
        crate::wgpu::debug_info::save_list(&map2,& format!("{folder}wgpu_{name}_test_{version}_b.json")).unwrap();
    
    
        let info: Vec<crate::wgpu::debug_info::ShaderInfo> = self.get_pipeline_info().unwrap();
        crate::wgpu::debug_info::save_list(&info,& format!("{folder}wgpu_{name}_test_{version}_c.json")).unwrap();

        let (pipelines, consts) = self.get_used_pipelines();
        std::fs::write(format!("{folder}wgpu_{name}_test_{version}_d.json"), pipelines)?;   
        std::fs::write(format!("{folder}wgpu_{name}_test_{version}_e.json"), consts)?;
        Ok(())
    }

    

    #[cfg(feature = "wgpu_debug")]
    pub fn get_used_pipelines(&self) -> (String, String) {
        let cache = self.cache.lock().unwrap();
        let queue = self.command_queue.lock().unwrap();
        let consts = &queue.id_to_const_array;

        let debug: Vec<_> = cache.debug.iter().map(|(_, v)| v).collect();

        return (
            serde_json::to_string(&debug).unwrap(),
            serde_json::to_string(consts).unwrap(),
        );
    }

    //allows to load const debug info(for simulating calls)
    pub fn load_debug_info(&self, consts: Vec<HashMap<String, f64>>) {
        let mut queue = self.command_queue.lock().unwrap();
        queue.id_to_const_array = consts;

        queue.const_id_map.next_id = queue.id_to_const_array.len();
    }

    pub fn simulate_command(
        &self,
        command: &DebugPipelineRecording,
        dest_buffer: &WgpuStorage,
        input1_buffer: &WgpuStorage,
        input2_buffer: &WgpuStorage,
        input3_buffer: &WgpuStorage,
    ) {
        let mut command_queue = wgpu_functions::get_meta(self);
        command_queue
            .get_meta_mut()
            .append(&mut command.meta.clone());

        let new_input = match command.bindgroup.get_input() {
            super::cache::BindgroupInputBase::Bindgroup0(alignment) => {
                super::cache::BindgroupInputBase::Bindgroup0(*alignment)
            }
            super::cache::BindgroupInputBase::Bindgroup1(_, alignment) => {
                super::cache::BindgroupInputBase::Bindgroup1(*input1_buffer.buffer(), *alignment)
            }
            super::cache::BindgroupInputBase::Bindgroup2(_, _, alignment) => {
                super::cache::BindgroupInputBase::Bindgroup2(
                    *input1_buffer.buffer(),
                    *input2_buffer.buffer(),
                    *alignment,
                )
            }
            super::cache::BindgroupInputBase::Bindgroup3(_, _, _, alignment) => {
                super::cache::BindgroupInputBase::Bindgroup3(
                    *input1_buffer.buffer(),
                    *input2_buffer.buffer(),
                    *input3_buffer.buffer(),
                    *alignment,
                )
            }
        };

        let q = MlQueue::Dispatch(super::device::MlQueueDispatch {
            x: command.x,
            y: command.y,
            z: command.z,
            pipeline: command.pipeline.clone(),
            pipeline_cached: None,
            bindgroup: BindGroupReference::new(*dest_buffer.buffer(), new_input),
            bindgroup_cached: None,
            meta: command_queue.current_meta,
            workload_size: 0_usize,
            #[cfg(feature = "wgpu_debug")]
            debug: None,
        });
        command_queue.command_queue.push(q);
    }

    #[cfg(feature = "wgpu_debug")]
    pub async fn get_debug_info_full(&self) -> crate::Result<Measurements> {
        use super::wgpu_functions::synchronize_async;
        synchronize_async(self).await?;
        let data = wgpu_functions::read_from_buffer_async::<u64>(
            self,
            &self.debug.query_set_buffer,
        )
        .await;

        let period = self.queue.get_timestamp_period();
        let mut result = Measurements::new(period);
        let mut last_end_time = 0u64;
        let mut i = 0;
        let mut shader_pipeline2 = self.debug.shader_pipeline.lock().unwrap();
        let shader_pipeline = shader_pipeline2.clone();
        let mut indexes: Vec<_> = shader_pipeline.into_iter().collect();
        indexes.sort_by_key(|f| f.0);
        for p in indexes {
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
            i += 1;
            result.data.push(MInfo::new(
                p.1 .0.to_owned(),
                start_time,
                end_time,
                p.1 .1,
                p.1 .2,
                p.1 .3,
                p.1 .4,
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
    ) -> crate::Result<std::collections::HashMap<String, Vec<(u64, u64, u32, u32, u32)>>> {
        let info = self.get_debug_info_full().await?;
        let mut map: std::collections::HashMap<String, Vec<(u64, u64, u32, u32, u32)>> =
            std::collections::HashMap::new();

        for item in info.data.iter() {
            map.entry(item.label.clone())
                .or_insert_with(Vec::new)
                .push((
                    item.end_time - item.start_time,
                    item.output_size,
                    item.x,
                    item.y,
                    item.z,
                ));
        }
        return Ok(map);
    }

    #[cfg(feature = "wgpu_debug")]
    pub fn get_pipeline_info(&self) -> crate::Result<Vec<ShaderInfo>> {
        use super::debug_info;
        let mut cache = self.cache.lock().unwrap();
        let shaders = &mut cache.shader;

        let queue = self.command_queue.lock().unwrap();

        return Ok(shaders.shaders
            .iter()
            .map(|(k, v)| {
                let pipelines = &v.pipelines;
                let s = debug_info::ShaderInfo {
                    name: format!("{:?}", k).to_owned(),
                    pipelines: pipelines
                        .iter()
                        .map(|(pk, _)| {
                            return debug_info::PipelineInfo {
                                name: format!("{:?}", pk.0).to_owned(),
                                consts: queue.id_to_const_array[pk.1].clone(),
                            };
                        })
                        .collect(),
                };
                return s;
            })
            .collect());
    }

    pub(crate) async fn synchronize_async(&self) -> crate::Result<()> {
        wgpu_functions::synchronize_async(self).await
    }

    pub(crate) fn is_dtype_available(&self, dtype: DType) -> bool {
        match dtype {
            DType::U32 => true,
            DType::F32 => true,
            DType::U8 => false,
            DType::I64 => {
                self.device_features.contains(wgpu::Features::SHADER_INT64)
            }
            DType::F64 => {
                self.device_features.contains(wgpu::Features::SHADER_F64)
            }

            DType::BF16 => false,
            DType::F16 => false,
        }
    }
}

impl crate::backend::BackendDevice for WgpuDevice {
    type Storage = WgpuStorage;

    fn new(_: usize) -> crate::Result<Self> {
        Err(crate::Error::Wgpu(
            "A WgpuDevice must be created using the asynchronous create method"
                .to_owned()
                .into(),
        ))
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Wgpu { gpu_id: 0 }
    }

    fn same_device(&self, other: &Self) -> bool {
        self.device.global_id() == other.device.global_id()
    }

    fn zeros_impl(
        &self,
        shape: &crate::Shape,
        dtype: crate::DType,
    ) -> crate::Result<Self::Storage> {
        let buffer = create_wgpu_storage(self, dtype, shape.elem_count() * dtype.size_in_bytes());
        if shape.elem_count() > 0 {
            wgpu_functions::queue_unary_inplace_op(
                self,
                *buffer.buffer(),
                UnaryOperation::SetZero,
                0.0,
                0.0,
                dtype,
                &Layout::contiguous(shape),
            )?;
        }

        Ok(buffer)
    }

    fn ones_impl(&self, shape: &crate::Shape, dtype: crate::DType) -> crate::Result<Self::Storage> {
        let buffer = create_wgpu_storage(self, dtype, shape.elem_count() * dtype.size_in_bytes());

        if shape.elem_count() > 0 {
            wgpu_functions::queue_unary_inplace_op(
                self,
                *buffer.buffer(),
                UnaryOperation::SetOne,
                0.0,
                0.0,
                dtype,
                &Layout::contiguous(shape),
            )?;
        }
        Ok(buffer)
    }

    unsafe fn alloc_uninit(
        &self,
        shape: &crate::Shape,
        dtype: crate::DType,
    ) -> crate::Result<Self::Storage> {
        if self.is_dtype_available(dtype) {
            Ok(create_wgpu_storage(
                self,
                dtype,
                shape.elem_count() * dtype.size_in_bytes(),
            ))
        } else {
            wrongType!(alloc_uninit, dtype);
        }
    }

    fn storage_from_slice<T: crate::WithDType>(&self, data: &[T]) -> crate::Result<Self::Storage> {
        let buffer;
        if T::DTYPE == crate::DType::F32 {
            let data =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len()) };
            buffer = create_wgpu_storage_init(self, T::DTYPE, data)?;
        } else if T::DTYPE == crate::DType::U32 {
            let data =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u32, data.len()) };
            buffer = create_wgpu_storage_init(self, T::DTYPE, data)?;
        } else {
            // Panic if T is not f32 or u32
            wrongType!(storage_from_slice, T::DTYPE);
        }
        Ok(buffer)
    }

    fn storage_from_cpu_storage(
        &self,
        storage: &crate::CpuStorage,
    ) -> crate::Result<Self::Storage> {
        match storage {
            crate::CpuStorage::F32(data) => {
                create_wgpu_storage_init(self, crate::DType::F32, data)
            }
            crate::CpuStorage::U32(data) => {
                create_wgpu_storage_init(self, crate::DType::U32, data)
            }
            _ => wrongType!(storage_from_cpu_storage, storage.dtype()),
        }
    }

    fn storage_from_cpu_storage_owned(
        &self,
        storage: crate::CpuStorage,
    ) -> crate::Result<Self::Storage> {
        match storage {
            crate::CpuStorage::F32(data) => {
                create_wgpu_storage_init(self, crate::DType::F32, &data)
            }
            crate::CpuStorage::U32(data) => {
                create_wgpu_storage_init(self, crate::DType::U32, &data)
            }
            crate::CpuStorage::I64(data) => {
                create_wgpu_storage_init(self, crate::DType::I64, &data)
            }
            crate::CpuStorage::F64(data) => {
                create_wgpu_storage_init(self, crate::DType::F64, &data)
            }
            _ => wrongType!(storage_from_cpu_storage_owned, storage.dtype()),
        }
    }

    fn rand_uniform(
        &self,
        shape: &crate::Shape,
        dtype: crate::DType,
        lo: f64,
        up: f64,
    ) -> crate::Result<Self::Storage> {
        let buffer = create_wgpu_storage(self, dtype, shape.elem_count() * dtype.size_in_bytes());
        wgpu_functions::queue_unary_inplace_op(
            self,
            *buffer.buffer(),
            UnaryOperation::RandUniform,
            lo as f32,
            up as f32,
            dtype,
            &Layout::contiguous(shape),
        )?;
        Ok(buffer)
    }

    fn rand_normal(
        &self,
        shape: &crate::Shape,
        dtype: crate::DType,
        mean: f64,
        std: f64,
    ) -> crate::Result<Self::Storage> {
        let buffer = create_wgpu_storage(self, dtype, shape.elem_count() * dtype.size_in_bytes());
        wgpu_functions::queue_unary_inplace_op(
            self,
            *buffer.buffer(),
            UnaryOperation::RandNormal,
            mean as f32,
            std as f32,
            dtype,
            &Layout::contiguous(shape),
        )?;
        Ok(buffer)
    }

    fn set_seed(&self, _: u64) -> crate::Result<()> {
        notImplemented!(set_seed)
    }

    #[cfg(target_arch = "wasm32")]
    fn synchronize(&self) -> crate::Result<()> {
        panic!("Synchronize is not possible on wasm. (on_submitted_work_done is currently not implemented in wgpu). In addition synchronize can only be handled async");
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn synchronize(&self) -> crate::Result<()> {
        wgpu_functions::synchronize(self)
    }
}
