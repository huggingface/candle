use std::fmt;
use std::sync::{Arc, Mutex};

use std::hash::Hash;

use rand::SeedableRng;
use tracing::instrument;
use wgpu::{Backends, InstanceDescriptor, InstanceFlags};

use crate::backend::{BackendDevice, BackendStorage};
use crate::{notImplemented, wrongType, DType, Layout};

#[cfg(feature = "wgpu_debug")]
use super::debug_info::{DebugInfo, MInfo, Measurements, ShaderInfo};

use super::cache::{BindgroupAlignment, BindgroupAlignmentLayout, BindgroupInputBase, BindgroupLayouts, BufferReferenceId, ModelCache};
use super::queue_buffer::{BindGroupReference, MlQueue, MlQueueDispatch, PipelineReference};
use super::util::ToU64;
use super::wgpu_functions::{self, unary::UnaryOperation};
use super::WgpuStorage;
use crate::wgpu_backend::queue_buffer::QueueBufferInner;

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
#[cfg_attr(
    any(feature = "wgpu_debug_serialize", feature = "wgpu_debug"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct DebugPipelineRecording {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub pipeline: PipelineReference,
    pub meta: Vec<u32>,
    pub bindgroup: BindGroupReference,
    pub count: u32,
}

#[derive(Clone)]
pub enum MatmulAlgorithm {
    MatmulX, //select best fitting kernel automatically
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
    Matmul1_64_32B,
    Matmul1_32_32B,
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
            Self::Matmul16_16 => write!(f, "Matmul_16_16"),
            Self::Matmul32_64 => write!(f, "Matmul_32_64"),
            Self::Matmul32_64B => write!(f, "Matmul_32_64B"),
            Self::Matmul32_32 => write!(f, "Matmul_32_32"),
            Self::Matmul64_64 => write!(f, "Matuml_64_64"),
            Self::Matmul64_64_8_8 => write!(f, "Matmul_64_64_8_8"),
            Self::Matmul64_64_4_8 => write!(f, "Matmul_64_64_4_8"),
            Self::Matmul1_64 => write!(f, "Matmul_1_64"),
            Self::Matmul1_64B => write!(f, "Matmul_1_64B"),
            Self::Matmul1_64_32B => write!(f, "Matmul_1_64_32B"),
            Self::Matmul1_32_32B => write!(f, "Matmul_1_32_32B"),
            Self::Matmul24_24 => write!(f, "Matmul_24_24"),
            Self::Matmul24_48 => write!(f, "Matmul_24_48"),
            Self::Matmul24_24B => write!(f, "Matmul_24_24B"),
            Self::Matmul24_48B => write!(f, "Matmul_24_48B"),
        }
    }
}

static DEVICE_COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

#[derive(Debug)]
pub struct WgpuDeviceInner {
    pub device_id: u32,
    pub device: wgpu::Device,
    pub backend: wgpu::Backend,
    pub device_limits: wgpu::Limits, //we cache the limits here, because device.limit() was relatively slow on the browser
    pub device_features: wgpu::Features,

    pub queue: wgpu::Queue,
    pub(crate) rand_state: Mutex<rand::rngs::StdRng>,

    pub(crate) command_queue: Mutex<QueueBufferInner>,
    pub(crate) meta_buffer: wgpu::Buffer, //buffer for storing meta information

    pub(crate) bindgroup_layouts: BindgroupLayouts,

    pub(crate) staging_probe_buffer: wgpu::Buffer, //wait for submission is not supported on wgpu, we use a mapping to a staging buffer as a work around.

    pub(crate) cache: Mutex<ModelCache>, //if cache is set, all commands are not queued to the gpu, but are cached inside ModelCache, so there can be reused later on
    //debug counter
    #[cfg(feature = "wgpu_debug")]
    pub debug: DebugInfo,

    pub configuration: crate::WgpuDeviceConfig,

    pub matmul_alg: Mutex<MatmulAlgorithm>, //MatMul algorithm override, used for testing and benchmarking.
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

        let instance = wgpu::Instance::new(&InstanceDescriptor {
            backends: backend,
            flags: InstanceFlags::default(),
            backend_options: wgpu::BackendOptions{
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
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: features,
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                    ..Default::default()
                }
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

        let max_memory_size: u64 = device_limits
            .max_buffer_size
            .min(device_limits.max_storage_buffer_binding_size as u64);

        Ok(WgpuDevice {
            inner: Arc::new(WgpuDeviceInner {
                device_id: DEVICE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
                device,
                device_limits,
                device_features: features,
                backend: adapter.get_info().backend,
                queue,
                rand_state: Mutex::new(rand::rngs::StdRng::from_os_rng()),
                #[cfg(feature = "wgpu_debug")]
                debug: debug_info,
                command_queue: Mutex::new(QueueBufferInner::new(configuration.meta_buffer_size)),
                meta_buffer,
                cache: Mutex::new(ModelCache::new(
                    configuration.buffer_mapping_size,
                    max_memory_size,
                )),
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

    pub fn add_wgpu_shader_loader<T: candle_wgpu_kernels::ShaderLoader + 'static + Send + Sync>(
        &self,
        index: candle_wgpu_kernels::LoaderIndex,
        shader_loader: impl Fn() -> T,
    ) {
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
            cache.unary_inplace_counter, cache.binary_inplace_counter, cache.copy_inplace_counter
        );
    }

    #[cfg(feature = "wgpu_debug")]
    pub fn log_debuginfo_to_file(
        &self,
        folder: &str,
        name: &str,
        version: &str,
    ) -> crate::Result<()> {
        let info = pollster::block_on(self.get_debug_info()).unwrap();
        let map2 = crate::wgpu::debug_info::calulate_measurment(&info);
        crate::wgpu::debug_info::save_list(
            &map2,
            &format!("{folder}wgpu_{name}_test_{version}_measurements.json"),
        )
        .unwrap();

        let info: Vec<crate::wgpu::debug_info::ShaderInfo> = self.get_pipeline_info().unwrap();
        crate::wgpu::debug_info::save_list(
            &info,
            &format!("{folder}wgpu_{name}_test_{version}_shaders.json"),
        )
        .unwrap();

        let (pipelines, consts) = self.get_used_pipelines();
        std::fs::write(
            format!("{folder}wgpu_{name}_test_{version}_used_pipelines.json"),
            pipelines,
        )?;
        std::fs::write(format!("{folder}wgpu_{name}_test_{version}_used_consts.json"), consts)?;
        Ok(())
    }

    #[cfg(feature = "wgpu_debug")]
    pub fn get_used_pipelines(&self) -> (String, String) {
        let cache = self.cache.lock().unwrap();
        let queue = self.command_queue.lock().unwrap();
        let consts = &queue.id_to_const_array;

        let debug: Vec<_> = cache.debug.values().collect();

        (
            serde_json::to_string(&debug).unwrap(),
            serde_json::to_string(consts).unwrap(),
        )
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

        let q = MlQueue::Dispatch(MlQueueDispatch {
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

        return Ok(shaders
            .shaders
            .iter()
            .map(|(k, v)| {
                let pipelines = &v.pipelines;
                let s = debug_info::ShaderInfo {
                    name: shaders.loader_cache.get_shader_name(*k),
                    pipelines: pipelines
                        .iter()
                        .map(|(pk, _)| debug_info::PipelineInfo {
                            name: shaders.loader_cache.get_entry_point(pk.0).to_owned(),
                            consts: queue.id_to_const_array[pk.1].iter()
                                .map(|(s, x)| (s.to_string(), *x))
                                .collect(),
                        })
                        .collect(),
                };
                s
            })
            .collect());
    }

    pub(crate) async fn synchronize_async(&self) -> crate::Result<()> {
        wgpu_functions::synchronize_async(self).await
    }

    pub fn is_dtype_available(&self, dtype: DType) -> bool {
        match dtype {
            DType::U32 => true,
            DType::F32 => true,
            DType::U8 => false,
            DType::I64 => self.device_features.contains(wgpu::Features::SHADER_INT64),
            DType::F64 => self.device_features.contains(wgpu::Features::SHADER_F64),
            DType::F16 => self.device_features.contains(wgpu::Features::SHADER_F16),
            DType::BF16 => false,
            DType::F8E4M3 => false,
        }
    }


    
    #[instrument(skip(self, size))]
    pub fn alloc_uninit_size<T: ToU64>(
        &self,
        dtype: crate::DType,
        size: T,
    ) -> WgpuStorage {
        let size = size.to_u64() * dtype.size_in_bytes() as u64;
        let buffer;
        {
            let mut cache = self.cache.lock().unwrap();
            buffer = cache.create_buffer_reference(size, true);
        }
        return WgpuStorage::new(buffer, self.clone(), dtype, size);
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
    pub fn alloc_from_bytes(
        &self,
        dtype: crate::DType,
        data: &[u8],
    ) -> crate::Result<WgpuStorage> {
        let size = data.len();
        let buffer;
        {
            if self.configuration.flush_gpu_before_buffer_init {
                self.flush_gpu_command()?;
            }
            let mut cache = self.cache.lock().unwrap();
            buffer = cache.create_buffer_reference_init(self, data, true);
        }
        return Ok(WgpuStorage::new(buffer, self.clone(), dtype, size as u64));
    }


    pub fn allocate_zeros(&self, size_in_bytes : u32) -> crate::Result<WgpuStorage>{
        self.zeros_impl(&((size_in_bytes/4) as usize,).into(), DType::U32)
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
            BindgroupAlignmentLayout::Bindgroup3(alignment, alignment, alignment,alignment),
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
        self.device_id == other.device_id
    }

    fn zeros_impl(
        &self,
        shape: &crate::Shape,
        dtype: crate::DType,
    ) -> crate::Result<Self::Storage> {
        let buffer = self.alloc_uninit_size(dtype, shape.elem_count());
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

    unsafe fn alloc_uninit(
        &self,
        shape: &crate::Shape,
        dtype: crate::DType,
    ) -> crate::Result<Self::Storage> {
        if self.is_dtype_available(dtype) {
            Ok(self.alloc_uninit_size(dtype, shape.elem_count()))
        } else {
            wrongType!(alloc_uninit, dtype);
        }
    }

    fn storage_from_slice<T: crate::WithDType>(&self, data: &[T]) -> crate::Result<Self::Storage> {
        let data =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * T::DTYPE.size_in_bytes()) };
        let buffer = self.alloc_from_bytes(T::DTYPE, data)?;
        Ok(buffer)
    }

    fn storage_from_cpu_storage(
        &self,
        storage: &crate::CpuStorage,
    ) -> crate::Result<Self::Storage> {
        match storage {
            crate::CpuStorage::F32(data) => self.alloc_from_slice(crate::DType::F32, data),
            crate::CpuStorage::U32(data) => self.alloc_from_slice(crate::DType::U32, data),
            crate::CpuStorage::F16(data) => self.alloc_from_slice(crate::DType::F16, data),
            crate::CpuStorage::F64(data) => self.alloc_from_slice(crate::DType::F64, data),
            crate::CpuStorage::I64(data) => self.alloc_from_slice(crate::DType::I64, data),
            _ => wrongType!(storage_from_cpu_storage, storage.dtype()),
        }
    }

    fn storage_from_cpu_storage_owned(
        &self,
        storage: crate::CpuStorage,
    ) -> crate::Result<Self::Storage> {
        match storage {
            crate::CpuStorage::F32(data) => {
                self.alloc_from_slice(crate::DType::F32, &data)
            }
            crate::CpuStorage::U32(data) => {
                self.alloc_from_slice(crate::DType::U32, &data)
            }
            crate::CpuStorage::I64(data) => {
                self.alloc_from_slice(crate::DType::I64, &data)
            }
            crate::CpuStorage::F64(data) => {
                self.alloc_from_slice(crate::DType::F64, &data)
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
        let buffer = self.alloc_uninit_size(dtype, shape.elem_count());
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
        let buffer = self.alloc_uninit_size(dtype, shape.elem_count());
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
