use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use rand::SeedableRng;


use crate::backend::BackendStorage;
use crate::{notImplemented, wrongType, Layout};

#[cfg(feature = "wgpu_debug")]
use super::debug_info::{DebugInfo, Measurements,MInfo};

use super::model::ModelCache;
use super::wgpu_functions::{DispatchIndirectArgs, MetaArray, MetaArrayToU32};
use super::wgpu_functions::{self, create_buffer, create_buffer_init, unary::UnaryOperation};
use super::WgpuStorage;

#[derive(Debug)]
pub (crate) enum MlQueue{
    Dispatch(MlQueueDispatch),
}

#[cfg(feature = "wgpu_debug")]
pub trait ToU64 {
    fn to_u64(self) -> u64;
}
#[cfg(feature = "wgpu_debug")]
macro_rules! impl_to_u64 {
    ($($ty:ty)*) => {
        $(
            impl ToU64 for $ty {
                #[inline]
                fn to_u64(self) -> u64 {
                    self as u64
                }
            }
        )*
    }
}
#[cfg(feature = "wgpu_debug")]
impl_to_u64!(u8 u16 u32 u64 usize);

#[cfg(feature = "wgpu_debug")]
#[derive(Debug)]
pub struct QueueDebugInfo{
    pub (crate) name : Option<String>,
    pub (crate) output_size : u64,
}

#[cfg(feature = "wgpu_debug")]
impl QueueDebugInfo {
    pub fn new<T : Into<String>, O : ToU64>(name: T, output_size: O) -> Self {
        Self { name : Some(name.into()), output_size: output_size.to_u64()}
    }
    // pub fn new_noname<O : ToU64>(output_size: O) -> Self {
    //     Self { name : None, output_size: output_size.to_u64() }
    // }
}



#[derive(Debug)]
pub (crate) struct MlQueueDispatch{
    pub (crate) x : u32, 
    pub (crate) y : u32, 
    pub (crate) z : u32,
    pub (crate) pipeline : Arc<wgpu::ComputePipeline>,
    pub (crate) bind_group : Arc<wgpu::BindGroup>,
    pub (crate) indirect_buffer : Option<usize>,
    
    #[cfg(feature = "wgpu_debug")]
    pub (crate) debug : QueueDebugInfo,
}

#[derive(Debug)]
pub struct ShaderModuleComputePipelines{
    shader : Arc<wgpu::ShaderModule>,
    pipelines : Mutex<HashMap<Pipelines, Arc<wgpu::ComputePipeline>>>
}

//a struct, where all operations are chunked 
#[derive(Debug)]
pub struct QueueBuffer{
    pub (crate) command_queue : Vec<MlQueue>,
    pub (crate) meta_array : MetaArray,
    pub (crate) indirect_array : Vec<DispatchIndirectArgs>,
}

impl QueueBuffer {
    pub fn new() -> Self {
        Self {  command_queue: vec![], meta_array :MetaArray::new(META_BUFFER_SIZE), indirect_array : vec![] }
    }

    pub (crate) fn add_layout(&mut self, layout : &Layout){
        self.meta_array.add_layout(layout);
    } 

    pub (crate) fn add<T : MetaArrayToU32>(&mut self, value : T){
        self.meta_array.add(value);
    }

}

#[derive(Debug, Clone)]
pub struct  WgpuDevice {
    pub device : Arc<wgpu::Device>, 
    pub device_limits : Arc<wgpu::Limits>, //we cache the limits here, because device.limit() was relatively slow on the browser

    pub queue : Arc<wgpu::Queue>,
    pub (crate) shader : Arc<Mutex<HashMap<wgpu_functions::Shader, ShaderModuleComputePipelines>>>,
    pub (crate) rand_state : Arc<Mutex<rand::rngs::StdRng>>,   

    pub (crate) command_queue : Arc<Mutex<QueueBuffer>>,
    pub (crate) meta_buffer : Arc<wgpu::Buffer>, //buffer for storing meta information
    pub (crate) indirect_buffer : Arc<wgpu::Buffer>, //buffer for storing meta information

    pub (crate) cache : Arc<Mutex<Option<ModelCache>>>, //if cache is set, all commands are not queued to the gpu, but are cached inside ModelCache, so there can be reused later on
    pub (crate) cache_counter : Arc<Mutex<u32>>,


    #[cfg(feature = "wgpu_debug")]
    pub debug : DebugInfo,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub (crate) enum Pipelines{
    UnaryInplace = 0,
    UnaryFromBuffer,
    UnaryFromBufferContiguous,
    BinaryBufferFromBuffer,
    BinaryBufferFromBufferContiguousBoth,
    MatmulBuffer,
    MatmulBuffer1b,
    Matmul3Buffer,
    Matmul4Buffer,
    Matmul4endBuffer,
    Matmul5Buffer, 
    Matmul6Buffer,
    Matmul1endBuffer,
    Reduce,
    ReduceIndex,
    RmsNorm,
    Softmax,
    CmpFromBuffer ,
    Conv2D,
    Conv2DTranspose,
    Conv1D,
    Conv1DTranspose,
    IndexSelect,
    Copy2d,
    CopyStrided,
    Copy,
    ConvertF32ToU32,
    ConvertU32ToF32,
    ConvertU8ToF32,
    WhereCondU32,
    MaxPool2d,
    AvgPool2d,
    Upsample1d,
    Upsample2d,
    Gather,
    ScatterAddInplace,
    IndexAddInplace
}


pub (crate) const META_BUFFER_SIZE : u32 = 65536;
pub (crate) const INDIRECT_BUFFER_SIZE : u32 = 10000;

impl WgpuDevice{
    pub (crate) async fn create(_: usize) -> crate::Result<Self>{
        let instance = wgpu::Instance::default();

        // `request_adapter` instantiates the general connection to the GPU
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();

        let mut limits = wgpu::Limits::downlevel_defaults();

        #[cfg(feature = "wgpu_debug")]
        let features = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
        #[cfg(not(feature = "wgpu_debug"))]
        let features = wgpu::Features::empty();

        limits.max_storage_buffers_per_shader_stage = 5;
        limits.max_storage_buffer_binding_size = adapter.limits().max_storage_buffer_binding_size; //use as much as possible
        limits.max_buffer_size = adapter.limits().max_buffer_size; //use as much as possible

     
        // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
        //  `features` being the available features.
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: features,
                    required_limits: limits,
                },
                None,
            ).await.map_err(|err| crate::Error::WebGpu(err.to_string().into()))?;
    
        
        #[cfg(feature = "wgpu_debug")]
        let debug_info = super::debug_info::DebugInfo::new(&device);
        
        let meta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: META_BUFFER_SIZE as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: INDIRECT_BUFFER_SIZE as u64,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let device_limits = device.limits();
        Ok(WgpuDevice {
            device: Arc::new(device),
            device_limits: Arc::new(device_limits),
            queue: Arc::new(queue),
            shader : Arc::new(Mutex::new(HashMap::new())),
            rand_state: Arc::new(Mutex::new(rand::rngs::StdRng::from_entropy())),
            #[cfg(feature = "wgpu_debug")]
            debug : debug_info,
            command_queue: Arc::new(Mutex::new(QueueBuffer::new())),
            meta_buffer : Arc::new(meta_buffer),
            indirect_buffer : Arc::new(indirect_buffer),
            cache : Arc::new(Mutex::new(None)),
            cache_counter : Arc::new(Mutex::new(0))
        })
    }

    
    #[cfg(feature = "wgpu_debug")]
    pub async fn get_debug_info_full(&self) -> crate::Result<Measurements>{
        let data = wgpu_functions::read_data_from_gpu_async::<u64>(self, &self.debug.query_set_buffer).await;
        let period = self.queue.get_timestamp_period();
        let mut result = Measurements::new(period);
        let mut last_end_time = 0u64;
        let mut i = 0;
        let mut shader_pipeline2 = self.debug.shader_pipeline.lock().unwrap();
        let shader_pipeline = shader_pipeline2.clone();
        let mut indexes : Vec<_> = shader_pipeline.into_iter().collect();
        indexes.sort_by_key(|f| f.0);
        for p in indexes{
            let start_time =data[(p.0 / 8) as usize];
            let end_time = data[(p.0 / 8) as usize + 1];
            if end_time < start_time {
                panic!("Start Time was after End Time! startTime: {start_time}, endTime:{end_time}, i:{i}");
            }

            if start_time < last_end_time {
                panic!("Start Time was before last End Time! startTime: {start_time}, last endTime:{last_end_time}, i:{i}");
            }
            last_end_time = end_time;
            i += 1;
            result.data.push(MInfo::new(p.1.0.to_owned(), start_time, end_time, p.1.1, p.1.2, p.1.3, p.1.4));
        }
        self.debug.counter.store(0u32, std::sync::atomic::Ordering::Relaxed);
        shader_pipeline2.clear();
        Ok(result)
    }

    #[cfg(feature = "wgpu_debug")]
    pub async fn get_debug_info(&self) -> crate::Result<std::collections::HashMap<String, Vec<(u64, u64, u32, u32, u32)>>>{
        let info = self.get_debug_info_full().await?;
        let mut map: std::collections::HashMap<String, Vec<(u64, u64, u32, u32, u32)>> = std::collections::HashMap::new();

        for item in info.data.iter() {
            map.entry(item.label.clone()).or_insert_with(Vec::new).push((item.end_time - item.start_time, item.output_size, item.x, item.y, item.z));
        }
        return Ok(map);
    }

    fn load_pipeline(device : &wgpu::Device, shader : Arc<wgpu::ShaderModule>, pipeline : Pipelines) -> wgpu::ComputePipeline{
        let entry_point = match pipeline{
            Pipelines::UnaryInplace => "unary_inplace",
            Pipelines::UnaryFromBuffer => "unary_from_buffer",
            Pipelines::UnaryFromBufferContiguous => "unary_from_buffer_contiguous",
            Pipelines::BinaryBufferFromBuffer => "binary_buffer_from_buffer",
            Pipelines::BinaryBufferFromBufferContiguousBoth => "binary_buffer_from_buffer_contiguous_both",
            Pipelines::MatmulBuffer => "matmul1",
            Pipelines::MatmulBuffer1b => "matmul1b",
            Pipelines::Matmul3Buffer => "matmul3",
            Pipelines::Matmul4Buffer => "matmul4",
            Pipelines::Matmul4endBuffer => "matmul4_end",
            Pipelines::Matmul1endBuffer => "matmul1_end",
            Pipelines::Matmul5Buffer => "matmul5",
            Pipelines::Matmul6Buffer => "matmul6",
            Pipelines::Reduce => "reduce",
            Pipelines::ReduceIndex => "reduce_index",
            Pipelines::RmsNorm => "rms_norm",
            Pipelines::Softmax => "softmax",
            Pipelines::CmpFromBuffer => "cmp_buffer_from_buffer",
            Pipelines::Conv2D => "conv2d",
            Pipelines::Conv2DTranspose => "conv2d_transpose",
            Pipelines::Conv1D => "conv1d",
            Pipelines::Conv1DTranspose => "conv1d_transpose",
            Pipelines::ConvertF32ToU32 => "convert_to_u32",
            Pipelines::ConvertU32ToF32 => "convert_to_f32",
            Pipelines::ConvertU8ToF32 => "convert_u8_to_f32",
            Pipelines::IndexSelect => "index_select",
            Pipelines::Copy2d => "copy2d",
            Pipelines::CopyStrided => "copy_strided",
            Pipelines::Copy => "copy",
            Pipelines::WhereCondU32 => "where_cond_index_u32",
            Pipelines::MaxPool2d => "max_pool2d",
            Pipelines::AvgPool2d => "avg_pool2d",
            Pipelines::Upsample1d => "upsample1d",
            Pipelines::Upsample2d => "upsample2d",
            Pipelines::Gather => "gather",
            Pipelines::ScatterAddInplace => "scatter_add_inplace",
            Pipelines::IndexAddInplace => "index_add_inplace",
        };
        
        return  device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: entry_point,
        });
    }

    pub (crate) fn get_pipeline(&self, shader : wgpu_functions::Shader, pipeline: Pipelines) -> crate::Result<Arc<wgpu::ComputePipeline>> {
        let mut shaders = self.shader.lock().unwrap();
        
        if !shaders.contains_key(&shader){
            let s = wgpu_functions::get_shader(&self.device, wgpu_functions::load_shader(shader.clone())?);
            shaders.insert(shader.clone(), ShaderModuleComputePipelines{ shader: Arc::new(s), pipelines: Mutex::new(HashMap::new())});
        }
     
        if let Some(s) = shaders.get(&shader){
            let mut pipelines = s.pipelines.lock().unwrap();

            if !pipelines.contains_key(&pipeline){
                let p = crate::WgpuDevice::load_pipeline(&self.device, s.shader.clone(), pipeline.clone());
                pipelines.insert(pipeline.clone(), Arc::new(p));
            }
            
            if let Some(p) = pipelines.get(&pipeline){
                return Ok(p.clone());
            }
            else{
                panic!("Not expected")
            }
        }
        else{
            panic!("Not expected")
        }
    }
}



impl crate::backend::BackendDevice for WgpuDevice{
    type Storage = WgpuStorage;

    fn new(_: usize) -> crate::Result<Self> {
        return Err(crate::Error::WebGpu("A WgpuDevice must be created using the asynchronous create method".to_owned().into()));
    }

    fn location(&self) -> crate::DeviceLocation {
        return crate::DeviceLocation::Wgpu { gpu_id: 0 }; //TODO: WGPU
    }

    fn same_device(&self, other: &Self) -> bool {
        return self.device.global_id() == other.device.global_id();
    }

    fn zeros_impl(&self, shape: &crate::Shape, dtype: crate::DType) -> crate::Result<Self::Storage> {
        let buffer = create_buffer(self, shape.elem_count() * 4);
        if shape.elem_count() > 0{
            wgpu_functions::queue_unary_inplace_op(self, &buffer, UnaryOperation::SetZero, 0.0, 0.0,dtype, Layout::contiguous(shape))?;
        }
        
        return Ok(WgpuStorage::new(buffer, self.clone(), dtype));
    }

    fn ones_impl(&self, shape: &crate::Shape, dtype: crate::DType) -> crate::Result<Self::Storage> {
        let buffer = create_buffer(self, shape.elem_count() * 4);
        if shape.elem_count() > 0{
            wgpu_functions::queue_unary_inplace_op(self, &buffer, UnaryOperation::SetOne, 0.0, 0.0,dtype,Layout::contiguous(shape))?;
        }
        return Ok(WgpuStorage::new(buffer, self.clone(), dtype));
    }

    unsafe fn alloc_uninit(&self, shape: &crate::Shape, dtype: crate::DType) -> crate::Result<Self::Storage> {
        if dtype == crate::DType::F32 || dtype == crate::DType::U32{
            let buffer = create_buffer(self, shape.elem_count() * 4);
            return Ok(WgpuStorage::new(buffer, self.clone(), dtype));
        }
        else{
            wrongType!(alloc_uninit, dtype);
        }
    }

    fn storage_from_slice<T: crate::WithDType>(&self, data : &[T]) -> crate::Result<Self::Storage> {
        let buffer;
        if T::DTYPE == crate::DType::F32{
            let data = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len()) };
            buffer = create_buffer_init(self, &data);
        }
        else if T::DTYPE == crate::DType::U32{
            let data = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u32, data.len()) };
            buffer = create_buffer_init(self, &data);
        }
        else{
            // Panic if T is not f32 or u32
            wrongType!(storage_from_slice, T::DTYPE);
        }
        return Ok(WgpuStorage::new(buffer, self.clone(),T::DTYPE));
    }

    fn storage_from_cpu_storage(&self, storage: &crate::CpuStorage) -> crate::Result<Self::Storage> {
        match storage{
            crate::CpuStorage::F32(data) => {
                let buffer = create_buffer_init(self, data);
                return Ok(WgpuStorage::new(buffer, self.clone(),crate::DType::F32));
            },
            crate::CpuStorage::U32(data) => {
                let buffer = create_buffer_init(self, data);
                return Ok(WgpuStorage::new(buffer, self.clone(),crate::DType::U32));
            },
            _ =>  wrongType!(storage_from_cpu_storage, storage.dtype()),
        }
    }

    fn storage_from_cpu_storage_owned(&self, storage: crate::CpuStorage) -> crate::Result<Self::Storage> {
        match storage{
            crate::CpuStorage::F32(data) => {
                let buffer = create_buffer_init(self, &data);
                return Ok(WgpuStorage::new(buffer, self.clone(),crate::DType::F32));
            },
            crate::CpuStorage::U32(data) => {
                let buffer = create_buffer_init(self, &data);
                return Ok(WgpuStorage::new(buffer, self.clone(),crate::DType::U32));
            },
            _ =>  wrongType!(storage_from_cpu_storage_owned, storage.dtype()),
        }
    }

    fn rand_uniform(&self, shape: &crate::Shape, dtype: crate::DType, lo: f64, up: f64) -> crate::Result<Self::Storage> {
        let buffer = create_buffer(self, shape.elem_count() * 4);
        wgpu_functions::queue_unary_inplace_op(self, &buffer, UnaryOperation::RandUniform, lo as f32, up as f32,dtype,Layout::contiguous(shape))?;
        return Ok(WgpuStorage::new(buffer, self.clone(), dtype));
    }

    fn rand_normal(&self, shape: &crate::Shape, dtype: crate::DType, mean: f64, std: f64) -> crate::Result<Self::Storage> {
        let buffer = create_buffer(self, shape.elem_count() * 4);
        wgpu_functions::queue_unary_inplace_op(self, &buffer, UnaryOperation::RandNormal, mean as  f32, std as f32, dtype,Layout::contiguous(shape))?;
        return Ok(WgpuStorage::new(buffer, self.clone(),dtype));
    }

    fn set_seed(&self, _: u64) -> crate::Result<()> {
        notImplemented!(set_seed)
    }

    fn synchronize(&self) -> crate::Result<()> {
        wgpu_functions::synchronize(self)
    }
}
