use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use rand::SeedableRng;


use crate::backend::BackendStorage;
use crate::{notImplemented, wrongType, Layout};

#[cfg(feature = "wgpu_debug")]
use super::debug_info::{DebugInfo, Measurements,MInfo};

use super::wgpu_functions::{DispatchIndirectArgs, MetaArray};
use super::wgpu_functions::{self, create_buffer, create_buffer_init, unary::UnaryOperation};
use super::WgpuStorage;

#[derive(Debug)]
pub (crate) enum MlQueue{
    Dispatch(MlQueueDispatch),
}

#[derive(Debug)]
pub (crate) struct MlQueueDispatch{
    pub (crate) x : u32, 
    pub (crate) y : u32, 
    pub (crate) z : u32,
    pub (crate) pipeline : Arc<wgpu::ComputePipeline>,
    pub (crate) bind_group : wgpu::BindGroup,
    pub (crate) indirect_buffer : Option<usize>,
    
    #[cfg(feature = "wgpu_debug")]
    pub (crate) name : Option<String>,
}

#[derive(Debug)]
pub struct ShaderModuleComputePipelines{
    shader : Arc<wgpu::ShaderModule>,
    pipelines : Mutex<HashMap<Pipelines, Arc<wgpu::ComputePipeline>>>
}

#[derive(Debug, Clone)]
pub struct  WgpuDevice {
    pub device : Arc<wgpu::Device>, 
    pub queue : Arc<wgpu::Queue>,
    pub (crate) shader : Arc<Mutex<HashMap<wgpu_functions::Shader, ShaderModuleComputePipelines>>>,
    pub (crate) rand_state : Arc<Mutex<rand::rngs::StdRng>>,

    pub (crate) command_queue : Arc<Mutex<Vec<MlQueue>>>,

    pub (crate) meta_buffer : Arc<wgpu::Buffer>, //buffer for storing meta information
    pub (crate) meta_array : Arc<Mutex<MetaArray>>,

    pub (crate) indirect_buffer : Arc<wgpu::Buffer>, //buffer for storing meta information
    pub (crate) indirect_array : Arc<Mutex<Vec<DispatchIndirectArgs>>>,
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


pub (crate) const META_BUFFER_SIZE : u32 = 10000;
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
        #[cfg(feature = "wgpu_debug")]{
            limits.max_buffer_size = 2560000000;
        }
        limits.max_storage_buffers_per_shader_stage = 5;
        //limits.min_storage_buffer_offset_alignment = 4;

        #[cfg(not(feature = "wgpu_debug"))]
        let features = wgpu::Features::empty();
        
        //limits.max_compute_workgroups_per_dimension = 1024*1024;
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

        Ok(WgpuDevice {
            device: Arc::new(device),
            queue: Arc::new(queue),
            shader : Arc::new(Mutex::new(HashMap::new())),
            rand_state: Arc::new(Mutex::new(rand::rngs::StdRng::from_entropy())),
            #[cfg(feature = "wgpu_debug")]
            debug : debug_info,
            command_queue: Arc::new(Mutex::new(vec![])),
            meta_buffer : Arc::new(meta_buffer),
            meta_array : Arc::new(Mutex::new(MetaArray::new(META_BUFFER_SIZE))),
            indirect_buffer : Arc::new(indirect_buffer),
            indirect_array : Arc::new(Mutex::new(vec![]))

        })
    }

    
    #[cfg(feature = "wgpu_debug")]
    pub async fn get_debug_info_full(&self) -> crate::Result<Measurements>{
        let data = wgpu_functions::read_data_from_gpu_async::<u64>(self, &self.debug.query_set_buffer).await;
        let period = self.queue.get_timestamp_period();
        let mut result = Measurements::new(period);
        for p in self.debug.shader_pipeline.lock().unwrap().iter(){
            result.data.push(MInfo::new(p.1.to_owned(), data[(*p.0) as usize], data[(*p.0) as usize + 1]));
        }
        
        Ok(result)
    }

    #[cfg(feature = "wgpu_debug")]
    pub async fn get_debug_info(&self) -> crate::Result<std::collections::HashMap<String, Vec<u64>>>{
        let info = self.get_debug_info_full().await?;
        let mut map: std::collections::HashMap<String, Vec<u64>> = std::collections::HashMap::new();

        for item in info.data.iter() {
            map.entry(item.label.clone()).or_insert_with(Vec::new).push(item.end_time - item.start_time);
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
            Pipelines::MatmulBuffer => "matmul",
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
        notImplemented!(synchronize)
    }
}
