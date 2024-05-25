use std::sync::{Arc, Mutex};

use rand::SeedableRng;


use crate::backend::BackendStorage;
use crate::{notImplemented, wrongType, Layout};

#[cfg(feature = "wgpu_debug")]
use super::debug_info::{DebugInfo, Measurements,MInfo};


use super::wgpu_functions::{self, create_buffer, create_buffer_init, UnaryOperation};
use super::WgpuStorage;

#[derive(Debug, Clone)]
pub struct  WgpuDevice {
    pub device : Arc<wgpu::Device>, 
    pub queue : Arc<wgpu::Queue>,
    pub pipelines : Arc<Mutex<Vec<Option<Arc<wgpu::ComputePipeline>>>>>,
    pub shader : Arc<Vec<wgpu::ShaderModule>>,
    pub rand_state : Arc<Mutex<rand::rngs::StdRng>>,

    #[cfg(feature = "wgpu_debug")]
    pub debug : DebugInfo,
}


const PIPELINES_COUNT : usize = 30;

#[derive(Debug, Clone)]
pub (crate) enum Pipelines{
    UnaryInplace = 0,
    UnaryFromBuffer,
    UnaryFromBufferContiguous,
    BinaryBufferInplace,
    BinaryBufferFromBuffer,
    BinaryBufferFromBufferContiguousBoth,
    MatmulBuffer,
    Reduce,
    ReduceIndex,
    RmsNorm,
    CmpFromBuffer ,
    Conv2D,
    Conv2DTranspose,
    ConvertF32ToU32,
    IndexSelect,
    Copy2d,
    CopyStrided,

    UnaryInplaceU32,
    UnaryFromBufferU32,
    UnaryFromBufferContiguousU32,
    BinaryBufferFromBufferContiguousBothU32,
    BinaryBufferInplaceU32,
    BinaryBufferFromBufferU32,
    MatmulBufferU32,
    ReduceU32,
    ReduceIndexU32,
    CmpFromBufferU32,
    Conv2DU32,
    Conv2DTransposeU32,
    ConvertU32ToF32,
}




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
        
        limits.max_buffer_size = 256 * 1000000000;
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
        let shader1 =  wgpu_functions::get_shader(&device, include_str!("shader.wgsl"));
        let shader2 =  wgpu_functions::get_shader(&device,include_str!("shader_u32.wgsl"));

        #[cfg(feature = "wgpu_debug")]
        let debug_info = super::debug_info::DebugInfo::new(&device);

        Ok(WgpuDevice {
            device: Arc::new(device),
            queue: Arc::new(queue),
            pipelines : Arc::new(Mutex::new(vec![None;PIPELINES_COUNT])),
            shader : Arc::new(vec![shader1,shader2]),
            rand_state: Arc::new(Mutex::new(rand::rngs::StdRng::from_entropy())),
            #[cfg(feature = "wgpu_debug")]
            debug : debug_info
        })
    }

    
    #[cfg(feature = "wgpu_debug")]
    pub async fn get_debug_info_full(&self) -> crate::Result<Measurements>{
        let data = wgpu_functions::read_data_from_gpu_async::<u64>(self, &self.debug.query_set_buffer).await;
        let period = self.queue.get_timestamp_period();
        let mut result = Measurements::new(period);
        for p in self.debug.shader_pipeline.lock().unwrap().iter(){
            result.data.push(MInfo::new(p.1.to_owned(), data[(*(p.0) * 32) as usize], data[(*(p.0) * 32) as usize + 1]));
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

    fn load_pipeline(device : &wgpu::Device, shader : &[wgpu::ShaderModule], pipeline : Pipelines) -> wgpu::ComputePipeline{
        let (entry_point, shader_index) = match pipeline{
            Pipelines::UnaryInplace => ("unary_inplace", 0),
            Pipelines::UnaryFromBuffer => ("unary_from_buffer", 0),
            Pipelines::UnaryFromBufferContiguous => ("unary_from_buffer_contiguous", 0),
            Pipelines::BinaryBufferInplace => ("binary_buffer_inplace", 0),
            Pipelines::BinaryBufferFromBuffer => ("binary_buffer_from_buffer", 0),
            Pipelines::BinaryBufferFromBufferContiguousBoth => ("binary_buffer_from_buffer_contiguous_both", 0),
            Pipelines::MatmulBuffer => ("matmul", 0),
            Pipelines::Reduce => ("reduce", 0),
            Pipelines::ReduceIndex => ("reduce_index", 0),
            Pipelines::RmsNorm => ("rms_norm", 0),
            Pipelines::CmpFromBuffer => ("cmp_buffer_from_buffer", 0),
            Pipelines::Conv2D => ("conv2d", 0),
            Pipelines::Conv2DTranspose => ("conv2d_transpose", 0),
            Pipelines::ConvertF32ToU32 => ("convert_to_u32", 0),
            Pipelines::IndexSelect => ("index_select", 0),
            Pipelines::Copy2d => ("copy2d", 0),
            Pipelines::CopyStrided => ("copy_strided", 0),


            Pipelines::UnaryInplaceU32 => ("unary_inplace", 1),
            Pipelines::UnaryFromBufferU32 => ("unary_from_buffer", 1),
            Pipelines::UnaryFromBufferContiguousU32 => ("unary_from_buffer_contiguous", 1),
            Pipelines::BinaryBufferInplaceU32 => ("binary_buffer_inplace", 1),
            Pipelines::BinaryBufferFromBufferU32 => ("binary_buffer_from_buffer", 1),
            Pipelines::BinaryBufferFromBufferContiguousBothU32 => ("binary_buffer_from_buffer_contiguous_both", 1),
            Pipelines::MatmulBufferU32 => ("matmul", 1),
            Pipelines::ReduceU32 => ("reduce", 1),
            Pipelines::ReduceIndexU32 => ("reduce_index", 1),
            Pipelines::CmpFromBufferU32 => ("cmp_buffer_from_buffer", 1),
            Pipelines::Conv2DU32 => ("conv2d", 1),
            Pipelines::Conv2DTransposeU32 => ("conv2d_transpose", 1),
            Pipelines::ConvertU32ToF32 => ("convert_to_f32", 1),
            
        };
        
        return  device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader[shader_index],
            entry_point: entry_point,
        });
    }

    pub (crate) fn get_pipeline(&self,pipeline: Pipelines) -> Arc<wgpu::ComputePipeline> { //Ref<'_, wgpu::ComputePipeline> 
        let mut pipelines = self.pipelines.lock().unwrap(); 
        let index = pipeline.clone() as usize;

        if pipelines[index].is_none(){
            let p = crate::WgpuDevice::load_pipeline(&self.device, &self.shader[..], pipeline);
            pipelines[index] = Some(Arc::new(p));
        }
     
        if let Some(p) = &pipelines[index]{
            return p.clone();
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
        return crate::DeviceLocation::Cpu; //TODO WGPU
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
        if T::DTYPE != crate::DType::F32 {
            // Panic if T is not f32
            wrongType!(storage_from_slice, T::DTYPE);
        }
        
        // Safe to cast data to &[f32] since T is f32
        // This is safe because T is known to be f32 due to the above check
        let data = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len()) };
        let buffer = create_buffer_init(self, &data);
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
