use std::sync::{Arc, Mutex};

use rand::SeedableRng;

use crate::backend::BackendStorage;
use crate::{notImplemented, wrongType, Layout};

use super::wgpu_functions::{self, create_buffer, create_buffer_init, UnaryOperation};
use super::WgpuStorage;


#[derive(Debug, Clone)]
pub struct  WgpuDevice {
    pub device : Arc<wgpu::Device>, 
    pub queue : Arc<wgpu::Queue>,
    pub pipelines : Arc<Vec<wgpu::ComputePipeline>>,
    pub shader : Arc<wgpu::ShaderModule>,
    pub rand_state : Arc<Mutex<rand::rngs::StdRng>>
}

pub (crate) enum Pipelines{
    UnaryInplace = 0,
    UnaryFromBuffer,
    BinaryBufferInplace,
    BinaryBufferFromBuffer,
    MatmulBuffer,
    Reduce,
    ReduceIndex,
    CmpFromBuffer ,
    Conv2D,
    Conv2DTranspose,

    UnaryInplaceU32,
    UnaryFromBufferU32,
    BinaryBufferInplaceU32,
    BinaryBufferFromBufferU32,
    MatmulBufferU32,
    ReduceU32,
    ReduceIndexU32,
    CmpFromBufferU32,
    Conv2DU32,
    Conv2DTransposeU32,
}


impl WgpuDevice{
    pub (crate) async fn create(_: usize) -> crate::Result<Self>{
        let instance = wgpu::Instance::default();

        // `request_adapter` instantiates the general connection to the GPU
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();

        // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
        //  `features` being the available features.
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            ).await.map_err(|err| crate::Error::WebGpu(err.to_string().into()))?;
        let shader1 =  wgpu_functions::get_shader(&device, include_str!("shader.wgsl"));
        let shader2 =  wgpu_functions::get_shader(&device,include_str!("shader_u32.wgsl"));
        
        let pipelines = 
        vec![
            Self::load_pipeline(&device, &shader1, Pipelines::UnaryInplace), 
            Self::load_pipeline(&device, &shader1, Pipelines::UnaryFromBuffer), 
            Self::load_pipeline(&device, &shader1, Pipelines::BinaryBufferInplace), 
            Self::load_pipeline(&device, &shader1, Pipelines::BinaryBufferFromBuffer), 
            Self::load_pipeline(&device, &shader1, Pipelines::MatmulBuffer), 
            Self::load_pipeline(&device, &shader1, Pipelines::Reduce), 
            Self::load_pipeline(&device, &shader1, Pipelines::ReduceIndex), 
            Self::load_pipeline(&device, &shader1, Pipelines::CmpFromBuffer), 
            Self::load_pipeline(&device, &shader1, Pipelines::Conv2D),
            Self::load_pipeline(&device, &shader1, Pipelines::Conv2DTranspose),

            Self::load_pipeline(&device, &shader2, Pipelines::UnaryInplaceU32),  
            Self::load_pipeline(&device, &shader2, Pipelines::UnaryFromBufferU32), 
            Self::load_pipeline(&device, &shader2, Pipelines::BinaryBufferInplaceU32), 
            Self::load_pipeline(&device, &shader2, Pipelines::BinaryBufferFromBufferU32),
            Self::load_pipeline(&device, &shader2, Pipelines::MatmulBufferU32),
            Self::load_pipeline(&device, &shader2, Pipelines::ReduceU32),
            Self::load_pipeline(&device, &shader2, Pipelines::ReduceIndexU32),
            Self::load_pipeline(&device, &shader2, Pipelines::CmpFromBufferU32),
            Self::load_pipeline(&device, &shader1, Pipelines::Conv2DU32),
            Self::load_pipeline(&device, &shader1, Pipelines::Conv2DTransposeU32),
            ];
        
        Ok(WgpuDevice {
            device: Arc::new(device),
            queue: Arc::new(queue),
            pipelines : Arc::new(pipelines),
            shader : Arc::new(shader1),
            rand_state: Arc::new(Mutex::new(rand::rngs::StdRng::from_entropy()))
        })
    }

    
    fn load_pipeline(device : &wgpu::Device, shader : &wgpu::ShaderModule, pipeline : Pipelines) -> wgpu::ComputePipeline{
        let entry_point = match pipeline{
            Pipelines::UnaryInplace => "unary_inplace",
            Pipelines::UnaryFromBuffer => "unary_from_buffer",
            Pipelines::BinaryBufferInplace => "binary_buffer_inplace",
            Pipelines::BinaryBufferFromBuffer => "binary_buffer_from_buffer",
            Pipelines::MatmulBuffer => "matmul",
            Pipelines::Reduce => "reduce",
            Pipelines::ReduceIndex => "reduce_index",
            Pipelines::CmpFromBuffer => "cmp_buffer_from_buffer",
            Pipelines::Conv2D => "conv2d",
            Pipelines::Conv2DTranspose => "conv2d_transpose",

            Pipelines::UnaryInplaceU32 => "unary_inplace",
            Pipelines::UnaryFromBufferU32 => "unary_from_buffer",
            Pipelines::BinaryBufferInplaceU32 => "binary_buffer_inplace",
            Pipelines::BinaryBufferFromBufferU32 => "binary_buffer_from_buffer",
            Pipelines::MatmulBufferU32 => "matmul",
            Pipelines::ReduceU32 => "reduce",
            Pipelines::ReduceIndexU32 => "reduce_index",
            Pipelines::CmpFromBufferU32 => "cmp_buffer_from_buffer",
            Pipelines::Conv2DU32 => "conv2d",
            Pipelines::Conv2DTransposeU32 => "conv2d_transpose",
        };
        
        return  device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: entry_point,
        });
    }

    pub (crate) fn get_pipeline(&self,pipeline: Pipelines) -> &wgpu::ComputePipeline { //Ref<'_, wgpu::ComputePipeline> 
        return &self.pipelines[pipeline as usize];
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
