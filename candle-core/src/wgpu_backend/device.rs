use std::sync::Arc;
use super::wgpu_functions::{self, create_buffer, create_buffer_init, UnaryOperation};
use super::WgpuStorage;


#[derive(Debug, Clone)]
pub struct  WgpuDevice {
    pub device : Arc<wgpu::Device>, 
    pub queue : Arc<wgpu::Queue>
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

        Ok(WgpuDevice {
            device: Arc::new(device),
            queue: Arc::new(queue)
        })
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
        if dtype == crate::DType::F32{
            let buffer = create_buffer(self, shape.elem_count() * 4);
            wgpu_functions::queue_unary_inplace_op(self, &buffer, shape.elem_count() as u32,UnaryOperation::SetZero);
            return Ok(WgpuStorage::new(buffer, self.clone()));
        }
        else{
            panic!("can not create wgpu array of type {:?}, onlf f32 is allowed", dtype);
        }
    }

    fn ones_impl(&self, shape: &crate::Shape, dtype: crate::DType) -> crate::Result<Self::Storage> {
        if dtype == crate::DType::F32{
            let buffer = create_buffer(self, shape.elem_count() * 4);
            wgpu_functions::queue_unary_inplace_op(self, &buffer, shape.elem_count() as u32,UnaryOperation::SetOne);
            return Ok(WgpuStorage::new(buffer, self.clone()));
        }
        else{
            panic!("can not create wgpu array of type {:?}, onlf f32 is allowed", dtype);
        }
    }

    unsafe fn alloc_uninit(&self, shape: &crate::Shape, dtype: crate::DType) -> crate::Result<Self::Storage> {
        if dtype == crate::DType::F32{
            let buffer = create_buffer(self, shape.elem_count() * 4);
            return Ok(WgpuStorage::new(buffer, self.clone()));
        }
        else{
            panic!("can not create wgpu array of type {:?}, onlf f32 is allowed", dtype);
        }
    }

    fn storage_from_slice<T: crate::WithDType>(&self, data : &[T]) -> crate::Result<Self::Storage> {
        if T::DTYPE != crate::DType::F32 {
            // Panic if T is not f32
            panic!("Expected type T to be f32");
        }
        
        // Safe to cast data to &[f32] since T is f32
        // This is safe because T is known to be f32 due to the above check
        let data = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len()) };
        let buffer = create_buffer_init(self, &data);
        return Ok(WgpuStorage::new(buffer, self.clone()));
    }

    fn storage_from_cpu_storage(&self, storage: &crate::CpuStorage) -> crate::Result<Self::Storage> {
        match storage{
            crate::CpuStorage::F32(data) => {
                let buffer = create_buffer_init(self, data);
                return Ok(WgpuStorage::new(buffer, self.clone()));
            },
            _ =>  panic!("can not create wgpu array other than f32"),
        }
    }

    fn storage_from_cpu_storage_owned(&self, storage: crate::CpuStorage) -> crate::Result<Self::Storage> {
        match storage{
            crate::CpuStorage::F32(data) => {
                let buffer = create_buffer_init(self, &data);
                return Ok(WgpuStorage::new(buffer, self.clone()));
            },
            _ =>  panic!("can not create wgpu array other than f32"),
        }
    }

    fn rand_uniform(&self, _: &crate::Shape, _: crate::DType, _: f64, _: f64) -> crate::Result<Self::Storage> {
        todo!()
    }

    fn rand_normal(&self, _: &crate::Shape, _: crate::DType, _: f64, _: f64) -> crate::Result<Self::Storage> {
        todo!()
    }

    fn set_seed(&self, _: u64) -> crate::Result<()> {
        todo!()
    }

    fn synchronize(&self) -> crate::Result<()> {
        todo!()
    }
}
