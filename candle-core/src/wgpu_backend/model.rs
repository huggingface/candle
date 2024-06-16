//use std::{collections::HashMap, sync::{Arc, Mutex}};


//use crate::{wgpu::wgpu_functions::flush_gpu_command, Tensor, Var, WgpuDevice};

//use super::device::QueueBuffer;


// #[derive(Debug)]
// pub struct Model{
//     pub (crate) input_buffer : Var,
//     pub (crate) output_buffer : Tensor,
//     cache : Arc<ModelCache>
// }

// impl Model {
//     pub fn new<F : Fn(&Tensor) -> Tensor >(device : &WgpuDevice, input : &Tensor,f : F) -> Self {
        
//         let cache = ModelCache::new(device);

//         let input_buffer = Var::from_tensor(input).unwrap();

//         let output = f(input_buffer.as_tensor()); //we calculate with this tensor

//         return Model{cache, input_buffer, output_buffer : output};
//     }

//     // pub fn forward(&self, dev : &WgpuDevice, input : &Tensor) -> &Tensor{
        
//     //     //1. copy 
//     //     self.input_buffer.set(input).unwrap();
        
//     //     //2. repeat recorded commands
//     //     let mut queue  = self.cache.queue.lock().unwrap();
//     //     for queue in queue.iter_mut(){
//     //         flush_gpu_command(dev, queue);
//     //     }

//     //     //3. return output
//     //     return &self.output_buffer;
//     // }
// }

use std::sync::Arc;

use crate::WgpuDevice;

use super::{device::Pipelines, wgpu_functions::Shader};


struct BufferType(Shader, Pipelines);

struct CachedBuffer{
    buffer : Arc<wgpu::Buffer>, //reference to the buffer, if Arc::strong_count is 1, the tensor is free to be reused, as long as it is big enaugh and not other enqued commands depend on that buffer
    id : u32, 
    buffer_type : BufferType,
    bindgroups : Vec<u32> //reference to Bindgroups, if we need to resize this Buffer, we also need to invalidate the referenced Bindgroups
}

#[derive(Debug)]
pub struct ModelCache{
    pub (crate) counter_buffer : u32, 
    pub (crate) cached_buffer : Vec<Arc<wgpu::Buffer>>, 
    
    pub (crate) counter_bindgroup : u32, 
    pub (crate) cached_bindgroup  :Vec<Arc<wgpu::BindGroup>>,
    
    pub (crate) cached_pipeline : Vec<Arc<wgpu::ComputePipeline>>,

}

impl ModelCache {
    pub fn start(device : &WgpuDevice) {
        
        let mut device_cache = device.cache.lock().unwrap();

        if device_cache.is_none(){
            println!("Start Cache_Create New");
            let cache =  Self { 
                counter_buffer:0, 
                cached_buffer: vec![], 
                counter_bindgroup: 0, 
                cached_bindgroup: vec![],
                cached_pipeline: vec![], };
            *device_cache = Some(cache);
        }

        if let Some(cache) = device_cache.as_mut(){
            cache.counter_buffer = 0;
            cache.counter_bindgroup = 0;
        }
    }

    pub fn finish(device : &WgpuDevice){
        println!("End Cache");
        let mut device_cache = device.cache.lock().unwrap();
        *device_cache = None;
    }
}

//device, iteration: as models may compute caches at the first iteration, one might want to start caching from the second iteration
pub fn start_cache(device : &crate::Device, iteration_start : u32){
    match device{
        crate::Device::Cpu => {},
        crate::Device::Cuda(_) => {},
        crate::Device::Metal(_) => {},
        crate::Device::WebGpu(device) => {
            let mut cache_counter = device.cache_counter.lock().unwrap();
            if *cache_counter > iteration_start{
                ModelCache::start(device)
            }
            *cache_counter += 1;
        },
    }
}

pub fn stop_cache(device : &crate::Device){
    match device{
        crate::Device::Cpu => {},
        crate::Device::Cuda(_) => {},
        crate::Device::Metal(_) => {},
        crate::Device::WebGpu(device) => ModelCache::finish(device),
    }
}