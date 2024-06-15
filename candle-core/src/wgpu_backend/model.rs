use std::sync::{Arc, Mutex};


use crate::{wgpu::wgpu_functions::flush_gpu_command, Tensor, Var, WgpuDevice};

use super::device::QueueBuffer;


#[derive(Debug)]
pub struct Model{
    pub (crate) input_buffer : Var,
    pub (crate) output_buffer : Tensor,
    cache : Arc<ModelCache>
}

impl Model {
    pub fn new<F : Fn(&Tensor) -> Tensor >(device : &WgpuDevice, input : &Tensor,f : F) -> Self {
        
        let cache = ModelCache::new(device);

        let input_buffer = Var::from_tensor(input).unwrap();

        let output = f(input_buffer.as_tensor()); //we calculate with this tensor

        return Model{cache, input_buffer, output_buffer : output};
    }

    pub fn forward(&self, dev : &WgpuDevice, input : &Tensor) -> &Tensor{
        
        //1. copy 
        self.input_buffer.set(input).unwrap();
        
        //2. repeat recorded commands
        let mut queue  = self.cache.queue.lock().unwrap();
        for queue in queue.iter_mut(){
            flush_gpu_command(dev, queue);
        }

        //3. return output
        return &self.output_buffer;
    }
}

#[derive(Debug)]
pub struct ModelCache{
    pub (crate) queue : Mutex<Vec<QueueBuffer>>,

}

impl ModelCache {
    pub fn new(device : &WgpuDevice) -> Arc<Self> {
        let cache =  Arc::new(Self { queue : Mutex::new(vec![])});
        let mut device_cache = device.cache.lock().unwrap();
        *device_cache = Some(cache.clone());
        return cache;
    }


    pub fn finish(&self, device : &WgpuDevice){
        let mut device_cache = device.cache.lock().unwrap();
        *device_cache = None;
    }
}