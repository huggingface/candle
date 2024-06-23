use std::{collections::{BTreeMap, HashMap}, fs::{File, OpenOptions}, num::NonZeroU64, sync::{Arc, Mutex}};
use std::io::Write;
use wgpu::{BindGroup, BindGroupLayoutDescriptor};

use crate::WgpuDevice;

use super::device::{BindGroupReference, PipelineType};


#[derive(Debug)]
pub(crate) struct BindgroupLayouts {
    pub bind_group_layout0: wgpu::BindGroupLayout,
    pub bind_group_layout1: wgpu::BindGroupLayout,
    pub bind_group_layout2: wgpu::BindGroupLayout,
    pub bind_group_layout3: wgpu::BindGroupLayout,
    pub pipeline_layout0: wgpu::PipelineLayout,
    pub pipeline_layout1: wgpu::PipelineLayout,
    pub pipeline_layout2: wgpu::PipelineLayout,
    pub pipeline_layout3: wgpu::PipelineLayout,
}

impl BindgroupLayouts {
    pub(crate) fn new(dev: &wgpu::Device) -> Self {
        let dest_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: Some(NonZeroU64::new(4).unwrap()),
            },
            count: None,
        };

        let meta_entry = wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: true,
                min_binding_size: Some(NonZeroU64::new(4).unwrap()),
            },
            count: None,
        };

        let input1_entry = wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: Some(NonZeroU64::new(4).unwrap()),
            },
            count: None,
        };

        let input2_entry = wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: Some(NonZeroU64::new(4).unwrap()),
            },
            count: None,
        };

        let input3_entry = wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: Some(NonZeroU64::new(4).unwrap()),
            },
            count: None,
        };
        let bind_group_layout0 = dev.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[dest_entry, meta_entry],
        });
        let bind_group_layout1 = dev.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[dest_entry, meta_entry, input1_entry],
        });
        let bind_group_layout2 = dev.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[dest_entry, meta_entry, input1_entry, input2_entry],
        });
        let bind_group_layout3 = dev.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                dest_entry,
                meta_entry,
                input1_entry,
                input2_entry,
                input3_entry,
            ],
        });

        let pipeline_layout0 = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout0],
            push_constant_ranges: &[],
        });
        let pipeline_layout1 = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout1],
            push_constant_ranges: &[],
        });
        let pipeline_layout2 = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout2],
            push_constant_ranges: &[],
        });
        let pipeline_layout3 = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout3],
            push_constant_ranges: &[],
        });

        Self {
            bind_group_layout0,
            bind_group_layout1,
            bind_group_layout2,
            bind_group_layout3,
            pipeline_layout0,
            pipeline_layout1,
            pipeline_layout2,
            pipeline_layout3,
        }
    }
}

/// Virtual Buffer, used in Compute Graph
#[derive(Debug)]
pub struct BufferReference{
    pub size : u64,
    pub (crate) id : u32,
    pub (crate) storage : Mutex<Option<Arc<CachedBuffer>>>,
    device : WgpuDevice
}

impl Drop for BufferReference{
    fn drop(&mut self) {
        //println!("BufferReference.drop BufferReference.Storage.lock_start");
        let mut storage = self.storage.lock().unwrap();
        //println!("BufferReference.drop BufferReference.Storage.lock_end");
        if let Some(s) = storage.as_ref(){
            {

                // let mut file = OpenOptions::new()
                // .write(true)
                // .append(true)
                // .create(true)
                // .open("debug-llama2c.txt")
                // .unwrap();
                // writeln!(file, "Buffer Cache {} dropped from {}", s.id, self.id).unwrap();
                //println!("BufferReference.drop  self.device.cache.lock_start");
                let mut cache = self.device.cache.lock().unwrap();

                //println!("Drop BufferReference to Buffer{}", s.id);

                //println!("BufferReference.drop  self.device.cache.lock_end");
                cache.buffers.add_buffer(s.clone());
            }
            *storage = None;
        }
    }
}

impl BufferReference{
    pub (crate) fn new(dev : &WgpuDevice, size : u64) -> Arc<Self>{
        Arc::new(Self{size, storage : Mutex::new(None), id: dev.cached_buffer_reference_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed), device : dev.clone()})
    }

    pub (crate) fn new_init(dev : &WgpuDevice, data : &[u8]) -> Arc<Self>{
        //println!("BufferReference new_init, lock_start");
        let mut cache = dev.cache.lock().unwrap();
        //println!("BufferReference new_init, lock_end");
        let id = dev.cached_buffer_reference_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let buffer = cache.buffers.get_buffer(dev, data.len() as u64, id);
        dev.queue.write_buffer(&buffer.buffer, 0, data);

        Arc::new(Self{size : data.len() as u64, storage : Mutex::new(Some(buffer)), id, device : dev.clone()})
    }
}



fn create_buffer(dev : &WgpuDevice, size : u64) -> wgpu::Buffer{
    dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn create_bindgroup(dev : &WgpuDevice, bindgroup : CachedBindGroupReference, pipeline: Arc<wgpu::ComputePipeline>) -> wgpu::BindGroup{

    // let meta_offset= bindgroup.get_meta();

    // let bind_group_layout = pipeline.get_bind_group_layout(0);    

    // let buffer_meta = &dev.meta_buffer;

    // let meta_binding = wgpu::BufferBinding{ buffer: &buffer_meta, offset: meta_offset  as u64 * 4, size:  None};
    // let meta_bindung = wgpu::BindingResource::Buffer(meta_binding);

    let buffer_meta = &dev.meta_buffer;

    let meta_binding = wgpu::BufferBinding {
        buffer: &buffer_meta,
        offset: 0,
        size: Some(NonZeroU64::new(256).unwrap()),
    };
    let meta_binding = wgpu::BindingResource::Buffer(meta_binding);

    let meta_entry = wgpu::BindGroupEntry {
        binding: 1,
        resource: meta_binding, //buffer_meta.as_entire_binding(),
    };

    let bind_group_layout = match bindgroup {
        BindGroupReferenceBase::Bindgroup0(_,_) => &dev.bindgroup_layouts.bind_group_layout0,
        BindGroupReferenceBase::Bindgroup1(_,_, _) => &dev.bindgroup_layouts.bind_group_layout1,
        BindGroupReferenceBase::Bindgroup2(_,_, _, _) => &dev.bindgroup_layouts.bind_group_layout2,
        BindGroupReferenceBase::Bindgroup3(_,_, _, _, _) => &dev.bindgroup_layouts.bind_group_layout3,
    };

    match bindgroup{
        CachedBindGroupReference::Bindgroup0(_, buffer_dest) => {
            let entries = &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_dest.buffer.as_entire_binding(),
            },
            meta_entry];
            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: entries
            })
        },
        CachedBindGroupReference::Bindgroup1(_, buffer_dest, buffer_input1) => {
            let entries = &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_dest.buffer.as_entire_binding(),
            },
            meta_entry,
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffer_input1.buffer.as_entire_binding(),
            },];
            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: entries
            })
        },
        CachedBindGroupReference::Bindgroup2(_, buffer_dest, buffer_input1, buffer_input2) => {
            let entries = &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_dest.buffer.as_entire_binding(),
            },
            meta_entry,
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffer_input1.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buffer_input2.buffer.as_entire_binding(),
            }];
            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: entries
            })
        },
        CachedBindGroupReference::Bindgroup3(_, buffer_dest, buffer_input1, buffer_input2, buffer_input3) => {
            let entries = &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_dest.buffer.as_entire_binding(),
            },
            meta_entry,
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffer_input1.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buffer_input2.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: buffer_input3.buffer.as_entire_binding(),
            },];
            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: entries
            })
        },
    }
}



#[derive(Debug)]
pub(crate) struct HashMapMulti<K, V>{
    map : HashMap<K, Vec<V>>,
    empty : Vec<V>,
}

#[derive(Debug)]
pub(crate) struct BTreeMulti<K, V>{
    map : BTreeMap<K, Vec<V>>,
    empty : Vec<V>,
}

impl<K: std::cmp::Eq + PartialEq + std::hash::Hash, V : std::cmp::PartialEq> HashMapMulti<K, V>{
    
    pub fn new() -> Self{
        Self{map : HashMap::new(), empty : vec![]}
    } 
    
    pub fn add_mapping(&mut self, key: K, value: V) {
       self.map.entry(key).or_insert_with(Vec::new).push(value);
    }
    
    pub fn remove_mapping(&mut self, key : K, value: &V) {
         if let Some(vec) = self.map.get_mut(&key) {
            if let Some(pos) = vec.iter().position(|x| x == value) {
                vec.remove(pos);
            }
            
            if vec.is_empty() {
                self.map.remove(&key);
            }
        }
    }

    pub fn get(&self, key : &K) -> &Vec<V>{
        self.map.get(key).unwrap_or(&self.empty)
    }
}


impl<K: std::cmp::Eq + PartialEq + std::hash::Hash + Ord, V : std::cmp::PartialEq> BTreeMulti<K, V>{
    
    pub fn new() -> Self{
        Self{map : BTreeMap::new(), empty : vec![]}
    } 
    
    pub fn add_mapping(&mut self, key: K, value: V) {
       let values = self.map.entry(key).or_insert_with(Vec::new);
       if !values.contains(&value){
        values.push(value);
       }
    }
    
    pub fn remove_mapping(&mut self, key : K, value: &V) {
         if let Some(vec) = self.map.get_mut(&key) {
            if let Some(pos) = vec.iter().position(|x| x == value) {
                vec.remove(pos);
            }
            
            if vec.is_empty() {
                self.map.remove(&key);
            }
        }
    }

    pub fn get(&self, key : &K) -> &Vec<V>{
        self.map.get(key).unwrap_or(&self.empty)
    }

    pub fn get_mut(&mut self, key : &K) -> &mut Vec<V>{
        self.map.get_mut(key).unwrap_or(&mut self.empty)
    }
}

/// Cache of all free CachedBuffers
#[derive(Debug)]
pub (crate) struct BufferCache{
    //buffers : Vec<Arc<CachedBuffer>>,
    buffers: BTreeMulti<u64, Arc<CachedBuffer>>,
}

impl BufferCache{
    fn new() -> Self{ 
        return Self{buffers: BTreeMulti::new()}
    }

    fn get_buffer(&mut self, dev : &WgpuDevice, size : u64, buffer_reference_id : u32) -> Arc<CachedBuffer>{
        // let mut file = OpenOptions::new()
        // .write(true)
        // .append(true)
        // .create(true)
        // .open("debug-llama2c.txt")
        // .unwrap();
        
        for (buffer_size, buffers) in self.buffers.map.range_mut(size..){
            if *buffer_size < size{
                panic!("Did not expect size to be smaller, than key");
            }

            if let Some(buffer) = buffers.pop(){
                dev.cached_buffer_reuse_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return buffer;
            }
        }

        let id = dev.cached_buffer_counter.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
        //writeln!(file, "Buffer Create {} for {}", id, buffer_reference_id).unwrap();
        Arc::new(CachedBuffer::new(create_buffer(dev, size),  id))
    }

    fn add_buffer(&mut self, buffer : Arc<CachedBuffer>){
        self.buffers.add_mapping(buffer.buffer.size(), buffer);
    }

    pub (crate) fn load_buffer(&mut self, dev : &WgpuDevice, buffer : Arc<BufferReference>) -> Arc<CachedBuffer>{
        //println!("BufferCache load_buffer, lock_start");
        let mut storage = buffer.storage.lock().unwrap();
        //println!("BufferCache load_buffer, lock_end");
        match storage.as_ref()
        {
            Some(storage) => return storage.clone(),
            None => {
                let buffer = self.get_buffer(dev, buffer.size, buffer.id);
                *storage = Some(buffer.clone());
                return buffer;
            },
        }
    }
    
     //tries to reference the cached buffer from refernece
     fn can_match_buffer(&mut self, cached : &Arc<CachedBuffer>, reference : &Arc<BufferReference>) -> bool{
        let reference_storage = reference.storage.lock().unwrap();
        if let Some(reference) = reference_storage.as_ref(){
            if cached.id == reference.id{ //the cached buffer is alredy used by the reference
                return true;
            }
            else{        //the data reference points to is alredy stored inside another cached
                return false;
            }
        }
        else{ //is the cached buffer free, and is is bit enaugh?

           
            if cached.buffer.size() >= reference.size{
                let buffers = self.buffers.get(&cached.buffer.size());
                if let Some(_) = buffers.iter().position(|c| c == cached)
                {
                    return true;
                }
            }
        }   
        return false;
    }

    //tries to reference the cached buffer from refernece
    fn match_buffer(&mut self, cached : &Arc<CachedBuffer>, reference : &Arc<BufferReference>){
        let mut reference_storage = reference.storage.lock().unwrap();
        if let None = reference_storage.as_ref(){

            let buffers = self.buffers.get_mut(&cached.buffer.size());

            if let Some(index) = buffers.iter().position(|c| c == cached)
            {
               *reference_storage = Some(buffers.remove(index));
            }
        }   
    }

}


/// Cache of all available CachedBindGroups
#[derive(Debug)]
pub (crate) struct BindGroupCache{
    bindgroups : Vec<Arc<CachedBindGroup>>
}



impl BindGroupCache {

    fn new() -> Self{ 
        return Self{bindgroups: vec![]}
    }

    // pub (crate) fn get_bind_group_old(&mut self, dev : &WgpuDevice, buffers : CachedBindGroupReference, pipeline: Arc<wgpu::ComputePipeline>, pipeline_type: PipelineType) -> Arc<CachedBindGroup>{
    //     for bind_group in self.bindgroups.iter(){
    //         if bind_group.buffers == buffers{
    //             println!("Reuse Bindgroup: {}", bind_group.id);
    //             return bind_group.clone();
    //         }
    //     }

    //     //println!("Create Bindgroup: {:?}", pipeline_type);

    //     return Arc::new(CachedBindGroup::new(create_bindgroup(dev, buffers.clone(), pipeline), buffers, dev.cached_bindgroup_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
    // }
}



/// A wgpu Buffer
#[derive(Debug)]
pub (crate) struct CachedBuffer{
    pub (crate) buffer : wgpu::Buffer, //reference to the buffer, if Arc::strong_count is 1, the tensor is free to be reused, as long as it is big enaugh and not other enqued commands depend on that buffer
    
    pub (crate) id : u32
    //bindgroups : Vec<u32> //reference to Bindgroups, if we need to resize this Buffer, we also need to invalidate the referenced Bindgroups
}

impl PartialEq for CachedBuffer{
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for CachedBuffer{

}

impl CachedBuffer {
    fn new(buffer: wgpu::Buffer, id : u32) -> Self {
        //println!("Create_new Buffer size:{}, id:{}", buffer.size(), id);
        Self { buffer,id}
    }
}

#[derive(Debug, Clone)]
pub (crate) enum BindGroupReferenceBase<T>{
    Bindgroup0(u32, T), //dest,
    Bindgroup1(u32, T,T), //dest, input1
    Bindgroup2(u32, T,T,T), //dest, input1, input2
    Bindgroup3(u32, T,T,T,T) //dest, input1, input2, input3
}

impl<T> BindGroupReferenceBase<T> {
    pub(crate) fn get_meta(&self) -> u32{
        match self{
            BindGroupReferenceBase::Bindgroup0(meta, _) => *meta,
            BindGroupReferenceBase::Bindgroup1(meta, _, _) => *meta,
            BindGroupReferenceBase::Bindgroup2(meta, _, _, _) => *meta,
            BindGroupReferenceBase::Bindgroup3(meta, _, _, _, _) => *meta,
        }
    }

    fn get_dest(&self) -> &T{
        match self{
            BindGroupReferenceBase::Bindgroup0(_, dest) => dest,
            BindGroupReferenceBase::Bindgroup1(_, dest, _) => dest,
            BindGroupReferenceBase::Bindgroup2(_, dest, _, _) => dest,
            BindGroupReferenceBase::Bindgroup3(_, dest, _, _, _) => dest,
        }
    }
}

pub (crate) type CachedBindGroupReference = BindGroupReferenceBase<Arc<CachedBuffer>>;

// #[derive(Debug, Clone)]
// pub (crate) enum CachedBindGroupReference{
//     Bindgroup0(u32, Arc<CachedBuffer>), //dest,
//     Bindgroup1(u32, Arc<CachedBuffer>,Arc<CachedBuffer>), //dest, input1
//     Bindgroup2(u32, Arc<CachedBuffer>,Arc<CachedBuffer>,Arc<CachedBuffer>), //dest, input1, input2
//     Bindgroup3(u32, Arc<CachedBuffer>,Arc<CachedBuffer>,Arc<CachedBuffer>,Arc<CachedBuffer>) //dest, input1, input2, input3
// }


impl PartialEq for CachedBindGroupReference {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (CachedBindGroupReference::Bindgroup0(a1, b1), CachedBindGroupReference::Bindgroup0(a2, b2)) => {
                a1 == a2 && Arc::ptr_eq(b1, b2)
            },
            (CachedBindGroupReference::Bindgroup1(a1, b1, c1), CachedBindGroupReference::Bindgroup1(a2, b2, c2)) => {
                a1 == a2 && Arc::ptr_eq(b1, b2) && Arc::ptr_eq(c1, c2)
            },
            (CachedBindGroupReference::Bindgroup2(a1, b1, c1, d1), CachedBindGroupReference::Bindgroup2(a2, b2, c2, d2)) => {
                a1 == a2 && Arc::ptr_eq(b1, b2) && Arc::ptr_eq(c1, c2) && Arc::ptr_eq(d1, d2)
            },
            (CachedBindGroupReference::Bindgroup3(a1, b1, c1, d1, e1), CachedBindGroupReference::Bindgroup3(a2, b2, c2, d2, e2)) => {
                a1 == a2 && Arc::ptr_eq(b1, b2) && Arc::ptr_eq(c1, c2) && Arc::ptr_eq(d1, d2) && Arc::ptr_eq(e1, e2)
            },
            _ => false,
        }
    }
}

impl Eq for CachedBindGroupReference {}

// wgpu Bindgroup
#[derive(Debug)]
pub (crate) struct CachedBindGroup{
    pub(crate) bindgroup : wgpu::BindGroup,
    pipeline : PipelineType,
    last_used : u32, //An Index referencing, when this cached item was used the last time,
    buffers : CachedBindGroupReference,
    id : u32
}

impl CachedBindGroup {
    fn new(bindgroup: wgpu::BindGroup, buffers : CachedBindGroupReference, id : u32, pipeline : PipelineType ) -> Self {
        // match &buffers{
        //     CachedBindGroupReference::Bindgroup0(meta, dest) =>  println!("Create_new Bindgroup0 id:{}, meta:{}, dest:{}", id, meta, dest.id),
        //     CachedBindGroupReference::Bindgroup1(meta, dest, input1) => println!("Create_new Bindgroup1 id:{}, meta:{}, dest:{}, input1:{}", id, meta, dest.id, input1.id),
        //     CachedBindGroupReference::Bindgroup2(meta, dest, input1, input2) => println!("Create_new Bindgroup2 id:{}, meta:{}, dest:{}, input1:{}, input2:{}", id, meta, dest.id, input1.id, input2.id),
        //     CachedBindGroupReference::Bindgroup3(meta, dest, input1, input2, input3) => println!("Create_new Bindgroup3 id:{}, meta:{}, dest:{}, input1:{}, input2:{}, input3:{}", id, meta, dest.id, input1.id, input2.id, input3.id),
        // }
        Self { bindgroup, last_used : 0, buffers, id, pipeline }
    }
}


#[derive(Debug)]
pub struct ModelCache{
    pub (crate) buffers : BufferCache,
    pub (crate) bindgroups : BindGroupCache

    //pub (crate) counter_buffer : u32, 
    //pub (crate) cached_buffer : Vec<Arc<wgpu::Buffer>>, 
    
    //pub (crate) counter_bindgroup : u32, 
    //pub (crate) cached_bindgroup  :Vec<Arc<wgpu::BindGroup>>,
    //pub (crate) cached_pipeline : Vec<Arc<wgpu::ComputePipeline>>,
}

impl ModelCache {
    pub fn new() -> Self {
        Self { buffers : BufferCache::new(), bindgroups : BindGroupCache::new() }
    }

    
    pub (crate) fn get_bind_group(&mut self, dev : &WgpuDevice, bindgroup_reference : &BindGroupReference, pipeline: Arc<wgpu::ComputePipeline>, pipeline_type: PipelineType) -> Arc<CachedBindGroup>{
        // let mut file = OpenOptions::new()
        // .write(true)
        // .append(true)
        // .create(true)
        // .open("debug-llama2c.txt")
        // .unwrap();

        
        // let bindgroup_reference = match bindgroup_reference {
        //     BindGroupReference::Bindgroup0(meta, dest_buffer) => {
        //         // //println!("flush_gpu_command cache lock_start");
        //         // let mut cache = dev.cache.lock().unwrap();
        //         // //println!("flush_gpu_command cache lock_end");
        //         CachedBindGroupReference::Bindgroup0(
        //             *meta,
        //             cache.buffers.load_buffer(dev, dest_buffer.clone()),
        //         )
        //     }
        //     BindGroupReference::Bindgroup1(meta, dest_buffer, input1) => {
        //         // //println!("flush_gpu_command cache lock_start");
        //         // let mut cache = dev.cache.lock().unwrap();
        //         // //println!("flush_gpu_command cache lock_end");
        //         CachedBindGroupReference::Bindgroup1(
        //             *meta,
        //             cache.buffers.load_buffer(dev, dest_buffer.clone()),
        //             cache.buffers.load_buffer(dev, input1.clone()),
        //         )
        //     }
        //     BindGroupReference::Bindgroup2(
        //         meta,
        //         dest_buffer,
        //         input1,
        //         input2,
        //     ) => {
        //         // //println!("flush_gpu_command cache lock_start");
        //         // let mut cache = dev.cache.lock().unwrap();
        //         // //println!("flush_gpu_command cache lock_end");
        //         CachedBindGroupReference::Bindgroup2(
        //         *meta,
        //         cache.buffers.load_buffer(dev, dest_buffer.clone()),
        //         cache.buffers.load_buffer(dev, input1.clone()),
        //         cache.buffers.load_buffer(dev, input2.clone()),
        //     )},
        //     BindGroupReference::Bindgroup3(
        //         meta,
        //         dest_buffer,
        //         input1,
        //         input2,
        //         input3,
        //     ) => {
        //         CachedBindGroupReference::Bindgroup3(
        //         *meta,
        //         cache.buffers.load_buffer(dev, dest_buffer.clone()),
        //         cache.buffers.load_buffer(dev, input1.clone()),
        //         cache.buffers.load_buffer(dev, input2.clone()),
        //         cache.buffers.load_buffer(dev, input3.clone()),
        //     )},
        // };
        for cached_bindgroup in self.bindgroups.bindgroups.iter(){
            if cached_bindgroup.buffers.get_meta() == bindgroup_reference.get_meta() && cached_bindgroup.pipeline == pipeline_type{
                //possible candiate, check if buffers match:

                match (&cached_bindgroup.buffers, bindgroup_reference){
                    (BindGroupReferenceBase::Bindgroup0(_, cbuf1), BindGroupReferenceBase::Bindgroup0(_, buf1)) => {
                        if !self.buffers.can_match_buffer(cbuf1, buf1){
                            continue;
                        }
                        self.buffers.match_buffer(cbuf1, buf1);
                    },
                    (BindGroupReferenceBase::Bindgroup1(_, cbuf1,cbuf2), BindGroupReferenceBase::Bindgroup1(_, buf1,buf2)) => {
                        if !(self.buffers.can_match_buffer(cbuf1, buf1) && self.buffers.can_match_buffer(cbuf2, buf2)){
                            continue;
                        }
                        self.buffers.match_buffer(cbuf1, buf1);
                        self.buffers.match_buffer(cbuf2, buf2);
                    },
                    (BindGroupReferenceBase::Bindgroup2(_, cbuf1, cbuf2, cbuf3), BindGroupReferenceBase::Bindgroup2(_, buf1,buf2, buf3)) => {
                        if !(self.buffers.can_match_buffer(cbuf1, buf1) && self.buffers.can_match_buffer(cbuf2, buf2) && self.buffers.can_match_buffer(cbuf3, buf3)){
                            continue;
                        }
                        self.buffers.match_buffer(cbuf1, buf1);
                        self.buffers.match_buffer(cbuf2, buf2);
                        self.buffers.match_buffer(cbuf3, buf3);
                    },
                    (BindGroupReferenceBase::Bindgroup3(_, cbuf1,cbuf2, cbuf3, cbuf4), BindGroupReferenceBase::Bindgroup3(_, buf1, buf2, buf3, buf4)) =>{
                        if !(self.buffers.can_match_buffer(cbuf1, buf1) && self.buffers.can_match_buffer(cbuf2, buf2) && self.buffers.can_match_buffer(cbuf3, buf3) && self.buffers.can_match_buffer(cbuf4, buf4)){
                            continue;
                        }
                        self.buffers.match_buffer(cbuf1, buf1);
                        self.buffers.match_buffer(cbuf2, buf2);
                        self.buffers.match_buffer(cbuf3, buf3);
                        self.buffers.match_buffer(cbuf4, buf4);                    },
                    _ => {continue;}
                }
                dev.cached_bindgroup_reuse_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                //writeln!(file, "Bindgroup Reuse Cached: {}, reference_dest id: {}, pipeline: {:?}", cached_bindgroup.id, bindgroup_reference.get_dest().id, pipeline_type).unwrap();
                //println!("Reuse Bindgroup: {}", cached_bindgroup.id);
                return cached_bindgroup.clone();
            }
        }

        //println!("Create Bindgroup: {:?}", pipeline_type);

        //create complete new Bindgroup, use first matching buffer
        let bindgroup_reference = match &bindgroup_reference {
            BindGroupReference::Bindgroup0(meta, dest_buffer) => {
                // //println!("flush_gpu_command cache lock_start");
                // let mut cache = dev.self.buffers.lock().unwrap();
                // //println!("flush_gpu_command cache lock_end");
                CachedBindGroupReference::Bindgroup0(
                    *meta,
                    self.buffers.load_buffer(dev, dest_buffer.clone()),
                )
            }
            BindGroupReference::Bindgroup1(meta, dest_buffer, input1) => {
                // //println!("flush_gpu_command cache lock_start");
                // let mut cache = dev.self.buffers.lock().unwrap();
                // //println!("flush_gpu_command cache lock_end");
                CachedBindGroupReference::Bindgroup1(
                    *meta,
                    self.buffers.load_buffer(dev, dest_buffer.clone()),
                    self.buffers.load_buffer(dev, input1.clone()),
                )
            }
            BindGroupReference::Bindgroup2(
                meta,
                dest_buffer,
                input1,
                input2,
            ) => {
                // //println!("flush_gpu_command cache lock_start");
                // let mut cache = dev.self.buffers.lock().unwrap();
                // //println!("flush_gpu_command cache lock_end");
                CachedBindGroupReference::Bindgroup2(
                *meta,
                self.buffers.load_buffer(dev, dest_buffer.clone()),
                self.buffers.load_buffer(dev, input1.clone()),
                self.buffers.load_buffer(dev, input2.clone()),
            )},
            BindGroupReference::Bindgroup3(
                meta,
                dest_buffer,
                input1,
                input2,
                input3,
            ) => {
                CachedBindGroupReference::Bindgroup3(
                *meta,
                self.buffers.load_buffer(dev, dest_buffer.clone()),
                self.buffers.load_buffer(dev, input1.clone()),
                self.buffers.load_buffer(dev, input2.clone()),
                self.buffers.load_buffer(dev, input3.clone()),
            )},
        };
        let id = dev.cached_bindgroup_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        //writeln!(file, "Bindgroup Create Cached: {}, reference_dest id: {}, pipeline: {:?}", id, bindgroup_reference.get_dest().id, pipeline_type).unwrap();
        let bindgroup = Arc::new(CachedBindGroup::new(create_bindgroup(dev, bindgroup_reference.clone(), pipeline), bindgroup_reference, id, pipeline_type));
        self.bindgroups.bindgroups.push(bindgroup.clone());
        return bindgroup;
    }

}



// impl ModelCache {
//     pub fn start(device : &WgpuDevice) {
        
//         let mut device_cache = device.cache.lock().unwrap();

//         if device_cache.is_none(){
//             println!("Start Cache_Create New");
//             let cache =  Self { 
//                 counter_buffer:0, 
//                 cached_buffer: vec![], 
//                 counter_bindgroup: 0, 
//                 cached_bindgroup: vec![],
//                 cached_pipeline: vec![], };
//             *device_cache = Some(cache);
//         }

//         if let Some(cache) = device_cache.as_mut(){
//             cache.counter_buffer = 0;
//             cache.counter_bindgroup = 0;
//         }
//     }

//     pub fn finish(device : &WgpuDevice){
//         println!("End Cache");
//         let mut device_cache = device.cache.lock().unwrap();
//         *device_cache = None;
//     }
// }

//device, iteration: as models may compute caches at the first iteration, one might want to start caching from the second iteration
pub fn start_cache(device : &crate::Device, value : u32){
    match device{
        crate::Device::Cpu => {},
        crate::Device::Cuda(_) => {},
        crate::Device::Metal(_) => {},
        crate::Device::WebGpu(device) => {
            // let mut file = OpenOptions::new()
            // .write(true)
            // .append(true)
            // .create(true)
            // .open("debug-llama2c.txt")
            // .unwrap();
            //writeln!(file, "START CACHE").unwrap();
            device.cached_buffer_reference_counter.store(value, std::sync::atomic::Ordering::Relaxed);
        },
    }
}

pub fn get_reference_cache(device : &crate::Device) -> u32{
    match device{
        crate::Device::Cpu => {0},
        crate::Device::Cuda(_) => {0},
        crate::Device::Metal(_) => {0},
        crate::Device::WebGpu(device) => {
            device.cached_buffer_reference_counter.load(std::sync::atomic::Ordering::Relaxed)
        },
    }
}