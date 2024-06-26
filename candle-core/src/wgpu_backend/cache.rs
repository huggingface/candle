use std::{num::NonZeroU64, sync::{atomic::AtomicU32, Arc, Mutex}};
use wgpu::BindGroupLayoutDescriptor;

use crate::WgpuDevice;

use super::{device::BindGroupReference, util::{BTreeMulti, HashMapMulti, ToU64}, wgpu_functions};

pub type BufferId = Arc<CachedBuffer>;
pub type BindgroupId = Arc<CachedBindGroup>;
pub type BufferReferenceId = Arc<BufferReference>;

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
    pub (crate) storage : Mutex<Option<BufferId>>,
    device : WgpuDevice
}

impl Drop for BufferReference{
    fn drop(&mut self) {
        let mut storage = self.storage.lock().unwrap();
        if let Some(s) = storage.as_ref(){
            {
                if self.device.use_cache{
                    let mut cache = self.device.cache.lock().unwrap();
                    cache.buffers.add_buffer(s.clone());
                }
            }
            *storage = None;
        }
    }
}

impl BufferReference{
    pub fn new<T : ToU64>(dev : &WgpuDevice, size : T) -> Arc<Self>{
        Arc::new(Self{size : size.to_u64(), storage : Mutex::new(None), device : dev.clone()})
    }

    pub (crate) fn new_init(dev : &WgpuDevice, data : &[u8]) -> Arc<Self>{
        let mut cache = dev.cache.lock().unwrap();
        let buffer = cache.buffers.get_buffer(dev, data.len() as u64);
        dev.queue.write_buffer(&buffer.buffer, 0, data);

        Arc::new(Self{size : data.len() as u64, storage : Mutex::new(Some(buffer)), device : dev.clone()})
    }
}




/// Cache of all free CachedBuffers
#[derive(Debug)]
pub (crate) struct BufferCache{
    buffers: BTreeMulti<u64, BufferId>,
}

impl BufferCache{
    fn new() -> Self{ 
        return Self{buffers: BTreeMulti::new()}
    }

    pub (crate) fn remove_unused(&mut self){
        self.buffers.map.retain(|_, buffers|{
            buffers.retain(|b| Arc::strong_count(b) != 1);
            return  buffers.len() > 1;
        });
    }

    // pub (crate) fn create_buffer_if_needed(&mut self, dev : &WgpuDevice, size : u64){
    //     for (buffer_size, buffers) in self.buffers.map.range_mut(size..){
    //         if *buffer_size < size{
    //             panic!("Did not expect size to be smaller, than key");
    //         }

    //         if buffers.len() > 0{
    //             return;
    //         }
    //     }

    //     let id = dev.cached_buffer_counter.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
    //     let buffer = Arc::new(CachedBuffer::new(wgpu_functions::create_buffer(dev, size),  id));
    //     self.add_buffer(buffer);
    // }


    fn get_buffer(&mut self, dev : &WgpuDevice, size : u64) -> BufferId{
        if dev.use_cache{
            for (buffer_size, buffers) in self.buffers.map.range_mut(size..){
                if *buffer_size < size{
                    panic!("Did not expect size to be smaller, than key");
                }
    
                if let Some(buffer) = buffers.pop(){
                    dev.cached_buffer_reuse_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return buffer;
                }
            }
        }
        
        let id = dev.cached_buffer_counter.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
        Arc::new(CachedBuffer::new(wgpu_functions::create_buffer(dev, size),  id))
    }

    fn add_buffer(&mut self, buffer : BufferId){
        self.buffers.add_mapping(buffer.buffer.size(), buffer);
    }

    pub (crate) fn load_buffer(&mut self, dev : &WgpuDevice, buffer : BufferReferenceId) -> BufferId{
        let mut storage = buffer.storage.lock().unwrap();
        match storage.as_ref()
        {
            Some(storage) => return storage.clone(),
            None => {
                let buffer = self.get_buffer(dev, buffer.size);
                *storage = Some(buffer.clone());
                return buffer;
            },
        }
    }
    
     //tries to reference the cached buffer from refernece
     fn can_match_buffer(&mut self, cached : &BufferId, reference : &BufferReferenceId) -> bool{
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
    fn match_buffer(&mut self, cached : &BufferId, reference : &BufferReferenceId){
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



#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BindGroupInput {
    Bindgroup0,                               //
    Bindgroup1(BufferId),                     //input1
    Bindgroup2(BufferId, BufferId),           //input1, input2
    Bindgroup3(BufferId, BufferId, BufferId), //input1, input2, input3
}

impl From<BindGroupReferenceBase<BufferId>> for BindGroupInput {
    fn from(value: BindGroupReferenceBase<BufferId>) -> Self {
        match value {
            BindGroupReferenceBase::Bindgroup0(_,_) => BindGroupInput::Bindgroup0,
            BindGroupReferenceBase::Bindgroup1(_,_, v1) => BindGroupInput::Bindgroup1(v1),
            BindGroupReferenceBase::Bindgroup2(_,_, v1, v2) => BindGroupInput::Bindgroup2(v1, v2),
            BindGroupReferenceBase::Bindgroup3(_,_, v1, v2, v3) => {
                BindGroupInput::Bindgroup3(v1, v2, v3)
            }
        }
    }
}

impl std::hash::Hash for BindGroupInput {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            BindGroupInput::Bindgroup0 => {
                state.write_u8(0);
            }
            BindGroupInput::Bindgroup1(id1) => {
                state.write_u8(1);
                Arc::as_ptr(id1).hash(state);
            }
            BindGroupInput::Bindgroup2(id1, id2) => {
                state.write_u8(2);
                Arc::as_ptr(id1).hash(state);
                Arc::as_ptr(id2).hash(state);
            }
            BindGroupInput::Bindgroup3(id1, id2, id3) => {
                state.write_u8(3);
                Arc::as_ptr(id1).hash(state);
                Arc::as_ptr(id2).hash(state);
                Arc::as_ptr(id3).hash(state);
            }
        }
    }
}

/// Cache of all available CachedBindGroups
#[derive(Debug)]
pub (crate) struct BindGroupCache{
    pub(crate) bindgroups: HashMapMulti<BindGroupInput, BindgroupId>,
}



impl BindGroupCache {
    fn new() -> Self{ 
        return Self{bindgroups: HashMapMulti::new()}
    }

    pub (crate) fn remove_unused(&mut self, counter : u32){

        self.bindgroups.map.retain(|_, bindgroups|{

            bindgroups.retain(|b| 
                {
                    let is_bindgroup_used = b.last_used.load(std::sync::atomic::Ordering::Relaxed) > counter;

                    if Arc::strong_count(b) != 1{
                        panic!("Expected to have a strong Count to this CachedBindGroup");
                    }
                    return is_bindgroup_used;
                });
            
            return bindgroups.len() > 0;
        });
        // for (_, bindgroups) in self.bindgroups.map.iter_mut(){
        //     bindgroups.retain(|b| 
        //         {
        //             let is_bindgroup_used = b.last_used.load(std::sync::atomic::Ordering::Relaxed) > counter;

        //             if Arc::strong_count(b) != 1{
        //                 panic!("Expected to have a strong Count to this CachedBindGroup");
        //             }

        //             // if !is_bindgroup_used{
        //             //     println!("Bindgroup is not used -> Remove");
        //             // }
        //             return is_bindgroup_used;
        //         }
        //        );
        // }
    }
}

/// A wgpu Buffer
#[derive(Debug)]
pub struct CachedBuffer{
    pub (crate) buffer : wgpu::Buffer, //reference to the buffer, if Arc::strong_count is 1, the tensor is free to be reused, as long as it is big enaugh and not other enqued commands depend on that buffer
    
    pub (crate) id : u32
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
        Self { buffer,id}
    }
}

#[derive(Debug, Clone)]
pub enum BindGroupReferenceBase<T>{
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

    pub (crate) fn get_dest(&self) -> &T{
        match self{
            BindGroupReferenceBase::Bindgroup0(_, dest) => dest,
            BindGroupReferenceBase::Bindgroup1(_, dest, _) => dest,
            BindGroupReferenceBase::Bindgroup2(_, dest, _, _) => dest,
            BindGroupReferenceBase::Bindgroup3(_, dest, _, _, _) => dest,
        }
    }
}

pub  type CachedBindGroupReference = BindGroupReferenceBase<BufferId>;

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
pub struct CachedBindGroup{
    pub(crate) bindgroup : wgpu::BindGroup,
    last_used : AtomicU32, //An Index referencing, when this cached item was used the last time,
    buffers : CachedBindGroupReference,
    //id : u32
}

impl CachedBindGroup {
    fn new(bindgroup: wgpu::BindGroup, buffers : CachedBindGroupReference ) -> Self {
        Self { bindgroup, last_used : AtomicU32::new(0), buffers}
    }
}


#[derive(Debug)]
pub struct ModelCache{
    pub (crate) buffers : BufferCache,
    pub (crate) bindgroups : BindGroupCache
}

impl ModelCache {
    pub fn new() -> Self {
        Self { buffers : BufferCache::new(), bindgroups : BindGroupCache::new() }
    }

    
    pub (crate) fn get_bind_group(&mut self, dev : &WgpuDevice, bindgroup_reference : &BindGroupReference) -> BindgroupId{
        if dev.use_cache{
            fn get_storage(v : &BufferReferenceId) -> BufferId{
                v.storage.lock().unwrap().as_ref().unwrap().clone()
            }
    
            let bindgroup_reference_buffers = bindgroup_reference;
            let bindgroup_inputs = match bindgroup_reference_buffers {
                BindGroupReferenceBase::Bindgroup0(_,_) => BindGroupInput::Bindgroup0,
                BindGroupReferenceBase::Bindgroup1(_,_, v1) => {
                    BindGroupInput::Bindgroup1(get_storage(v1))
                }
                BindGroupReferenceBase::Bindgroup2(_,_, v1, v2) => {
                    BindGroupInput::Bindgroup2(get_storage(v1), get_storage(v2))
                }
                BindGroupReferenceBase::Bindgroup3(_,_, v1, v2, v3) => {
                    BindGroupInput::Bindgroup3(get_storage(v1), get_storage(v2), get_storage(v3))
                }
            };
            let buf_dest = bindgroup_reference_buffers.get_dest();
    
            let candidates_to_process = self.bindgroups.bindgroups.get(&bindgroup_inputs)
            .iter()
            .filter(|id| {
                let cbuf_dest = id.buffers.get_dest();
                return self.buffers.can_match_buffer(cbuf_dest, buf_dest);
            });
    
            let mut candidate_to_process = None;
            let mut best_size = u64::MAX;
    
            for ele in candidates_to_process{
                
                let cbuf_dest = ele.buffers.get_dest();
                let buffer_size = cbuf_dest.buffer.size();
                if buffer_size < best_size{
                    candidate_to_process = Some(ele);
                    best_size = buffer_size;
                }
            }
    
            if let Some(cached_bindgroup) = candidate_to_process {
                let cbuf1 = cached_bindgroup.buffers.get_dest();
                self.buffers.match_buffer(cbuf1, buf_dest);
    
                dev.cached_bindgroup_reuse_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                cached_bindgroup.last_used.store(dev.cached_bindgroup_use_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed), std::sync::atomic::Ordering::Relaxed);
                return cached_bindgroup.clone();
            }
        }
        
        //create complete new Bindgroup, use first matching buffer
        let bindgroup_reference = match &bindgroup_reference {
            BindGroupReference::Bindgroup0(meta, dest_buffer) => {
                CachedBindGroupReference::Bindgroup0(
                    *meta,
                    self.buffers.load_buffer(dev, dest_buffer.clone()),
                )
            }
            BindGroupReference::Bindgroup1(meta, dest_buffer, input1) => {
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
        let bindgroup = Arc::new(CachedBindGroup::new(wgpu_functions::create_bindgroup(dev, bindgroup_reference.clone()), bindgroup_reference));
        
        if dev.use_cache{
            self.bindgroups.bindgroups.add_mapping(bindgroup.buffers.clone().into(), bindgroup.clone());
            bindgroup.last_used.store(dev.cached_bindgroup_use_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed), std::sync::atomic::Ordering::Relaxed);
        }
        return bindgroup;
    }

}