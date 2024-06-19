// use std::{fs::{File, OpenOptions}, ops::Index, sync::{Arc, Mutex}};
// use std::io::Write;
// use wgpu::BindGroup;

use std::{ops::{Index, IndexMut}, sync::Arc};

use crate::WgpuDevice;

use super::device::PipelineType;

//use super::device::{BindGroupReference, PipelineType};


//pub type BufferReferenceId = u32;
//pub type BufferId = u32;
//pub type BindgroupReferenceId = u32;
//pub type BindgroupId = u32;

macro_rules! impl_id {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct $name(u32);

        impl From<u32> for $name {
            fn from(value: u32) -> Self {
                $name(value)
            }
        }

        impl From<$name> for u32 {
            fn from(id: $name) -> Self {
                id.0
            }
        }

       

    };
}
impl_id!(BufferId);
impl_id!(BufferReferenceId);
impl_id!(BindgroupId);
impl_id!(BindgroupReferenceId);

impl BufferId{
    pub fn get(self, cache : &ModelCache) -> &wgpu::Buffer{
        return &cache.buffers[self].buffer;
    }
}

impl BufferReferenceId{
    pub fn get(self, cache : &ModelCache) -> &BufferReference{
        return &cache.buffer_reference[self];
    }

    pub fn get_mut(self, cache : &mut ModelCache) -> &mut BufferReference{
        return &mut cache.buffer_reference[self];
    }
}

impl BindgroupId{
    pub fn get(self, cache : &ModelCache) -> &BindgroupCache{
        return &cache.bindgroups[self];
    }
}

impl BindgroupReferenceId{
    pub fn get(self, cache : &ModelCache) -> &BindGroupReference{
        return &cache.bindgroups_reference[self];
    }
}

#[derive(Debug)]
pub struct VecMap<KEY, VALUE>{
    data : Vec<VALUE>, 
    free_indexes : Vec<KEY>
}

impl<KEY : Into<u32> + From<u32> + Copy, VALUE> VecMap<KEY, VALUE> {
    pub fn new() -> Self {
        Self { data : vec![], free_indexes : vec![] }
    }

    pub fn insert(&mut self, value : VALUE) -> KEY{        
        if let Some(index) = self.free_indexes.pop(){
            self.data[index.into() as usize] = value;
            return index;
        }
        else{
            self.data.push(value);
            return ((self.data.len() - 1) as u32).into();
        }
    }

    pub fn free(&mut self, key : KEY){
        self.free_indexes.push(key);
    }

    pub fn get(&self, key : KEY) -> &VALUE{
        return &self.data[key.into() as usize];
    }

    pub fn get_mut(&mut self, key : KEY) -> &mut VALUE{
        return &mut self.data[key.into() as usize];
    }

    pub fn iter(&self) -> impl Iterator<Item=(KEY, &VALUE)> + '_ {
       self.data.iter().enumerate().map(|(index, v)| ((index as u32).into(), v))
    }

}

impl<KEY : Into<u32> + From<u32> + Copy, VALUE> Index<KEY> for VecMap<KEY, VALUE>{
    type Output = VALUE;

    fn index(&self, index: KEY) -> &Self::Output {
        return self.get(index);
    }
} 

impl<KEY : Into<u32> + From<u32> + Copy, VALUE> IndexMut<KEY> for VecMap<KEY, VALUE>{
    fn index_mut(&mut self, index: KEY) -> &mut Self::Output {
        return self.get_mut(index);
    }
} 




#[derive(Debug)]
pub struct ModelCache{
    pub (crate) bindgroups_reference : VecMap<BindgroupReferenceId, BindGroupReference>,
    pub (crate) buffer_reference : VecMap<BufferReferenceId, BufferReference>,
    pub (crate) buffers : VecMap<BufferId, BufferCache>,
    pub (crate) bindgroups : VecMap<BindgroupId, BindgroupCache>,
}

#[derive(Debug)]
pub (crate) struct BufferReference{
    pub size : u64,
    pub(crate) referenced : bool, //if this buffer is referenced by a Tensor
    pub(crate) cache : Option<BufferId>,
    pub bindgroups : Vec<BindgroupReferenceId>
}

impl BufferReference {
    pub (crate) fn new(size: u64) -> Self {
        Self { size, referenced : true, cache : None, bindgroups : vec![] }
    }

    pub (crate) fn new_init(size: u64, cache : BufferId) -> Self{
        Self { size, referenced : true, cache : Some(cache), bindgroups : vec![] }
    }
}
#[derive(Debug)]
pub struct BufferCache{
    pub(crate) buffer : wgpu::Buffer,
    used : bool, //is this buffer currently used by a buffer_reference
}

impl BufferCache {
    fn new(buffer: wgpu::Buffer) -> Self {
        Self { buffer, used:false }
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
    fn get_meta(&self) -> u32{
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

#[derive(Debug)]
pub (crate) struct BindGroupReference{
    pub cache : Option<BindgroupId>,
    pub buffers : BindGroupReferenceBase<BufferReferenceId>,
    pub pipeline : PipelineType
}

impl BindGroupReference {
    pub (crate) fn new(buffers: BindGroupReferenceBase<BufferReferenceId>, pipeline : PipelineType) -> Self {
        Self { cache : None, buffers, pipeline }
    }
}

#[derive(Debug)]
pub struct BindgroupCache{
    pub(crate) bindgroup: wgpu::BindGroup, 
    pub(crate) buffers : BindGroupReferenceBase<BufferId>,
    pub pipeline : PipelineType
}

impl BindgroupCache {
    fn new(bindgroup: wgpu::BindGroup, buffers: BindGroupReferenceBase<BufferId>, pipeline : PipelineType) -> Self {
        Self { bindgroup, buffers, pipeline}
    }
}


impl ModelCache {
    pub fn new() -> Self {
        Self { bindgroups_reference: VecMap::new(), buffer_reference: VecMap::new(), buffers: VecMap::new(), bindgroups: VecMap::new()  }
    }

    pub fn set_buffer_reference_cache(&mut self, reference : BufferReferenceId, buffer : BufferId)
    {
        self.buffer_reference[reference].cache = Some(buffer);
        self.buffers[buffer].used = true;
    }

    pub fn release_buffer_reference_cache(&mut self, reference : BufferReferenceId){
        if let Some(buffer) = self.buffer_reference[reference].cache {
            self.buffers[buffer].used = false;
        }
        self.buffer_reference[reference].cache = None;
    }

    pub fn get_free_buffer(&mut self, dev : &WgpuDevice, buffer_reference : BufferReferenceId) -> BufferId{
        let size = self.buffer_reference[buffer_reference].size;

        let mut buffer_index = None;

        for (index, buffer) in self.buffers.iter(){
            if !buffer.used && buffer.buffer.size() >= size{
                buffer_index = Some(index);
                break;
            }
        }

        if let Some(index) = buffer_index{
            self.set_buffer_reference_cache(buffer_reference, index);
            dev.cached_buffer_reuse_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return index;
        }

        dev.cached_buffer_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let key = self.buffers.insert(BufferCache::new(create_buffer(dev, size)));
        self.set_buffer_reference_cache(buffer_reference, key);
        return key;
    }

    pub (crate) fn load_buffer(&mut self, dev : &WgpuDevice, buffer_reference : BufferReferenceId) -> BufferId{
        match self.buffer_reference[buffer_reference].cache
        {
            Some(storage) => return storage,
            None => {
                self.get_free_buffer(dev, buffer_reference)
            },
        }
    }
    
    //tries to reference the cached buffer from refernece
    pub fn can_match_buffer(&self, cached : BufferId, reference : BufferReferenceId) -> bool{
        if let Some(buffer) = self.buffer_reference[reference].cache{
            if cached == buffer{ //the cached buffer is alredy used by the reference
                return true;
            }
            else{        //the data reference points to is alredy stored inside another cached
                return false;
            }
        }
        else{ //is the cached buffer free, and is is big enaugh?
            return !self.buffers[cached].used && self.buffers[cached].buffer.size() >= self.buffer_reference[reference].size;
        }   
    }

    //tries to reference the cached buffer from refernece
    pub fn match_buffer(&mut self, cached : BufferId, reference : BufferReferenceId){
        if let None = self.buffer_reference[reference].cache{
            self.set_buffer_reference_cache(reference,cached);
        }   
    }


    pub (crate) fn get_bind_group(&mut self, dev : &WgpuDevice, bindgroup_reference : BindgroupReferenceId, pipeline: Arc<wgpu::ComputePipeline>) -> BindgroupId{
        let bindgroup_reference_meta = self.bindgroups_reference[bindgroup_reference].buffers.get_meta();
        let bindgroup_reference_buffers = self.bindgroups_reference[bindgroup_reference].buffers.clone();
        let pipeline_type : PipelineType = self.bindgroups_reference[bindgroup_reference].pipeline.clone();
        
        let mut candidates_to_process = Vec::new();
        {
            for (cached_id, cached_bindgroup) in self.bindgroups.iter(){
                if cached_bindgroup.buffers.get_meta() == bindgroup_reference_meta && cached_bindgroup.pipeline == pipeline_type{
                    candidates_to_process.push((cached_id, cached_bindgroup.buffers.clone()));
                }
            }
        }
        

        for (cached_id, cached_bindgroup) in candidates_to_process.iter(){
                //possible candiate, check if buffers match:
                match (&cached_bindgroup, &bindgroup_reference_buffers){
                    (BindGroupReferenceBase::Bindgroup0(_, cbuf1), BindGroupReferenceBase::Bindgroup0(_, buf1)) => {
                        if !self.can_match_buffer(*cbuf1, *buf1){
                            continue;
                        }
                        self.match_buffer(*cbuf1, *buf1);
                    },
                    (BindGroupReferenceBase::Bindgroup1(_, cbuf1,cbuf2), BindGroupReferenceBase::Bindgroup1(_, buf1,buf2)) => {
                        if !(self.can_match_buffer(*cbuf1, *buf1) && self.can_match_buffer(*cbuf2, *buf2)){
                            continue;
                        }
                        self.match_buffer(*cbuf1, *buf1);
                        self.match_buffer(*cbuf2, *buf2);
                    },
                    (BindGroupReferenceBase::Bindgroup2(_, cbuf1, cbuf2, cbuf3), BindGroupReferenceBase::Bindgroup2(_, buf1,buf2, buf3)) => {
                        if !(self.can_match_buffer(*cbuf1, *buf1) && self.can_match_buffer(*cbuf2, *buf2) && self.can_match_buffer(*cbuf3, *buf3)){
                            continue;
                        }
                        self.match_buffer(*cbuf1, *buf1);
                        self.match_buffer(*cbuf2, *buf2);
                        self.match_buffer(*cbuf3, *buf3);
                    },
                    (BindGroupReferenceBase::Bindgroup3(_, cbuf1,cbuf2, cbuf3, cbuf4), BindGroupReferenceBase::Bindgroup3(_, buf1, buf2, buf3, buf4)) =>{
                        if !(self.can_match_buffer(*cbuf1, *buf1) && self.can_match_buffer(*cbuf2, *buf2) && self.can_match_buffer(*cbuf3, *buf3) && self.can_match_buffer(*cbuf4, *buf4)){
                            continue;
                        }
                        self.match_buffer(*cbuf1, *buf1);
                        self.match_buffer(*cbuf2, *buf2);
                        self.match_buffer(*cbuf3, *buf3);
                        self.match_buffer(*cbuf4, *buf4);                    },
                    _ => {continue;}
                }
                dev.cached_bindgroup_reuse_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return *cached_id;
        }


        //create complete new Bindgroup, use first matching buffer
        let buffers = match &bindgroup_reference_buffers {
            BindGroupReferenceBase::Bindgroup0(meta, dest_buffer) => {
                BindGroupReferenceBase::Bindgroup0(
                    *meta,
                    self.load_buffer(dev, *dest_buffer),
                )
            }
            BindGroupReferenceBase::Bindgroup1(meta, dest_buffer, input1) => {
                BindGroupReferenceBase::Bindgroup1(
                    *meta,
                    self.load_buffer(dev, *dest_buffer),
                    self.load_buffer(dev, *input1),
                )
            }
            BindGroupReferenceBase::Bindgroup2(
                meta,
                dest_buffer,
                input1,
                input2,
            ) => {
                BindGroupReferenceBase::Bindgroup2(
                *meta,
                self.load_buffer(dev, *dest_buffer),
                self.load_buffer(dev, *input1),
                self.load_buffer(dev, *input2),
            )},
            BindGroupReferenceBase::Bindgroup3(
                meta,
                dest_buffer,
                input1,
                input2,
                input3,
            ) => {
                BindGroupReferenceBase::Bindgroup3(
                *meta,
                self.load_buffer(dev, *dest_buffer),
                self.load_buffer(dev, *input1),
                self.load_buffer(dev, *input2),
                self.load_buffer(dev, *input3),
            )},
        };
        dev.cached_bindgroup_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        //writeln!(file, "Bindgroup Create Cached: {}, reference_dest id: {}, pipeline: {:?}", id, bindgroup_reference.get_dest().id, pipeline_type).unwrap();
        
        return self.bindgroups.insert(BindgroupCache::new(create_bindgroup(dev,self, buffers.clone(), pipeline), buffers, pipeline_type))
    }

}



















// /// Virtual Buffer, used in Compute Graph
// #[derive(Debug)]
// pub struct BufferReference{
//     pub size : u64,
//     pub (crate) id : u32,
//     pub (crate) storage : Mutex<Option<Arc<CachedBuffer>>>,
//     device : WgpuDevice
// }

// impl Drop for BufferReference{
//     fn drop(&mut self) {
//         //println!("BufferReference.drop BufferReference.Storage.lock_start");
//         let mut storage = self.storage.lock().unwrap();
//         //println!("BufferReference.drop BufferReference.Storage.lock_end");
//         if let Some(s) = storage.as_ref(){
//             {

//                 let mut file = OpenOptions::new()
//                 .write(true)
//                 .append(true)
//                 .create(true)
//                 .open("debug-llama2c.txt")
//                 .unwrap();
//                 writeln!(file, "Buffer Cache {} dropped from {}", s.id, self.id).unwrap();
//                 //println!("BufferReference.drop  self.device.cache.lock_start");
//                 let mut cache = self.device.cache.lock().unwrap();

//                 //println!("Drop BufferReference to Buffer{}", s.id);

//                 //println!("BufferReference.drop  self.device.cache.lock_end");
//                 cache.buffers.add_buffer(s.clone());
//             }
//             *storage = None;
//         }
//     }
// }

// impl BufferReference{
//     pub (crate) fn new(dev : &WgpuDevice, size : u64) -> Arc<Self>{
//         Arc::new(Self{size, storage : Mutex::new(None), id: dev.cached_buffer_reference_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed), device : dev.clone()})
//     }

//     pub (crate) fn new_init(dev : &WgpuDevice, data : &[u8]) -> Arc<Self>{
//         //println!("BufferReference new_init, lock_start");
//         let mut cache = dev.cache.lock().unwrap();
//         //println!("BufferReference new_init, lock_end");
//         let id = dev.cached_buffer_reference_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
//         let buffer = cache.buffers.get_buffer(dev, data.len() as u64, id);
//         dev.queue.write_buffer(&buffer.buffer, 0, data);

//         Arc::new(Self{size : data.len() as u64, storage : Mutex::new(Some(buffer)), id, device : dev.clone()})
//     }
// }



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

fn create_bindgroup(dev : &WgpuDevice, cache : &ModelCache, bindgroup : BindGroupReferenceBase<BufferId>, pipeline: Arc<wgpu::ComputePipeline>) -> wgpu::BindGroup{

    let meta_offset= bindgroup.get_meta();

    let bind_group_layout = pipeline.get_bind_group_layout(0);    

    let buffer_meta = &dev.meta_buffer;

    let meta_binding = wgpu::BufferBinding{ buffer: &buffer_meta, offset: meta_offset  as u64 * 4, size:  None};
    let meta_bindung = wgpu::BindingResource::Buffer(meta_binding);

   match bindgroup{
        BindGroupReferenceBase::Bindgroup0(_, buffer_dest) => {
            let entries = &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_dest.get(cache).as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: meta_bindung, //buffer_meta.as_entire_binding(),
            }];
            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: entries
            })
        },
        BindGroupReferenceBase::Bindgroup1(_, buffer_dest, buffer_input1) => {
            let entries = &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_dest.get(cache).as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: meta_bindung, //buffer_meta.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffer_input1.get(cache).as_entire_binding(),
            },];
            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: entries
            })
        },
        BindGroupReferenceBase::Bindgroup2(_, buffer_dest, buffer_input1, buffer_input2) => {
            let entries = &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_dest.get(cache).as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: meta_bindung, //buffer_meta.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffer_input1.get(cache).as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buffer_input2.get(cache).as_entire_binding(),
            }];
            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: entries
            })
        },
        BindGroupReferenceBase::Bindgroup3(_, buffer_dest, buffer_input1, buffer_input2, buffer_input3) => {
            let entries: &[wgpu::BindGroupEntry; 5] = &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_dest.get(cache).as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: meta_bindung, //buffer_meta.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffer_input1.get(cache).as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buffer_input2.get(cache).as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: buffer_input3.get(cache).as_entire_binding(),
            },];
            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: entries
            })
        },
    }
}




// /// Cache of all free CachedBuffers
// #[derive(Debug)]
// pub (crate) struct BufferCache{
//     buffers : Vec<Arc<CachedBuffer>>
// }

// impl BufferCache{
//     fn new() -> Self{ 
//         return Self{buffers: vec![]}
//     }

//     fn get_buffer(&mut self, dev : &WgpuDevice, size : u64, buffer_reference_id : u32) -> Arc<CachedBuffer>{
//         let mut file = OpenOptions::new()
//         .write(true)
//         .append(true)
//         .create(true)
//         .open("debug-llama2c.txt")
//         .unwrap();
//         for (index, buffer) in self.buffers.iter().enumerate(){
//             if buffer.buffer.size() >= size{
//                 let buffer = self.buffers.remove(index);
//                 writeln!(file, "Buffer Reuse {} for {}", buffer.id, buffer_reference_id).unwrap();
//                 //println!("Reuse Buffer {}", buffer.id);
//                 dev.cached_buffer_reuse_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
//                 return buffer;
//             }
//         }

//         let id = dev.cached_buffer_counter.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
//         writeln!(file, "Buffer Create {} for {}", id, buffer_reference_id).unwrap();
//         Arc::new(CachedBuffer::new(create_buffer(dev, size),  id))
//     }

//     fn add_buffer(&mut self, buffer : Arc<CachedBuffer>){
//         self.buffers.push(buffer);
//     }

//     pub (crate) fn load_buffer(&mut self, dev : &WgpuDevice, buffer : Arc<BufferReference>) -> Arc<CachedBuffer>{
//         //println!("BufferCache load_buffer, lock_start");
//         let mut storage = buffer.storage.lock().unwrap();
//         //println!("BufferCache load_buffer, lock_end");
//         match storage.as_ref()
//         {
//             Some(storage) => return storage.clone(),
//             None => {
//                 let buffer = self.get_buffer(dev, buffer.size, buffer.id);
//                 *storage = Some(buffer.clone());
//                 return buffer;
//             },
//         }
//     }
    
//      //tries to reference the cached buffer from refernece
//      fn can_match_buffer(&mut self, cached : &Arc<CachedBuffer>, reference : &Arc<BufferReference>) -> bool{
//         let reference_storage = reference.storage.lock().unwrap();
//         if let Some(reference) = reference_storage.as_ref(){
//             if cached.id == reference.id{ //the cached buffer is alredy used by the reference
//                 return true;
//             }
//             else{        //the data reference points to is alredy stored inside another cached
//                 return false;
//             }
//         }
//         else{ //is the cached buffer free, and is is bit enaugh?
//             if let Some(index) = self.buffers.iter().position(|c| c == cached)
//             {
//                 if self.buffers[index].buffer.size() >= reference.size{
//                     return true;
//                 }
//             }
//         }   
//         return false;
//     }

//     //tries to reference the cached buffer from refernece
//     fn match_buffer(&mut self, cached : &Arc<CachedBuffer>, reference : &Arc<BufferReference>){
//         let mut reference_storage = reference.storage.lock().unwrap();
//         if let None = reference_storage.as_ref(){
//             if let Some(index) = self.buffers.iter().position(|c| c == cached)
//             {
//                *reference_storage = Some(self.buffers.remove(index));
//             }
//         }   
//     }

// }


// /// Cache of all available CachedBindGroups
// #[derive(Debug)]
// pub (crate) struct BindGroupCache{
//     bindgroups : Vec<Arc<CachedBindGroup>>
// }



// impl BindGroupCache {

//     fn new() -> Self{ 
//         return Self{bindgroups: vec![]}
//     }
// }



// /// A wgpu Buffer
// #[derive(Debug)]
// pub (crate) struct CachedBuffer{
//     pub (crate) buffer : wgpu::Buffer, //reference to the buffer, if Arc::strong_count is 1, the tensor is free to be reused, as long as it is big enaugh and not other enqued commands depend on that buffer
    
//     pub (crate) id : u32
// }

// impl PartialEq for CachedBuffer{
//     fn eq(&self, other: &Self) -> bool {
//         self.id == other.id
//     }
// }

// impl Eq for CachedBuffer{

// }

// impl CachedBuffer {
//     fn new(buffer: wgpu::Buffer, id : u32) -> Self {
//         //println!("Create_new Buffer size:{}, id:{}", buffer.size(), id);
//         Self { buffer,id}
//     }
// }


// pub (crate) type CachedBindGroupReference = BindGroupReferenceBase<Arc<CachedBuffer>>;

// impl PartialEq for CachedBindGroupReference {
//     fn eq(&self, other: &Self) -> bool {
//         match (self, other) {
//             (CachedBindGroupReference::Bindgroup0(a1, b1), CachedBindGroupReference::Bindgroup0(a2, b2)) => {
//                 a1 == a2 && Arc::ptr_eq(b1, b2)
//             },
//             (CachedBindGroupReference::Bindgroup1(a1, b1, c1), CachedBindGroupReference::Bindgroup1(a2, b2, c2)) => {
//                 a1 == a2 && Arc::ptr_eq(b1, b2) && Arc::ptr_eq(c1, c2)
//             },
//             (CachedBindGroupReference::Bindgroup2(a1, b1, c1, d1), CachedBindGroupReference::Bindgroup2(a2, b2, c2, d2)) => {
//                 a1 == a2 && Arc::ptr_eq(b1, b2) && Arc::ptr_eq(c1, c2) && Arc::ptr_eq(d1, d2)
//             },
//             (CachedBindGroupReference::Bindgroup3(a1, b1, c1, d1, e1), CachedBindGroupReference::Bindgroup3(a2, b2, c2, d2, e2)) => {
//                 a1 == a2 && Arc::ptr_eq(b1, b2) && Arc::ptr_eq(c1, c2) && Arc::ptr_eq(d1, d2) && Arc::ptr_eq(e1, e2)
//             },
//             _ => false,
//         }
//     }
// }

// impl Eq for CachedBindGroupReference {}

// // wgpu Bindgroup
// #[derive(Debug)]
// pub (crate) struct CachedBindGroup{
//     pub(crate) bindgroup : wgpu::BindGroup,
//     pipeline : PipelineType,
//     last_used : u32, //An Index referencing, when this cached item was used the last time,
//     buffers : CachedBindGroupReference,
//     id : u32
// }

// impl CachedBindGroup {
//     fn new(bindgroup: wgpu::BindGroup, buffers : CachedBindGroupReference, id : u32, pipeline : PipelineType ) -> Self {
//         Self { bindgroup, last_used : 0, buffers, id, pipeline }
//     }
// }


// #[derive(Debug)]
// pub struct ModelCache{
//     pub (crate) buffers : BufferCache,
//     pub (crate) bindgroups : BindGroupCache
// }

// impl ModelCache {
//     pub fn new() -> Self {
//         Self { buffers : BufferCache::new(), bindgroups : BindGroupCache::new() }
//     }

    
//     pub (crate) fn get_bind_group(&mut self, dev : &WgpuDevice, bindgroup_reference : &BindGroupReference, pipeline: Arc<wgpu::ComputePipeline>, pipeline_type: PipelineType) -> Arc<CachedBindGroup>{
//         let mut file = OpenOptions::new()
//         .write(true)
//         .append(true)
//         .create(true)
//         .open("debug-llama2c.txt")
//         .unwrap();

//         for cached_bindgroup in self.bindgroups.bindgroups.iter(){
//             if cached_bindgroup.buffers.get_meta() == bindgroup_reference.get_meta() && cached_bindgroup.pipeline == pipeline_type{
//                 //possible candiate, check if buffers match:

//                 match (&cached_bindgroup.buffers, bindgroup_reference){
//                     (BindGroupReferenceBase::Bindgroup0(_, cbuf1), BindGroupReferenceBase::Bindgroup0(_, buf1)) => {
//                         if !self.buffers.can_match_buffer(cbuf1, buf1){
//                             continue;
//                         }
//                         self.buffers.match_buffer(cbuf1, buf1);
//                     },
//                     (BindGroupReferenceBase::Bindgroup1(_, cbuf1,cbuf2), BindGroupReferenceBase::Bindgroup1(_, buf1,buf2)) => {
//                         if !(self.buffers.can_match_buffer(cbuf1, buf1) && self.buffers.can_match_buffer(cbuf2, buf2)){
//                             continue;
//                         }
//                         self.buffers.match_buffer(cbuf1, buf1);
//                         self.buffers.match_buffer(cbuf2, buf2);
//                     },
//                     (BindGroupReferenceBase::Bindgroup2(_, cbuf1, cbuf2, cbuf3), BindGroupReferenceBase::Bindgroup2(_, buf1,buf2, buf3)) => {
//                         if !(self.buffers.can_match_buffer(cbuf1, buf1) && self.buffers.can_match_buffer(cbuf2, buf2) && self.buffers.can_match_buffer(cbuf3, buf3)){
//                             continue;
//                         }
//                         self.buffers.match_buffer(cbuf1, buf1);
//                         self.buffers.match_buffer(cbuf2, buf2);
//                         self.buffers.match_buffer(cbuf3, buf3);
//                     },
//                     (BindGroupReferenceBase::Bindgroup3(_, cbuf1,cbuf2, cbuf3, cbuf4), BindGroupReferenceBase::Bindgroup3(_, buf1, buf2, buf3, buf4)) =>{
//                         if !(self.buffers.can_match_buffer(cbuf1, buf1) && self.buffers.can_match_buffer(cbuf2, buf2) && self.buffers.can_match_buffer(cbuf3, buf3) && self.buffers.can_match_buffer(cbuf4, buf4)){
//                             continue;
//                         }
//                         self.buffers.match_buffer(cbuf1, buf1);
//                         self.buffers.match_buffer(cbuf2, buf2);
//                         self.buffers.match_buffer(cbuf3, buf3);
//                         self.buffers.match_buffer(cbuf4, buf4);                    },
//                     _ => {continue;}
//                 }
//                 dev.cached_bindgroup_reuse_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
//                 writeln!(file, "Bindgroup Reuse Cached: {}, reference_dest id: {}, pipeline: {:?}", cached_bindgroup.id, bindgroup_reference.get_dest().id, pipeline_type).unwrap();
//                 //println!("Reuse Bindgroup: {}", cached_bindgroup.id);
//                 return cached_bindgroup.clone();
//             }
//         }

//         //create complete new Bindgroup, use first matching buffer
//         let bindgroup_reference = match &bindgroup_reference {
//             BindGroupReference::Bindgroup0(meta, dest_buffer) => {
//                 CachedBindGroupReference::Bindgroup0(
//                     *meta,
//                     self.buffers.load_buffer(dev, dest_buffer.clone()),
//                 )
//             }
//             BindGroupReference::Bindgroup1(meta, dest_buffer, input1) => {
//                 CachedBindGroupReference::Bindgroup1(
//                     *meta,
//                     self.buffers.load_buffer(dev, dest_buffer.clone()),
//                     self.buffers.load_buffer(dev, input1.clone()),
//                 )
//             }
//             BindGroupReference::Bindgroup2(
//                 meta,
//                 dest_buffer,
//                 input1,
//                 input2,
//             ) => {
//                 CachedBindGroupReference::Bindgroup2(
//                 *meta,
//                 self.buffers.load_buffer(dev, dest_buffer.clone()),
//                 self.buffers.load_buffer(dev, input1.clone()),
//                 self.buffers.load_buffer(dev, input2.clone()),
//             )},
//             BindGroupReference::Bindgroup3(
//                 meta,
//                 dest_buffer,
//                 input1,
//                 input2,
//                 input3,
//             ) => {
//                 CachedBindGroupReference::Bindgroup3(
//                 *meta,
//                 self.buffers.load_buffer(dev, dest_buffer.clone()),
//                 self.buffers.load_buffer(dev, input1.clone()),
//                 self.buffers.load_buffer(dev, input2.clone()),
//                 self.buffers.load_buffer(dev, input3.clone()),
//             )},
//         };
//         let id = dev.cached_bindgroup_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
//         writeln!(file, "Bindgroup Create Cached: {}, reference_dest id: {}, pipeline: {:?}", id, bindgroup_reference.get_dest().id, pipeline_type).unwrap();
//         let bindgroup = Arc::new(CachedBindGroup::new(create_bindgroup(dev, bindgroup_reference.clone(), pipeline), bindgroup_reference, id, pipeline_type));
//         self.bindgroups.bindgroups.push(bindgroup.clone());
//         return bindgroup;
//     }

// }

// //device, iteration: as models may compute caches at the first iteration, one might want to start caching from the second iteration
// pub fn start_cache(device : &crate::Device, value : u32){
//     match device{
//         crate::Device::Cpu => {},
//         crate::Device::Cuda(_) => {},
//         crate::Device::Metal(_) => {},
//         crate::Device::WebGpu(device) => {
//             let mut file = OpenOptions::new()
//             .write(true)
//             .append(true)
//             .create(true)
//             .open("debug-llama2c.txt")
//             .unwrap();
//             writeln!(file, "START CACHE").unwrap();
//             device.cached_buffer_reference_counter.store(value, std::sync::atomic::Ordering::Relaxed);
//         },
//     }
// }

// pub fn get_reference_cache(device : &crate::Device) -> u32{
//     match device{
//         crate::Device::Cpu => {0},
//         crate::Device::Cuda(_) => {0},
//         crate::Device::Metal(_) => {0},
//         crate::Device::WebGpu(device) => {
//             device.cached_buffer_reference_counter.load(std::sync::atomic::Ordering::Relaxed)
//         },
//     }
// }