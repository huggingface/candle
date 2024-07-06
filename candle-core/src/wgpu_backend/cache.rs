use std::{
    cmp::Ordering,
    collections::{BTreeSet, HashMap},
    num::NonZeroU64,
    sync::{
        atomic::{AtomicBool, AtomicU32},
        Arc, Mutex, Weak,
    },
};
use wgpu::BindGroupLayoutDescriptor;

use crate::WgpuDevice;

use super::{
    device::{BindGroupReference, PipelineType},
    util::{BTreeMulti, FixedSizeQueue, HashMapMulti, ToU64},
    wgpu_functions,
};

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
pub struct BufferReference {
    pub size: u64,
    pub(crate) storage: Mutex<Option<BufferId>>,
    device: WgpuDevice,
    pub(crate) is_referenced_by_storage: AtomicBool,
}

impl Drop for BufferReference {
    fn drop(&mut self) {
        let mut storage = self.storage.lock().unwrap();
        if let Some(s) = storage.as_ref() {
            {
                if self.device.use_cache {
                    let mut cache = self.device.cache.lock().unwrap();
                    cache.buffers.add_buffer(s.clone());
                }
            }
            *storage = None;
        }
    }
}

impl BufferReference {
    pub fn new<T: ToU64>(dev: &WgpuDevice, size: T) -> Arc<Self> {
        Arc::new(Self {
            size: size.to_u64(),
            storage: Mutex::new(None),
            device: dev.clone(),
            is_referenced_by_storage: AtomicBool::new(false),
        })
    }

    pub(crate) fn new_init(dev: &WgpuDevice, data: &[u8]) -> Arc<Self> {
        
        let mut queue = dev.command_queue.lock().unwrap();
        wgpu_functions::flush_gpu_command(dev, &mut queue);
        
        let mut cache = dev.cache.lock().unwrap();

        let buffer = cache.buffers.get_buffer(dev, data.len() as u64, true);
        dev.queue.write_buffer(&buffer.buffer, 0, data);

        Arc::new(Self {
            size: data.len() as u64,
            storage: Mutex::new(Some(buffer)),
            device: dev.clone(),
            is_referenced_by_storage: AtomicBool::new(false),
        })
    }
}

/// Cache of all free CachedBuffers
#[derive(Debug)]
pub(crate) struct BufferCache {
    buffers: BTreeMulti<u64, BufferId>,
    pub(crate) buffer_counter: u32, //total number of buffers created
    pub(crate) buffer_memory: u64,  //total memory allocated
    pub(crate) buffer_memory_free: u64, //total memory in buffers array
    pub(crate) max_memory_allowed: u64,
}

impl BufferCache {
    fn new() -> Self {
        return Self {
            buffers: BTreeMulti::new(),
            buffer_counter: 0,
            buffer_memory: 0,
            buffer_memory_free: 0,
            max_memory_allowed: 0,
        };
    }

    pub(crate) fn remove_unused(&mut self) {
        self.buffers.map.retain(|_, buffers| {
            buffers.retain(|b| {
                let keep = Arc::strong_count(b) != 1;
                if !keep {
                    self.buffer_counter -= 1;
                    self.buffer_memory -= b.buffer.size();
                    self.buffer_memory_free -= b.buffer.size();
                }
                keep
            });
            return buffers.len() > 1;
        });
    }

    fn check_buffer(&mut self, buffer: &Arc<CachedBuffer>) {
        let ref_count = Arc::strong_count(buffer);
        if ref_count == 2 {
            //remove buffer
            if let Some(buffers) = self.buffers.map.get_mut(&buffer.buffer.size()) {
                buffers.retain(|f| 
                    {
                        let keep = !Arc::ptr_eq(f, buffer);
                        if !keep{
                            self.buffer_memory_free -= buffer.buffer.size();    
                        }
                        return keep;
                        
                    });
                self.buffer_memory -= buffer.buffer.size();    
            }
        }
    }

    fn get_buffer(&mut self, dev: &WgpuDevice, size: u64, exact_size: bool) -> BufferId {
        let max_size = (size as f64 + self.max_memory_allowed as f64 * 0.05) as u64;

        if dev.use_cache {
            for (buffer_size, buffers) in self.buffers.map.range_mut(size..) {
                if *buffer_size < size {
                    panic!("Did not expect size to be smaller, than key");
                }

                if exact_size {
                    if *buffer_size != size {
                        break;
                    }
                } else {
                    if *buffer_size > max_size {
                        //allow max. 2 times the size
                        break;
                    }
                }

                if let Some(buffer) = buffers.pop() {
                    dev.cached_buffer_reuse_counter
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    self.buffer_memory_free -= buffer.buffer.size();
                    return buffer;
                }
            }
        }
        self.buffer_counter += 1;
        self.buffer_memory += size;

        if self.buffer_memory > self.max_memory_allowed { //we need to delete some free buffers
        }

        let id = dev
            .cached_buffer_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Arc::new(CachedBuffer::new(
            wgpu_functions::create_buffer(dev, size),
            id,
        ))
    }

    fn add_buffer(&mut self, buffer: BufferId) {
        let size = buffer.buffer.size();
        if self.buffers.add_mapping(size, buffer) {
            self.buffer_memory_free += size;
        }
    }

    fn is_buffer_free(&self, buffer: &BufferId) -> bool {
        self.buffers
            .get(&buffer.buffer.size())
            .iter()
            .any(|f| Arc::ptr_eq(buffer, f))
    }

    fn match_buffer_locked(
        &mut self,
        cached: BufferId,
        _: &BufferReferenceId,
        reference_storage: &mut Option<BufferId>,
    ) {
        if let None = reference_storage.as_ref() {
            let buffers = self.buffers.get_mut(&cached.buffer.size());
            if let Some(index) = buffers.iter().position(|c| c == &cached) {
                buffers.remove(index);
                self.buffer_memory_free -= cached.buffer.size();
            }
            *reference_storage = Some(cached);
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
            BindGroupReferenceBase::Bindgroup0(_) => BindGroupInput::Bindgroup0,
            BindGroupReferenceBase::Bindgroup1(_, v1) => BindGroupInput::Bindgroup1(v1),
            BindGroupReferenceBase::Bindgroup2(_, v1, v2) => BindGroupInput::Bindgroup2(v1, v2),
            BindGroupReferenceBase::Bindgroup3(_, v1, v2, v3) => {
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
pub(crate) struct BindGroupCache {
    bindgroups: HashMapMulti<BindGroupInput, BindgroupId>, //all bindgroups based on input buffers
    bindgroups_full: HashMap<BindGroupReferenceBase<BufferId>, BindgroupId>, //all bindgroups based on input und dest buffers
    order: BTreeSet<BindgroupCacheEntry>,
    pub(crate) bindgroup_counter: u32,
    pub(crate) cached_bindgroup_use_counter: u32,
}

#[derive(Debug)]
struct BindgroupCacheEntry(BindgroupId);

impl PartialEq for BindgroupCacheEntry {
    fn eq(&self, other: &Self) -> bool {
        self.0.last_used.load(std::sync::atomic::Ordering::Relaxed)
            == other.0.last_used.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Eq for BindgroupCacheEntry {}

impl PartialOrd for BindgroupCacheEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BindgroupCacheEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .last_used
            .load(std::sync::atomic::Ordering::Relaxed)
            .cmp(&other.0.last_used.load(std::sync::atomic::Ordering::Relaxed))
    }
}

impl BindGroupCache {
    fn new() -> Self {
        return Self {
            bindgroups: HashMapMulti::new(),
            bindgroups_full: HashMap::new(),
            bindgroup_counter: 0,
            order: BTreeSet::new(),
            cached_bindgroup_use_counter: 0,
        };
    }

    fn update_last_used(&mut self, value: &BindgroupId) {
        self.order.remove(&BindgroupCacheEntry(value.clone()));
        self.cached_bindgroup_use_counter += 1;
        value.last_used.store(
            self.cached_bindgroup_use_counter,
            std::sync::atomic::Ordering::Relaxed,
        );
        self.order.insert(BindgroupCacheEntry(value.clone()));
    }

    pub(crate) fn add(&mut self, key: BindGroupReferenceBase<BufferId>, value: BindgroupId) {
        if !self.bindgroups_full.contains_key(&key) {
            self.bindgroups_full.insert(key.clone(), value.clone());

            let key1: BindGroupInput = key.into();
            self.bindgroups.add_mapping(key1, value.clone());
            self.order.insert(BindgroupCacheEntry(value));
            self.bindgroup_counter += 1;
        }
    }

    pub(crate) fn get(&mut self, key: &BindGroupReferenceBase<BufferId>) -> Option<&BindgroupId> {
        self.bindgroups_full.get(key)
    }

    pub(crate) fn get_inputs(&mut self, key: &BindGroupInput) -> &Vec<BindgroupId> {
        self.bindgroups.get(key)
    }

    pub(crate) fn remove_bindgroup(&mut self, bindgroup: BindgroupId) {
        let key: BindGroupInput = bindgroup.buffers.clone().into();

        if let Some(values) = self.bindgroups.map.get_mut(&key) {
            if let Some(pos) = values.iter().position(|x| Arc::ptr_eq(x, &bindgroup)) {
                values.remove(pos);
            }
            if values.is_empty() {
                self.bindgroups.map.remove(&key);
            }
        }

        self.bindgroups_full.remove(&bindgroup.buffers);
        self.order.remove(&BindgroupCacheEntry(bindgroup));
    }
}

/// A wgpu Buffer
#[derive(Debug)]
pub struct CachedBuffer {
    pub(crate) buffer: wgpu::Buffer, //reference to the buffer, if Arc::strong_count is 1, the tensor is free to be reused, as long as it is big enaugh and not other enqued commands depend on that buffer

    pub(crate) id: u32,
}

impl PartialEq for CachedBuffer {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for CachedBuffer {}

impl CachedBuffer {
    fn new(buffer: wgpu::Buffer, id: u32) -> Self {
        Self { buffer, id }
    }
}

#[derive(Debug, Clone)]
pub enum BindGroupReferenceBase<T> {
    Bindgroup0(T),          //dest,
    Bindgroup1(T, T),       //dest, input1
    Bindgroup2(T, T, T),    //dest, input1, input2
    Bindgroup3(T, T, T, T), //dest, input1, input2, input3
}

impl<T> BindGroupReferenceBase<T> {
    pub(crate) fn get_dest(&self) -> &T {
        match self {
            BindGroupReferenceBase::Bindgroup0(dest) => dest,
            BindGroupReferenceBase::Bindgroup1(dest, _) => dest,
            BindGroupReferenceBase::Bindgroup2(dest, _, _) => dest,
            BindGroupReferenceBase::Bindgroup3(dest, _, _, _) => dest,
        }
    }
}

impl std::hash::Hash for BindGroupReferenceBase<BufferId> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            BindGroupReferenceBase::Bindgroup0(id0) => {
                state.write_u8(0);
                Arc::as_ptr(id0).hash(state);
            }
            BindGroupReferenceBase::Bindgroup1(id0, id1) => {
                state.write_u8(1);
                Arc::as_ptr(id0).hash(state);
                Arc::as_ptr(id1).hash(state);
            }
            BindGroupReferenceBase::Bindgroup2(id0, id1, id2) => {
                state.write_u8(2);
                Arc::as_ptr(id0).hash(state);
                Arc::as_ptr(id1).hash(state);
                Arc::as_ptr(id2).hash(state);
            }
            BindGroupReferenceBase::Bindgroup3(id0, id1, id2, id3) => {
                state.write_u8(3);
                Arc::as_ptr(id0).hash(state);
                Arc::as_ptr(id1).hash(state);
                Arc::as_ptr(id2).hash(state);
                Arc::as_ptr(id3).hash(state);
            }
        }
    }
}

pub type CachedBindGroupReference = BindGroupReferenceBase<BufferId>;

impl PartialEq for CachedBindGroupReference {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                CachedBindGroupReference::Bindgroup0(b1),
                CachedBindGroupReference::Bindgroup0(b2),
            ) => Arc::ptr_eq(b1, b2),
            (
                CachedBindGroupReference::Bindgroup1(b1, c1),
                CachedBindGroupReference::Bindgroup1(b2, c2),
            ) => Arc::ptr_eq(b1, b2) && Arc::ptr_eq(c1, c2),
            (
                CachedBindGroupReference::Bindgroup2(b1, c1, d1),
                CachedBindGroupReference::Bindgroup2(b2, c2, d2),
            ) => Arc::ptr_eq(b1, b2) && Arc::ptr_eq(c1, c2) && Arc::ptr_eq(d1, d2),
            (
                CachedBindGroupReference::Bindgroup3(b1, c1, d1, e1),
                CachedBindGroupReference::Bindgroup3(b2, c2, d2, e2),
            ) => {
                Arc::ptr_eq(b1, b2)
                    && Arc::ptr_eq(c1, c2)
                    && Arc::ptr_eq(d1, d2)
                    && Arc::ptr_eq(e1, e2)
            }
            _ => false,
        }
    }
}

impl Eq for CachedBindGroupReference {}

// wgpu Bindgroup
#[derive(Debug)]
pub struct CachedBindGroup {
    pub(crate) bindgroup: wgpu::BindGroup,
    last_used: AtomicU32, //An Index referencing, when this cached item was used the last time,
    buffers: CachedBindGroupReference,
    //id : u32
}

impl CachedBindGroup {
    fn new(bindgroup: wgpu::BindGroup, buffers: CachedBindGroupReference) -> Self {
        Self {
            bindgroup,
            last_used: AtomicU32::new(0),
            buffers,
        }
    }
}

#[derive(Debug)]
pub struct ModelCache {
    pub(crate) buffers: BufferCache,
    pub(crate) bindgroups: BindGroupCache,
    pub(crate) mappings: BufferMappingCache,
}

impl ModelCache {
    pub fn new() -> Self {
        Self {
            buffers: BufferCache::new(),
            bindgroups: BindGroupCache::new(),
            mappings: BufferMappingCache::new(),
        }
    }

    pub fn clear(&mut self) {
        self.bindgroups.bindgroups.map.clear();
        self.bindgroups.bindgroups.empty.clear();
        self.bindgroups.bindgroups_full.clear();
        self.bindgroups.cached_bindgroup_use_counter = 0;
        self.bindgroups.bindgroup_counter = 0;
        self.bindgroups.order.clear();
     
        self.buffers.buffer_counter = 0;
        self.buffers.buffer_memory = 0;
        self.buffers.buffers.empty.clear();
        self.buffers.buffers.map.clear();

        self.mappings.current_buffer_mapping = None;
        self.mappings.current_index = 0;
        self.mappings.last_buffer_mappings.deque.clear();
    }


    pub fn remove_unused(&mut self) -> bool {
        let mut counter = 0;

        if self.buffers.buffer_memory > (self.buffers.max_memory_allowed as f64 * 1.1) as u64{
            while self.buffers.buffer_memory > self.buffers.max_memory_allowed {
                if let Some(first) = self.bindgroups.order.first() {
                    counter += 1;
                    match &first.0.buffers {
                        BindGroupReferenceBase::Bindgroup0(v0) => self.buffers.check_buffer(v0),
                        BindGroupReferenceBase::Bindgroup1(v0, v1) => {
                            self.buffers.check_buffer(v0);
                            self.buffers.check_buffer(v1);
                        }
                        BindGroupReferenceBase::Bindgroup2(v0, v1, v2) => {
                            self.buffers.check_buffer(v0);
                            self.buffers.check_buffer(v1);
                            self.buffers.check_buffer(v2);
                        }
                        BindGroupReferenceBase::Bindgroup3(v0, v1, v2, v3) => {
                            self.buffers.check_buffer(v0);
                            self.buffers.check_buffer(v1);
                            self.buffers.check_buffer(v2);
                            self.buffers.check_buffer(v3);
                        }
                    }
    
                    self.bindgroups.remove_bindgroup(first.0.clone());
                } else {
                    break;
                }
            }
        }

        if counter > 1{
            log::info!("removed {counter} bindgroups, current memory {} / {}", self.buffers.buffer_memory, self.buffers.max_memory_allowed);
        }

        return counter > 1;
    }

    pub(crate) fn get_bind_group(
        &mut self,
        dev: &WgpuDevice,
        bindgroup_reference: &BindGroupReference,
        pipeline: PipelineType,
    ) -> BindgroupId {
        let buf_dest = bindgroup_reference.get_dest();

        let mut required_size = buf_dest.size;

        let mut reference_storage = buf_dest.storage.lock().unwrap();
        fn get_storage(v: &BufferReferenceId) -> BufferId {
            v.storage.lock().unwrap().as_ref().unwrap().clone()
        }
        fn get_buffer_referece_key(
            dest_buffer: BufferId,
            bindgroup_reference: &BindGroupReference,
        ) -> BindGroupReferenceBase<BufferId> {
            match bindgroup_reference {
                BindGroupReferenceBase::Bindgroup0(_) => {
                    BindGroupReferenceBase::Bindgroup0(dest_buffer)
                }
                BindGroupReferenceBase::Bindgroup1(_, v1) => {
                    BindGroupReferenceBase::Bindgroup1(dest_buffer, get_storage(v1))
                }
                BindGroupReferenceBase::Bindgroup2(_, v1, v2) => {
                    BindGroupReferenceBase::Bindgroup2(
                        dest_buffer,
                        get_storage(v1),
                        get_storage(v2),
                    )
                }
                BindGroupReferenceBase::Bindgroup3(_, v1, v2, v3) => {
                    BindGroupReferenceBase::Bindgroup3(
                        dest_buffer,
                        get_storage(v1),
                        get_storage(v2),
                        get_storage(v3),
                    )
                }
            }
        }
        fn get_bind_group_inner(
            cache: &mut ModelCache,
            dev: &WgpuDevice,
            pipeline: PipelineType,
            mut reference_storage: std::sync::MutexGuard<Option<Arc<CachedBuffer>>>,
            buf_dest: &Arc<BufferReference>,
            bindgroup_reference: &BindGroupReference,
            required_size: u64,
        ) -> BindgroupId {
            let exact_size = buf_dest
                .is_referenced_by_storage
                .load(std::sync::atomic::Ordering::Relaxed);

            let buffer = match reference_storage.as_ref() {
                Some(buffer) => buffer.clone(),
                None => cache.buffers.get_buffer(dev, required_size, exact_size),
            };
            cache
                .buffers
                .match_buffer_locked(buffer.clone(), buf_dest, &mut reference_storage);
            let buffer_weak_reference = Arc::downgrade(&buffer);
            let bindgroup_reference = get_buffer_referece_key(buffer, bindgroup_reference);
            let bindgroup = Arc::new(CachedBindGroup::new(
                wgpu_functions::create_bindgroup(dev, bindgroup_reference.clone()),
                bindgroup_reference,
            ));

            if dev.use_cache {
                cache
                    .bindgroups
                    .add(bindgroup.buffers.clone().into(), bindgroup.clone());

                cache.bindgroups.update_last_used(&bindgroup);

                cache
                    .mappings
                    .add_buffer(buffer_weak_reference, pipeline, buf_dest.size);
            }
            return bindgroup;
        }

        if dev.use_cache {
            if let Some(reference_storage) = reference_storage.as_ref().cloned() {
                //if dest already has a storage, seach exact matching bindgroup
                let buffer_weak_reference = Arc::downgrade(&reference_storage);
                let bindgroup_inputs =
                    get_buffer_referece_key(reference_storage.clone(), bindgroup_reference);

                if let Some(bg) = self.bindgroups.get(&bindgroup_inputs) {
                    self.mappings
                        .add_buffer(buffer_weak_reference, pipeline, buf_dest.size);
                    return bg.clone();
                }
            } else {
                if let Some(buffer) = self.mappings.get_buffer(pipeline.clone()) {
                    if self.buffers.is_buffer_free(&buffer) {
                        if buffer.buffer.size() >= required_size {
                            let buffer_weak_reference = Arc::downgrade(&buffer);

                            self.buffers.match_buffer_locked(
                                buffer.clone(),
                                buf_dest,
                                &mut reference_storage,
                            );
                            let bindgroup_inputs =
                                get_buffer_referece_key(buffer, bindgroup_reference);

                            if let Some(bg) = self.bindgroups.get(&bindgroup_inputs) {
                                self.mappings.add_buffer(
                                    buffer_weak_reference,
                                    pipeline,
                                    buf_dest.size,
                                );
                                return bg.clone();
                            } else {
                                //TODO
                                return get_bind_group_inner(
                                    self,
                                    dev,
                                    pipeline,
                                    reference_storage,
                                    buf_dest,
                                    bindgroup_reference,
                                    required_size,
                                );
                            }
                        } else {
                            //the required size increased -> also request a little bit more
                            required_size *= 2;
                        }
                    }
                }

                let bindgroup_inputs = match bindgroup_reference {
                    BindGroupReferenceBase::Bindgroup0(_) => BindGroupInput::Bindgroup0,
                    BindGroupReferenceBase::Bindgroup1(_, v1) => {
                        BindGroupInput::Bindgroup1(get_storage(v1))
                    }
                    BindGroupReferenceBase::Bindgroup2(_, v1, v2) => {
                        BindGroupInput::Bindgroup2(get_storage(v1), get_storage(v2))
                    }
                    BindGroupReferenceBase::Bindgroup3(_, v1, v2, v3) => {
                        BindGroupInput::Bindgroup3(
                            get_storage(v1),
                            get_storage(v2),
                            get_storage(v3),
                        )
                    }
                };

                let candidates_to_process = self
                    .bindgroups
                    .get_inputs(&bindgroup_inputs)
                    .iter()
                    .filter(|id| {
                        let cbuf_dest = id.buffers.get_dest();

                        if let Some(buf_dest) = reference_storage.as_ref() {
                            if cbuf_dest.id == buf_dest.id {
                                //the cached buffer is alredy used by the reference
                                return true;
                            } else {
                                //the data reference points to is alredy stored inside another cached
                                return false;
                            }
                        } else if cbuf_dest.buffer.size() >= required_size
                            && self.buffers.is_buffer_free(cbuf_dest)
                        {
                            //is the cached buffer free, and is is bit enaugh?
                            return true;
                        }
                        return false;
                    });

                let mut candidate_to_process = None;
                let mut best_size = u64::MAX;

                for ele in candidates_to_process {
                    let cbuf_dest = ele.buffers.get_dest();
                    let buffer_size = cbuf_dest.buffer.size();
                    if buffer_size < best_size {
                        candidate_to_process = Some(ele);
                        best_size = buffer_size;
                    }
                }

                if let Some(cached_bindgroup) = candidate_to_process.cloned() {
                    let cbuf1 = cached_bindgroup.buffers.get_dest();
                    let buffer_weak_reference = Arc::downgrade(&cbuf1);

                    self.buffers.match_buffer_locked(
                        cbuf1.clone(),
                        buf_dest,
                        &mut reference_storage,
                    );

                    dev.cached_bindgroup_reuse_counter
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                    self.bindgroups.update_last_used(&cached_bindgroup);

                    self.mappings
                        .add_buffer(buffer_weak_reference, pipeline, buf_dest.size);
                    return cached_bindgroup.clone();
                }
            }
        }

        return get_bind_group_inner(
            self,
            dev,
            pipeline,
            reference_storage,
            buf_dest,
            bindgroup_reference,
            required_size,
        );
    }
}

///Cache, that stores previously Flushed Gpu Commands, we try to use the same buffers as the last time
#[derive(Debug)]
pub(crate) struct BufferMappingCache {
    pub(crate) last_buffer_mappings: FixedSizeQueue<CachedBufferMappings>,
    pub(crate) current_buffer_mapping: Option<CachedBufferMappings>,
    pub(crate) current_index: u32,
}

impl BufferMappingCache {
    fn new() -> Self {
        Self {
            last_buffer_mappings: FixedSizeQueue::new(10),
            current_buffer_mapping: None,
            current_index: 0,
        }
    }

    pub(crate) fn set_current_buffer_mapping(&mut self, hash: u64) {
        let index = self
            .last_buffer_mappings
            .iter()
            .position(|b| b.hash == hash);
        if let Some(index) = index {
            self.current_buffer_mapping = self.last_buffer_mappings.deque.remove(index);
        } else {
            self.current_buffer_mapping = Some(CachedBufferMappings::new(hash));
        }
    }

    pub(crate) fn get_buffer(&mut self, pipeline: PipelineType) -> Option<BufferId> {
        if let Some(mapping) = &self.current_buffer_mapping {
            if let Some(mapping) = &mapping.data.get(self.current_index as usize) {
                if mapping.pipeline == pipeline {
                    if let Some(buffer) = mapping.used_buffer.upgrade() {
                        return Some(buffer);
                    }
                }
            }
        } else {
            panic!("expected current buffer to be set");
        }
        return None;
    }

    ///Stores, that at the provided buffer was used
    pub(crate) fn add_buffer(
        &mut self,
        buffer: Weak<CachedBuffer>,
        pipeline: PipelineType,
        size: u64,
    ) {
        if let Some(mapping) = &mut self.current_buffer_mapping {
            let data = CachedBufferMapping::new(pipeline, size, buffer);
            if (self.current_index as usize) < mapping.data.len() {
                mapping.data[self.current_index as usize] = data;
            } else {
                mapping.data.push(data);
            }
        } else {
            panic!("expected current buffer to be set");
        }

        self.current_index += 1;
    }

    pub(crate) fn finish(&mut self) {
        if let Some(value) = self.current_buffer_mapping.take() {
            self.last_buffer_mappings.push(value);
        }
        self.current_index = 0;
    }
}

#[derive(Debug)]
pub(crate) struct CachedBufferMapping {
    pub(crate) pipeline: PipelineType,
    pub(crate) requested_size: u64,
    pub(crate) used_buffer: Weak<CachedBuffer>,
}

impl CachedBufferMapping {
    fn new(pipeline: PipelineType, requested_size: u64, used_buffer: Weak<CachedBuffer>) -> Self {
        Self {
            pipeline,
            requested_size,
            used_buffer,
        }
    }
}

#[derive(Debug)]
pub(crate) struct CachedBufferMappings {
    pub(crate) data: Vec<CachedBufferMapping>, //all shader calls, and there used BufferCache
    pub(crate) hash: u64,
}

impl CachedBufferMappings {
    fn new(hash: u64) -> Self {
        Self { data: vec![], hash }
    }
}
