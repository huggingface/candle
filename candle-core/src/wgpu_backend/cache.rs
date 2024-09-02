use std::{
    collections::{BTreeSet, HashMap},
    num::NonZeroU64,
    u32,
};

use tracing::instrument;
use wgpu::BindGroupLayoutDescriptor;

use crate::wgpu_backend::util::StorageTrait;

use super::{
    device::PipelineType,
    util::{FixedSizeQueue, HashMapMulti},
    wgpu_functions,
};
use super::{
    util::{Reference, ReferenceTrait, Storage, StorageOptional, ToU64},
    WgpuDevice,
};

//time = 0 is undefined
// pub type BufferReferenceId  = Reference;
// pub type CachedBufferId  = Reference;
// pub type CachedBindgroupId = Reference;

#[derive(Debug, PartialEq, Eq, Hash, Clone, std::marker::Copy, Default)]
#[cfg_attr(
    any(feature = "wgpu_debug_serialize", feature = "wgpu_debug"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct BufferReferenceId(Reference);

#[derive(Debug, PartialEq, Eq, Hash, Clone, std::marker::Copy)]
pub struct CachedBufferId(Reference);
#[derive(Debug, PartialEq, Eq, Hash, Clone, std::marker::Copy)]
pub struct CachedBindgroupId(Reference);

impl ReferenceTrait for BufferReferenceId {
    fn new(id: u32, time: u32) -> Self {
        Self(Reference::new(id, time))
    }

    fn id(&self) -> u32 {
        self.0.id()
    }

    fn time(&self) -> u32 {
        self.0.time()
    }
}

impl ReferenceTrait for CachedBufferId {
    fn new(id: u32, time: u32) -> Self {
        Self(Reference::new(id, time))
    }

    fn id(&self) -> u32 {
        self.0.id()
    }

    fn time(&self) -> u32 {
        self.0.time()
    }
}

impl ReferenceTrait for CachedBindgroupId {
    fn new(id: u32, time: u32) -> Self {
        Self(Reference::new(id, time))
    }

    fn id(&self) -> u32 {
        self.0.id()
    }

    fn time(&self) -> u32 {
        self.0.time()
    }
}

#[derive(Debug)]
pub(crate) struct BindgroupLayouts {
    pub bind_group_layout0: wgpu::BindGroupLayout,
    pub bind_group_layout1: wgpu::BindGroupLayout,
    pub bind_group_layout1_16: wgpu::BindGroupLayout,
    pub bind_group_layout2: wgpu::BindGroupLayout,
    pub bind_group_layout2_16: wgpu::BindGroupLayout, //for matmul, input buffer may be vec4
    pub bind_group_layout3: wgpu::BindGroupLayout,
    pub pipeline_layout0: wgpu::PipelineLayout,
    pub pipeline_layout1: wgpu::PipelineLayout,
    pub pipeline_layout1_16: wgpu::PipelineLayout,
    pub pipeline_layout2: wgpu::PipelineLayout,
    pub pipeline_layout2_16: wgpu::PipelineLayout, //for matmul, input buffer may be vec4
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

        let dest_entry_16 = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: Some(NonZeroU64::new(16).unwrap()),
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

        let input1_entry_16 = wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: Some(NonZeroU64::new(16).unwrap()),
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

        let input2_entry_16 = wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: Some(NonZeroU64::new(16).unwrap()),
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
        let bind_group_layout1_16 = dev.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[dest_entry_16, meta_entry, input1_entry_16],
        });
        let bind_group_layout2 = dev.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[dest_entry, meta_entry, input1_entry, input2_entry],
        });
        let bind_group_layout2_16 = dev.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[dest_entry, meta_entry, input1_entry_16, input2_entry_16],
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
        let pipeline_layout1_16 = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout1_16],
            push_constant_ranges: &[],
        });
        let pipeline_layout2 = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout2],
            push_constant_ranges: &[],
        });
        let pipeline_layout2_16 = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout2_16],
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
            bind_group_layout1_16,
            bind_group_layout2,
            bind_group_layout2_16,
            bind_group_layout3,
            pipeline_layout0,
            pipeline_layout1,
            pipeline_layout1_16,
            pipeline_layout2,
            pipeline_layout2_16,
            pipeline_layout3,
        }
    }
}

////////////////// BUFFER REFERENCE:

/// Virtual Buffer, used in Compute Graph
#[derive(Debug)]
pub struct BufferReference {
    size: u64,
    referenced_by_candle_storage: bool,
    cached_buffer_id: CachedBufferId,
    first_used: u32,
    last_used: u32, //u32::max means indefitly
}

impl BufferReference {
    pub fn new(size: u64, referenced_by_candle_storage: bool) -> Self {
        Self {
            size,
            cached_buffer_id: CachedBufferId::new(0, 0),
            referenced_by_candle_storage,
            first_used: 0,
            last_used: if referenced_by_candle_storage {
                u32::MAX
            } else {
                0
            },
        }
    }

    pub fn new_with_storage(
        size: u64,
        cached_buffer_id: CachedBufferId,
        referenced_by_candle_storage: bool,
    ) -> Self {
        Self {
            size,
            cached_buffer_id,
            referenced_by_candle_storage,
            first_used: 0,
            last_used: if referenced_by_candle_storage {
                u32::MAX
            } else {
                0
            },
        }
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn set_cached_buffer_id(&mut self, cached_buffer_id: CachedBufferId) {
        self.cached_buffer_id = cached_buffer_id;
    }

    pub fn cached_buffer_id(&self) -> &CachedBufferId {
        &self.cached_buffer_id
    }

    pub fn referenced_by_candle_storage(&self) -> bool {
        self.referenced_by_candle_storage
    }

    pub fn set_referenced_by_candle_storage(&mut self, referenced_by_candle_storage: bool) {
        self.referenced_by_candle_storage = referenced_by_candle_storage;
    }

    pub fn first_used(&self) -> u32 {
        self.first_used
    }

    pub fn set_first_used(&mut self, first_used: u32) {
        self.first_used = first_used;
    }

    pub fn last_used(&self) -> u32 {
        self.last_used
    }

    pub fn set_last_used(&mut self, last_used: u32) {
        self.last_used = last_used;
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(
    any(feature = "wgpu_debug_serialize", feature = "wgpu_debug"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum BindgroupInputBase<T> {
    Bindgroup0, //
    Bindgroup1(
        #[cfg_attr(
            any(feature = "wgpu_debug_serialize", feature = "wgpu_debug"),
            serde(skip)
        )]
        T,
        bool,
    ), //input1
    Bindgroup2(
        #[cfg_attr(
            any(feature = "wgpu_debug_serialize", feature = "wgpu_debug"),
            serde(skip)
        )]
        T,
        #[cfg_attr(
            any(feature = "wgpu_debug_serialize", feature = "wgpu_debug"),
            serde(skip)
        )]
        T,
        bool,
    ), //input1, input2, is_16
    Bindgroup3(
        #[cfg_attr(
            any(feature = "wgpu_debug_serialize", feature = "wgpu_debug"),
            serde(skip)
        )]
        T,
        #[cfg_attr(
            any(feature = "wgpu_debug_serialize", feature = "wgpu_debug"),
            serde(skip)
        )]
        T,
        #[cfg_attr(
            any(feature = "wgpu_debug_serialize", feature = "wgpu_debug"),
            serde(skip)
        )]
        T,
    ), //input1, input2, input3
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(
    any(feature = "wgpu_debug_serialize", feature = "wgpu_debug"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct BindgroupFullBase<T>(
    #[cfg_attr(
        any(feature = "wgpu_debug_serialize", feature = "wgpu_debug"),
        serde(skip)
    )]
    T,
    BindgroupInputBase<T>,
);

impl<T> BindgroupFullBase<T> {
    pub(crate) fn new(dest: T, input: BindgroupInputBase<T>) -> Self {
        return BindgroupFullBase(dest, input);
    }

    pub(crate) fn get_dest(&self) -> &T {
        return &self.0;
    }

    pub(crate) fn get_input(&self) -> &BindgroupInputBase<T> {
        return &self.1;
    }
}

pub type CachedBindgroupInput = BindgroupInputBase<CachedBufferId>;
pub type CachedBindgroupFull = BindgroupFullBase<CachedBufferId>;
pub type BindgroupReferenceInput = BindgroupInputBase<BufferReferenceId>;
pub type BindgroupReferenceFull = BindgroupFullBase<BufferReferenceId>;

////////////////// CACHED BUFFER:

#[derive(Debug)]
pub struct CachedBuffer {
    buffer: wgpu::Buffer,
    //stored_free : bool,    //wheter this buffer was free at the beginning to the queue
    is_free: bool, //wheter this buffer is currently free
    last_used_counter: u32,
    //used_memory : u64, //the total memory this buffer was unsed for. Together with usage_counter we get the average buffer size, this buffer is used for
}

impl CachedBuffer {
    pub fn new(buffer: wgpu::Buffer) -> Self {
        Self {
            buffer,
            is_free: false,
            last_used_counter: 0,
        } //stored_free : false, used_memory : 0  }
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn is_free(&self) -> bool {
        self.is_free
    }
}

#[derive(Debug)]
pub struct CachedBindgroup {
    bindgroup: wgpu::BindGroup,
    buffer: CachedBindgroupFull,
}

impl CachedBindgroup {
    pub fn new(bindgroup: wgpu::BindGroup, buffer: CachedBindgroupFull) -> Self {
        Self { bindgroup, buffer }
    }

    pub fn bindgroup(&self) -> &wgpu::BindGroup {
        &self.bindgroup
    }

    pub(crate) fn buffer(&self) -> &CachedBindgroupFull {
        &self.buffer
    }
}

#[derive(Debug)]
pub struct ModelCache {
    pub(crate) buffer_reference: BufferReferenceStorage,
    pub(crate) buffers: BufferCacheStorage,
    pub(crate) bindgroups: BindgroupCacheStorage,
    pub(crate) mappings: BufferMappingCache,

    #[cfg(feature = "wgpu_debug")]
    pub(crate) debug:
        HashMap<super::device::DebugPipelineRecording, super::device::DebugPipelineRecording>, //stores all queed commands (e.g. used to test performance in the browser so we can measure the performance of each unnique operation alone)
}

impl ModelCache {
    pub fn new(mapping_size: u32) -> Self {
        Self {
            buffer_reference: BufferReferenceStorage::new(),
            buffers: BufferCacheStorage::new(),
            bindgroups: BindgroupCacheStorage::new(),
            mappings: BufferMappingCache::new(mapping_size),

            #[cfg(feature = "wgpu_debug")]
            debug: HashMap::new(),
        }
    }

    #[instrument(skip(self, size))]
    pub fn create_buffer_reference<T: ToU64>(
        &mut self,
        size: T,
        referenced_by_candle_storage: bool,
    ) -> BufferReferenceId {
        let buffer_reference = BufferReference::new(size.to_u64(), referenced_by_candle_storage);
        return self.buffer_reference.insert(buffer_reference);
    }

    #[instrument(skip(self, data))]
    pub fn create_buffer_reference_init<T: bytemuck::Pod>(
        &mut self,
        dev: &WgpuDevice,
        data: &[T],
        referenced_by_candle_storage: bool,
    ) -> BufferReferenceId {
        let data = bytemuck::cast_slice(data);

        let buffer = self
            .buffers
            .search_buffer(dev, data.len() as u64, 0, u32::MAX - 1); //TODO use exact size?
        dev.queue
            .write_buffer(&self.buffers.get_buffer(&buffer).unwrap().buffer, 0, data);

        let buffer_reference = BufferReference::new_with_storage(
            data.len() as u64,
            buffer,
            referenced_by_candle_storage,
        );
        return self.buffer_reference.insert(buffer_reference);
    }

    /// returns, wheter we should stop the command_queue and delete not used buffers
    #[instrument(skip(self))]
    pub fn should_delete_unused(&mut self) -> bool {
        let current_memory = self.buffers.buffer_memory;
        let memory_margin = self.buffers.max_memory_allowed;
        if current_memory > memory_margin {
            return !self.buffers.order.is_empty();
        }
        return false;
    }

    #[instrument(skip(self))]
    pub fn remove_unused(&mut self) -> bool {
        let current_memory = self.buffers.buffer_memory;
        let memory_margin = self.buffers.max_memory_allowed;
        let delete_until_margin = (self.buffers.max_memory_allowed * 4) / 5;

        //remove buffers, that
        // 1. were not used for a long time
        // 2. have a big memory diff (the actual buffer size vs the average size the buffer is used with)
        let mut check_bindgroups = false;
        if current_memory > memory_margin {
            log::debug!(
                "deleting buffers: ({}) current {current_memory}/{memory_margin}",
                self.buffers.storage.len()
            );

            //every entry in self.buffers.order will be free and can be potentially deleted
            //this is ordered from small to big.
            let buffers: Vec<_> = self
                .buffers
                .order
                .iter()
                .map(|entry| {
                    let (id, val) = self
                        .buffers
                        .storage
                        .get_reference(entry.index)
                        .expect("item in order, that could ne be found in storage");
                    return (id, val.last_used_counter);
                })
                .collect();

            for (id, _) in buffers {
                check_bindgroups = true;
                self.buffers.delete_buffer(&id);

                if self.buffers.buffer_memory <= delete_until_margin {
                    break; //deleted enaugh
                }
            }
            let current_memory = self.buffers.buffer_memory;
            log::debug!(
                "after deleting: ({}) current {current_memory}/{}",
                self.buffers.storage.len(),
                self.buffers.max_memory_allowed
            );
        }

        //remove bindgroups:
        //1. if we removed a buffer, we should also remove the bindgroup
        //2. bindgroups that werent used for a long time may be deleted

        if check_bindgroups {
            self.bindgroups.retain_bindgroups(|bindgroup| {
                let check_buffer = |buffer_reference| {
                    return self.buffers.get_buffer(buffer_reference).is_some();
                };

                let is_valid = check_buffer(bindgroup.buffer.get_dest())
                    && match &bindgroup.buffer.get_input() {
                        BindgroupInputBase::Bindgroup0 => true,
                        BindgroupInputBase::Bindgroup1(v1, _) => check_buffer(v1),
                        BindgroupInputBase::Bindgroup2(v1, v2, _) => {
                            check_buffer(v1) && check_buffer(v2)
                        }
                        BindgroupInputBase::Bindgroup3(v1, v2, v3) => {
                            check_buffer(v1) && check_buffer(v2) && check_buffer(v3)
                        }
                    };

                //check if all buffers for this bindgroup still exist!
                if !is_valid {
                    return false;
                }

                return true;
            });
        }
        return false;
    }

    #[instrument(skip(self, dev))]
    pub(crate) fn get_bind_group(
        &mut self,
        dev: &WgpuDevice,
        bindgroup_reference: &BindgroupReferenceFull,
        pipeline: PipelineType,
        command_id: u32,
    ) -> CachedBindgroupId {
        fn check_buffer_reference(
            cache: &mut ModelCache,
            bindgroup_reference: &BindgroupReferenceFull,
            pipeline: PipelineType,
        ) {
            let check_buffer = |buffer_reference_id| {
                if let Some(buffer_reference) = cache.buffer_reference.get(buffer_reference_id) {
                    if !buffer_reference.cached_buffer_id.is_valid() {
                        panic!("input buffer {:?}({:?}) in {:?} had no cached_storage set for pipeline {:?}", buffer_reference,buffer_reference_id, bindgroup_reference, pipeline);
                    } else {
                        if cache
                            .buffers
                            .get_buffer(&buffer_reference.cached_buffer_id)
                            .is_none()
                        {
                            if let Some(buffer_reference) =
                                cache.buffer_reference.get(buffer_reference_id)
                            {
                                panic!("input buffer {:?}({:?}) in {:?} had no cached_storage set to {:?} widch could not be found for pipeline {:?}", buffer_reference,buffer_reference_id, bindgroup_reference,buffer_reference.cached_buffer_id, pipeline);
                            }
                        }
                    }
                } else {
                    if let Some(val) = cache
                        .buffer_reference
                        .get_reference(buffer_reference_id.id())
                    {
                        panic!("Reference {:?} inside Bindgroup {:?} invalid for pipeline {:?}, Reference was replaced, current: {:?}", buffer_reference_id, bindgroup_reference, pipeline, val.0);
                    } else {
                        panic!("Reference {:?} inside Bindgroup {:?} invalid for pipeline {:?} (Reference was deleted)", buffer_reference_id, bindgroup_reference, pipeline);
                    }
                }
            };

            match &bindgroup_reference.1 {
                BindgroupInputBase::Bindgroup0 => BindgroupInputBase::Bindgroup0,
                BindgroupInputBase::Bindgroup1(v1, is_16) => {
                    BindgroupInputBase::Bindgroup1(check_buffer(v1), *is_16)
                }
                BindgroupInputBase::Bindgroup2(v1, v2, is_16) => {
                    BindgroupInputBase::Bindgroup2(check_buffer(v1), check_buffer(v2), *is_16)
                }
                BindgroupInputBase::Bindgroup3(v1, v2, v3) => BindgroupInputBase::Bindgroup3(
                    check_buffer(v1),
                    check_buffer(v2),
                    check_buffer(v3),
                ),
            };
        }

        check_buffer_reference(self, bindgroup_reference, pipeline.clone());

        fn get_storage(cache: &ModelCache, id: &BufferReferenceId) -> CachedBufferId {
            cache
                .buffer_reference
                .get(&id)
                .unwrap()
                .cached_buffer_id
                .clone()
        }

        fn get_buffer_referece_key(
            cache: &ModelCache,
            dest_buffer: CachedBufferId,
            bindgroup_reference: &BindgroupReferenceFull,
        ) -> CachedBindgroupFull {
            return BindgroupFullBase(
                dest_buffer,
                match &bindgroup_reference.1 {
                    BindgroupInputBase::Bindgroup0 => BindgroupInputBase::Bindgroup0,
                    BindgroupInputBase::Bindgroup1(v1, is_16) => {
                        BindgroupInputBase::Bindgroup1(get_storage(cache, v1), *is_16)
                    }
                    BindgroupInputBase::Bindgroup2(v1, v2, is_16) => {
                        BindgroupInputBase::Bindgroup2(
                            get_storage(cache, v1),
                            get_storage(cache, v2),
                            *is_16,
                        )
                    }
                    BindgroupInputBase::Bindgroup3(v1, v2, v3) => BindgroupInputBase::Bindgroup3(
                        get_storage(cache, v1),
                        get_storage(cache, v2),
                        get_storage(cache, v3),
                    ),
                },
            );
        }

        let buf_dest_id = bindgroup_reference.get_dest();
        let buf_dest_length;
        let mut buf_dest_cached_id;
        let mut required_size;
        {
            let buf_dest_reference = self.buffer_reference.get(buf_dest_id).unwrap();
            buf_dest_cached_id = buf_dest_reference.cached_buffer_id.clone();
            required_size = buf_dest_reference.size;
            buf_dest_length = buf_dest_reference.last_used - buf_dest_reference.first_used;
            if buf_dest_reference.last_used < buf_dest_reference.first_used {
                panic!("buffer {:?}({:?})", buf_dest_reference, buf_dest_id);
            }
        }

        if dev.configuration.use_cache {
            //the destination buffer of this bindgroup already has a buffer set
            if buf_dest_cached_id.is_valid() {
                let bindgroup_inputs =
                    get_buffer_referece_key(self, buf_dest_cached_id, &bindgroup_reference);
                if let Some(bg) = self
                    .bindgroups
                    .get_bindgroup_reference_by_description(&bindgroup_inputs)
                    .cloned()
                {
                    self.bindgroups.cached_bindgroup_use_counter += 1;
                    self.mappings.add_buffer(buf_dest_cached_id, pipeline);
                    return bg;
                }
            }
            //reference storage is not set -> search a free buffer or create new one
            else {
                if let Some(buffer_id) = self.mappings.get_buffer(pipeline.clone()) {
                    let buffer: Option<&CachedBuffer> = self.buffers.get_buffer(&buffer_id);
                    if let Some(buffer) = buffer {
                        if buffer.is_free() {
                            if buffer.buffer.size() >= required_size {
                                let buf_dest_reference =
                                    self.buffer_reference.get_mut(buf_dest_id).unwrap();
                                //use this buffer for the buffer reference:
                                buf_dest_reference.cached_buffer_id = buffer_id;
                                buf_dest_cached_id = buffer_id;
                                self.buffers.use_buffer(&buffer_id, command_id);

                                //reuse a bindgroup, if we could find one:
                                let bindgroup_inputs =
                                    get_buffer_referece_key(self, buffer_id, &bindgroup_reference);
                                if let Some(bg) = self
                                    .bindgroups
                                    .get_bindgroup_reference_by_description(&bindgroup_inputs)
                                    .cloned()
                                {
                                    self.bindgroups.cached_bindgroup_use_counter += 1;
                                    self.mappings.add_buffer(buffer_id, pipeline);
                                    return bg.clone();
                                }
                            } else {
                                //the required size increased -> also request a little bit more
                                required_size *= 2;
                            }
                        }
                    }
                }

                let bindgroup_inputs =
                    get_buffer_referece_key(self, CachedBufferId::new(0, 0), &bindgroup_reference);
                // let bindgroup_inputs = &bindgroup_reference.1;
                let max_size: u64 =
                    BufferCacheStorage::max_cached_size(required_size as u64, buf_dest_length);

                let candidates_to_process = self
                    .bindgroups
                    .enumerate_bindgroup_by_description_input(&bindgroup_inputs.1)
                    .filter_map(|(id, bindgroup)| {
                        let cbuf_dest_id = bindgroup.buffer.get_dest();

                        if buf_dest_cached_id.is_valid() {
                            if let Some(bindgroup) = self.bindgroups.get_bindgroup(&id) {
                                if buf_dest_cached_id == *bindgroup.buffer.get_dest() {
                                    if let Some(c_buf_dest) = self.buffers.get_buffer(cbuf_dest_id)
                                    {
                                        return Some((id, bindgroup, c_buf_dest.buffer.size()));
                                    }
                                }
                            }
                        } else {
                            if let Some(c_buf_dest) = self.buffers.get_buffer(cbuf_dest_id) {
                                if c_buf_dest.buffer.size() >= required_size
                                    && c_buf_dest.is_free()
                                    && c_buf_dest.buffer.size() <= max_size
                                {
                                    return Some((id, bindgroup, c_buf_dest.buffer.size()));
                                }
                            }
                        }
                        return None;
                    });

                //cachedBindgroupId, CachedDest_BufferId
                let mut candidate_to_process = None;
                let mut best_size = u64::MAX;

                for (ele_id, ele_bindgroup, dest_buffer_size) in candidates_to_process {
                    let cbuf_dest_id = ele_bindgroup.buffer.get_dest();
                    if dest_buffer_size < best_size {
                        candidate_to_process = Some((ele_id, cbuf_dest_id));
                        best_size = dest_buffer_size;
                    }
                }

                //if we found a bindgroup we can reuse -> use this bindgroup
                if let Some((cached_bindgroup_id, cached_dest_buffer_id)) = candidate_to_process {
                    //use this buffer for the buffer reference:
                    let buf_dest_reference = self.buffer_reference.get_mut(buf_dest_id).unwrap();
                    buf_dest_reference.cached_buffer_id = *cached_dest_buffer_id;
                    self.buffers.use_buffer(&cached_dest_buffer_id, command_id);
                    self.mappings.add_buffer(*cached_dest_buffer_id, pipeline);
                    self.bindgroups.cached_bindgroup_use_counter += 1;
                    return cached_bindgroup_id;
                }
            }
        }

        let buf_dest_reference = self.buffer_reference.get_mut(buf_dest_id).unwrap();

        //create new buffer, if buffer was not already set:
        let dest_buffer_id;
        if buf_dest_reference.cached_buffer_id.is_valid() {
            //this buffer reference already has a buffer connected,use this buffer
            dest_buffer_id = buf_dest_reference.cached_buffer_id;
        } else {
            //create a new buffer
            dest_buffer_id =
                self.buffers
                    .search_buffer(dev, required_size, command_id, buf_dest_length);
            //use this buffer for the buffer reference:
            buf_dest_reference.cached_buffer_id = dest_buffer_id;
        }

        //create new bindgroup:
        let bindgroup_reference =
            get_buffer_referece_key(self, dest_buffer_id, bindgroup_reference);
        let bindgroup_id = self.create_bindgroup(dev, bindgroup_reference);

        if dev.configuration.use_cache {
            self.mappings.add_buffer(dest_buffer_id, pipeline);
        }
        return bindgroup_id;
    }

    //creats a Bindgroup
    #[instrument(skip(self, dev))]
    fn create_bindgroup(
        &mut self,
        dev: &WgpuDevice,
        bindgroup_d: CachedBindgroupFull,
    ) -> CachedBindgroupId {
        let bindgroup = wgpu_functions::create_bindgroup(dev, bindgroup_d.clone(), self);
        let bindgroup = CachedBindgroup::new(bindgroup, bindgroup_d.clone());
        let id = self.bindgroups.storage.insert(bindgroup);

        self.bindgroups
            .bindgroups
            .add_mapping(bindgroup_d.1.clone(), id);
        self.bindgroups.bindgroups_full.insert(bindgroup_d, id);

        self.bindgroups.bindgroup_counter += 1;

        return id;
    }
}

#[derive(Debug)]
pub(crate) struct BufferReferenceStorage {
    storage: Storage<BufferReference, BufferReferenceId>,
    deletion_queue: Vec<BufferReferenceId>, //entires that are marked for deletion
}

impl BufferReferenceStorage {
    fn new() -> Self {
        Self {
            storage: Storage::new(),
            deletion_queue: vec![],
        }
    }

    #[instrument(skip(self))]
    fn insert(&mut self, referece: BufferReference) -> BufferReferenceId {
        let id = self.storage.insert(referece);
        //println!("create new buffer Reference: {:?}", id);
        return id;
    }

    pub fn get(&self, id: &BufferReferenceId) -> Option<&BufferReference> {
        self.storage.get(id)
    }

    pub fn get_mut(&mut self, id: &BufferReferenceId) -> Option<&mut BufferReference> {
        self.storage.get_mut(id)
    }

    pub fn queue_for_deletion(&mut self, id: &BufferReferenceId) {
        self.deletion_queue.push(*id);
    }

    pub fn get_deletion_entries(&mut self) -> Vec<BufferReferenceId> {
        std::mem::take(&mut self.deletion_queue)
    }

    pub fn delete(&mut self, id: &BufferReferenceId) -> bool {
        //println!("deleting buffer Reference: {:?}", id);
        self.storage.delete(id)
    }

    pub fn get_reference(&self, id: u32) -> Option<(BufferReferenceId, &BufferReference)> {
        self.storage.get_reference(id)
    }
}

// Struct used for ordering by size
#[derive(Debug, Eq, PartialEq)]
struct OrderedIndex<TI, TV> {
    index: TI,
    value: TV,
}

impl<TI, TV> OrderedIndex<TI, TV> {
    fn new(index: TI, value: TV) -> Self {
        OrderedIndex { index, value }
    }
}

// Implementing Ord and PartialOrd for OrderedIndex so it can be stored in BTreeSet
impl<TI: Ord, TV: Ord> Ord for OrderedIndex<TI, TV> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.value
            .cmp(&other.value)
            .then_with(|| self.index.cmp(&other.index))
    }
}

impl<TI: PartialOrd + Ord, TV: PartialOrd + Ord> PartialOrd for OrderedIndex<TI, TV> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Cache of all free CachedBuffers
#[derive(Debug)]
pub(crate) struct BufferCacheStorage {
    storage: StorageOptional<CachedBuffer, CachedBufferId>,
    order: BTreeSet<OrderedIndex<u32, u64>>, //contains a ordered list of the currently free buffers in the storage
    //(a buffer may be free if it is currently not used, this does not mean that it was deleted. a deleted buffer was complely removed from the storage and droped)
    buffer_counter: u32,       //total number of buffers created
    buffer_reuse_counter: u32, //total number of buffers created
    buffer_memory: u64,        //total memory allocated
    buffer_memory_free: u64,   //total memory in buffers btree map
    max_memory_allowed: u64,
}

impl BufferCacheStorage {
    pub fn new() -> Self {
        return Self {
            storage: StorageOptional::new(),
            order: BTreeSet::new(),
            buffer_counter: 0,
            buffer_reuse_counter: 0,
            buffer_memory: 0,
            buffer_memory_free: 0,
            max_memory_allowed: 0,
        };
    }

    //creats a Buffer, expect that it will be used and not be part of free memory
    #[instrument(skip(self, dev))]
    fn create_buffer(&mut self, dev: &WgpuDevice, size: u64, command_id: u32) -> CachedBufferId {
        let buffer = wgpu_functions::create_buffer(dev, size);
        let mut buffer = CachedBuffer::new(buffer);
        buffer.last_used_counter = command_id;
        let id = self.storage.insert(buffer);
        self.buffer_memory += size;
        self.buffer_counter += 1;

        return id;
    }

    #[instrument(skip(self))]
    pub fn delete_buffer(&mut self, id: &CachedBufferId) {
        let value = self.storage.delete_move(id);
        if let Some(val) = value {
            if self
                .order
                .remove(&OrderedIndex::new(id.id(), val.buffer.size()))
            {
                self.buffer_memory_free -= val.buffer.size();
            }
            self.buffer_memory -= val.buffer.size();
        }
    }

    pub fn get_buffer(&self, id: &CachedBufferId) -> Option<&CachedBuffer> {
        self.storage.get(id)
    }

    //will not delete the buffer, but mark it free
    #[instrument(skip(self))]
    pub fn free_buffer(&mut self, id: &CachedBufferId) {
        let buffer: Option<&mut CachedBuffer> = self.storage.get_mut(id);
        if let Some(buffer) = buffer {
            if buffer.is_free == false {
                //the buffer is currently not free -> add it into the free order list
                self.order
                    .insert(OrderedIndex::new(id.id(), buffer.buffer.size()));
                buffer.is_free = true;
                self.buffer_memory_free += buffer.buffer.size()
            }
        }
    }

    //will not create a buffer, but mark the buffer as used
    #[instrument(skip(self))]
    pub fn use_buffer(&mut self, id: &CachedBufferId, command_id: u32) {
        let buffer: Option<&mut CachedBuffer> = self.storage.get_mut(id);
        if let Some(buffer) = buffer {
            if buffer.is_free == true {
                //the buffer is currently free -> remove it from the free order list
                self.order
                    .remove(&OrderedIndex::new(id.id(), buffer.buffer.size()));
                buffer.is_free = false;
                buffer.last_used_counter = command_id;
                self.buffer_reuse_counter += 1;
                //println!("use_buffer, remove buffer from free buffers");
                self.buffer_memory_free -= buffer.buffer.size()
            }
        }
    }

    // //will save the currentl free buffers and not free buffers
    // pub fn store_usage(&mut self){
    //     for b in self.storage.iter_mut_option(){
    //         b.stored_free  = b.is_free;
    //     }
    // }

    // //will reset all buffers usage to last storage usage
    // pub fn reset_usage(&mut self){
    //     for (id, b) in self.storage.enumerate_mut_option(){
    //         if b.stored_free && !b.is_free{
    //             self.order.insert(OrderedIndex::new(id.id(), b.buffer.size()));
    //             b.is_free = true;
    //             self.buffer_memory_free += b.buffer.size()
    //         }
    //         else if !b.stored_free  && b.is_free{
    //             self.order.remove(&OrderedIndex::new(id.id(), b.buffer.size()));
    //             b.is_free = false;
    //             self.buffer_memory_free -= b.buffer.size()
    //         }
    //     }
    // }

    //the length, this buffer should be used for(if a buffer is only used temporary we may use a way bigger buffer for just one command)
    fn max_cached_size(size: u64, length: u32) -> u64 {
        let length = (length + 1).min(100);
        let i = (300 / (length * length * length)).min(64).max(1) as u64;

        const TRANSITION_POINT: u64 = 1000 * 1024;
        return size + (i * size * TRANSITION_POINT / (TRANSITION_POINT + size));
    }

    //will try to find a free buffer in the cache, or create a new one
    #[instrument(skip(self, dev))]
    pub fn search_buffer(
        &mut self,
        dev: &WgpuDevice,
        size: u64,
        command_id: u32,
        length: u32,
    ) -> CachedBufferId {
        //println!("search buffer: size: {size}");
        let max_size = BufferCacheStorage::max_cached_size(size, length);

        if dev.configuration.use_cache {
            let mut buffer_found = None;
            for id in self.order.range(OrderedIndex::new(0, size)..) {
                if id.value < size {
                    panic!("Did not expect size to be smaller, than key");
                }

                if id.value > max_size {
                    break;
                }
                buffer_found = Some(id);
            }

            //remove this buffer from free memory:
            if let Some(buffer_found) = buffer_found {
                if let Some((reference, _)) = self.storage.get_reference(buffer_found.index) {
                    //println!("search buffer: found free buffer, using: {:?}, size: {:?}", reference, buffer.buffer.size());
                    self.use_buffer(&reference, command_id);
                    return reference;
                }
            }
        }
        return self.create_buffer(dev, size, command_id);
    }

    pub fn max_memory_allowed(&self) -> u64 {
        self.max_memory_allowed
    }

    pub fn set_max_memory_allowed(&mut self, max_memory_allowed: u64) {
        self.max_memory_allowed = max_memory_allowed;
    }

    pub(crate) fn buffer_memory(&self) -> u64 {
        self.buffer_memory
    }

    pub(crate) fn buffer_reuse_counter(&self) -> u32 {
        self.buffer_reuse_counter
    }

    pub(crate) fn buffer_counter(&self) -> u32 {
        self.buffer_counter
    }
}

/// Cache of all available CachedBindGroups
#[derive(Debug)]
pub(crate) struct BindgroupCacheStorage {
    storage: StorageOptional<CachedBindgroup, CachedBindgroupId>,
    bindgroups: HashMapMulti<CachedBindgroupInput, CachedBindgroupId>, //all bindgroups based on input buffers
    bindgroups_full: HashMap<CachedBindgroupFull, CachedBindgroupId>, //all bindgroups based on input und dest buffers
    bindgroup_counter: u32,
    cached_bindgroup_use_counter: u32,
}

impl BindgroupCacheStorage {
    fn new() -> Self {
        return Self {
            storage: StorageOptional::new(),
            bindgroups: HashMapMulti::new(),
            bindgroups_full: HashMap::new(),
            bindgroup_counter: 0,
            cached_bindgroup_use_counter: 0,
        };
    }

    #[instrument(skip(self, keep))]
    fn retain_bindgroups(&mut self, mut keep: impl FnMut(&CachedBindgroup) -> bool) {
        self.storage.retain_mut(|(id, bg)| {
            let keep = keep(bg);

            if !keep {
                let id = id.clone();
                let buf_reference_input_full = bg.buffer.clone();
                self.bindgroups
                    .remove_mapping(buf_reference_input_full.1.clone(), &id);
                self.bindgroups_full.remove(&buf_reference_input_full);
            }
            return keep;
        });
    }

    pub fn get_bindgroup(&self, id: &CachedBindgroupId) -> Option<&CachedBindgroup> {
        self.storage.get(id)
    }

    fn get_bindgroup_reference_by_description(
        &self,
        bindgroup_d: &CachedBindgroupFull,
    ) -> Option<&CachedBindgroupId> {
        self.bindgroups_full.get(bindgroup_d)
    }

    fn get_bindgroup_reference_by_description_input(
        &self,
        bindgroup_d: &CachedBindgroupInput,
    ) -> &Vec<CachedBindgroupId> {
        self.bindgroups.get(bindgroup_d)
    }

    fn enumerate_bindgroup_by_description_input(
        &self,
        bindgroup_d: &CachedBindgroupInput,
    ) -> impl Iterator<Item = (CachedBindgroupId, &CachedBindgroup)> {
        self.get_bindgroup_reference_by_description_input(bindgroup_d)
            .iter()
            .filter_map(|c| Some((c.clone(), self.get_bindgroup(c)?)))
    }

    pub(crate) fn bindgroup_counter(&self) -> u32 {
        self.bindgroup_counter
    }

    pub(crate) fn cached_bindgroup_use_counter(&self) -> u32 {
        self.cached_bindgroup_use_counter
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
    fn new(size: u32) -> Self {
        Self {
            last_buffer_mappings: FixedSizeQueue::new(size as usize),
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
            log::debug!(
                "reuse mapping: {index}, hash: {hash}, mappings: {}",
                self.last_buffer_mappings.deque.len()
            );
            self.current_buffer_mapping = self.last_buffer_mappings.deque.remove(index);
        } else {
            log::debug!(
                "create new mapping: hash: {hash}, mappings: {}",
                self.last_buffer_mappings.deque.len()
            );
            self.current_buffer_mapping = Some(CachedBufferMappings::new(hash));
        }
    }

    pub(crate) fn get_buffer(&mut self, pipeline: PipelineType) -> Option<CachedBufferId> {
        if let Some(mapping) = &self.current_buffer_mapping {
            if let Some(mapping) = &mapping.data.get(self.current_index as usize) {
                if mapping.pipeline == pipeline {
                    return Some(mapping.used_buffer.clone());
                }
            }
        } else {
            panic!("expected current buffer to be set");
        }
        return None;
    }

    ///Stores, that at the provided buffer was used
    pub(crate) fn add_buffer(&mut self, buffer: CachedBufferId, pipeline: PipelineType) {
        if let Some(mapping) = &mut self.current_buffer_mapping {
            let data = CachedBufferMapping::new(pipeline, buffer);
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
    pub(crate) used_buffer: CachedBufferId,
}

impl CachedBufferMapping {
    fn new(pipeline: PipelineType, used_buffer: CachedBufferId) -> Self {
        Self {
            pipeline,
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
