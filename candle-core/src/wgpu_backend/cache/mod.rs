mod bindgroup_layout;
mod buffer_mapping;
mod buffer_reference;
mod cached_bindgroup;
mod cached_buffer;
mod shader;

pub use bindgroup_layout::*;
use buffer_mapping::*;
pub use buffer_reference::*;
pub use cached_bindgroup::*;
pub use cached_buffer::*;
pub use shader::*;

use tracing::{instrument, span};

use super::queue_buffer::PipelineReference;
use super::wgpu_functions;
use super::{
    util::{Reference, ReferenceTrait, ToU64},
    WgpuDevice,
};

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

#[derive(Debug, Clone, PartialEq, Eq, Hash, std::marker::Copy)]
#[cfg_attr(
    any(feature = "wgpu_debug_serialize", feature = "wgpu_debug"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum BindgroupAlignment {
    Aligned2,
    Aligned4,
    Aligned8,
    Aligned16,
}

impl BindgroupAlignment {
    pub fn get_index(&self) -> usize {
        match self {
            BindgroupAlignment::Aligned2 => 0,
            BindgroupAlignment::Aligned4 => 1,
            BindgroupAlignment::Aligned8 => 2,
            BindgroupAlignment::Aligned16 => 3,
        }
    }
}

impl From<crate::DType> for BindgroupAlignment {
    fn from(value: crate::DType) -> Self {
        match value {
            crate::DType::U8 => panic!("alignment not supported"),
            crate::DType::F8E4M3 => panic!("alignment not supported"),
            crate::DType::U32 => BindgroupAlignment::Aligned4,
            crate::DType::I64 => BindgroupAlignment::Aligned8,
            crate::DType::BF16 => BindgroupAlignment::Aligned4,
            crate::DType::F16 => BindgroupAlignment::Aligned4,
            crate::DType::F32 => BindgroupAlignment::Aligned4,
            crate::DType::F64 => BindgroupAlignment::Aligned8,
            crate::DType::I16 => panic!("alignment not supported"),
            crate::DType::I32 => panic!("alignment not supported"),
            crate::DType::F6E2M3 => panic!("alignment not supported"),
            crate::DType::F6E3M2 => panic!("alignment not supported"),
            crate::DType::F4 => panic!("alignment not supported"),
            crate::DType::F8E8M0 => panic!("alignment not supported"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(
    any(feature = "wgpu_debug_serialize", feature = "wgpu_debug"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum BindgroupInputBase<T> {
    Bindgroup0(BindgroupAlignmentLayout), //
    Bindgroup1(
        #[cfg_attr(
            any(feature = "wgpu_debug_serialize", feature = "wgpu_debug"),
            serde(skip)
        )]
        T,
        BindgroupAlignmentLayout,
    ), //input1, dest_alignment, input1_alignment
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
        BindgroupAlignmentLayout,
    ), //input1, input2, alignment dest
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
        BindgroupAlignmentLayout,
    ), //input1, input2, input3
}

impl<T: Clone> BindgroupInputBase<T> {
    pub fn get_input1(&self) -> Option<&T> {
        match self {
            BindgroupInputBase::Bindgroup0(_) => None,
            BindgroupInputBase::Bindgroup1(input1, _) => Some(input1),
            BindgroupInputBase::Bindgroup2(input1, _, _) => Some(input1),
            BindgroupInputBase::Bindgroup3(input1, _, _, _) => Some(input1),
        }
    }

    pub fn get_input2(&self) -> Option<&T> {
        match self {
            BindgroupInputBase::Bindgroup0(_) => None,
            BindgroupInputBase::Bindgroup1(_, _) => None,
            BindgroupInputBase::Bindgroup2(_, input2, _) => Some(input2),
            BindgroupInputBase::Bindgroup3(_, input2, _, _) => Some(input2),
        }
    }

    pub fn get_input3(&self) -> Option<&T> {
        match self {
            BindgroupInputBase::Bindgroup0(_) => None,
            BindgroupInputBase::Bindgroup1(_, _) => None,
            BindgroupInputBase::Bindgroup2(_, _, _) => None,
            BindgroupInputBase::Bindgroup3(_, _, input3, _) => Some(input3),
        }
    }

    pub fn fold<TOut>(&self, mut f: impl FnMut(&T) -> TOut) -> BindgroupInputBase<TOut> {
        match self {
            super::cache::BindgroupInputBase::Bindgroup0(alignment) => {
                super::cache::BindgroupInputBase::Bindgroup0(*alignment)
            }
            super::cache::BindgroupInputBase::Bindgroup1(buf1, alignment) => {
                super::cache::BindgroupInputBase::Bindgroup1(f(buf1), *alignment)
            }
            super::cache::BindgroupInputBase::Bindgroup2(buf1, buf2, alignment) => {
                super::cache::BindgroupInputBase::Bindgroup2(f(buf1), f(buf2), *alignment)
            }
            super::cache::BindgroupInputBase::Bindgroup3(buf1, buf2, buf3, alignment) => {
                super::cache::BindgroupInputBase::Bindgroup3(f(buf1), f(buf2), f(buf3), *alignment)
            }
        }
    }

    pub fn fold_owned<TOut>(&self, mut f: impl FnMut(T) -> TOut) -> BindgroupInputBase<TOut> {
        self.fold(|k| f(k.clone()))
    }
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
        BindgroupFullBase(dest, input)
    }

    pub(crate) fn get_dest(&self) -> &T {
        &self.0
    }

    pub(crate) fn get_input(&self) -> &BindgroupInputBase<T> {
        &self.1
    }
}

pub type CachedBindgroupInput = BindgroupInputBase<CachedBufferId>;
pub type CachedBindgroupFull = BindgroupFullBase<CachedBufferId>;
pub type BindgroupReferenceInput = BindgroupInputBase<BufferReferenceId>;
pub type BindgroupReferenceFull = BindgroupFullBase<BufferReferenceId>;

#[derive(Debug, Eq, PartialEq, Clone)]
#[cfg_attr(
    any(feature = "wgpu_debug_serialize", feature = "wgpu_debug"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct AverageBufferInfo {
    pub size: u32,
    pub is_free: bool,
    pub count: u32,
}

#[derive(Debug, Eq, PartialEq, Clone)]
#[cfg_attr(
    any(feature = "wgpu_debug_serialize", feature = "wgpu_debug"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct DebugBufferUsage {
    pub memory_alloc: u64,

    pub memory_free: u64,

    pub buffers: Vec<AverageBufferInfo>, // (size, is_free) -> count

    /// Identifier of the command buffer / queue pass this snapshot belongs to.
    pub command_buffer_id: u32,
}

#[derive(Debug)]
pub struct ModelCache {
    pub(crate) buffer_reference: BufferReferenceStorage,
    pub(crate) buffers: BufferCacheStorage,
    pub(crate) bindgroups: BindgroupCacheStorage,
    pub(crate) mappings: BufferMappingCache,
    pub(crate) shader: ShaderCache,

    pub(crate) unary_inplace_counter: u32,
    pub(crate) binary_inplace_counter: u32,
    pub(crate) copy_inplace_counter: u32,

    pub(crate) max_memory_size: u64,

    #[cfg(feature = "wgpu_debug")]
    pub(crate) debug: std::collections::HashMap<
        super::device::DebugPipelineRecording,
        super::device::DebugPipelineRecording,
    >, //stores all queed commands (e.g. used to test performance in the browser so we can measure the performance of each unnique operation alone)

    #[cfg(feature = "wgpu_debug")]
    pub(crate) debug_buffer_info: Vec<DebugBufferUsage>,

    #[cfg(feature = "wgpu_debug")]
    pub(crate) full_recording: super::debug_info::DebugRecordingWithData,
}

impl ModelCache {
    pub fn new(mapping_size: u32, max_memory_size: u64) -> Self {
        Self {
            buffer_reference: BufferReferenceStorage::new(),
            buffers: BufferCacheStorage::new(),
            bindgroups: BindgroupCacheStorage::new(),
            mappings: BufferMappingCache::new(mapping_size),
            shader: ShaderCache::new(),

            unary_inplace_counter: 0,
            binary_inplace_counter: 0,
            copy_inplace_counter: 0,
            max_memory_size,
            #[cfg(feature = "wgpu_debug")]
            debug: std::collections::HashMap::new(),
            #[cfg(feature = "wgpu_debug")]
            debug_buffer_info: Vec::new(),
            #[cfg(feature = "wgpu_debug")]
            full_recording: super::debug_info::DebugRecordingWithData {
                recordings: Vec::new(),
                should_record: false,
            },
        }
    }

    #[instrument(skip(self, size))]
    pub fn create_buffer_reference<T: ToU64>(
        &mut self,
        size: T,
        referenced_by_candle_storage: bool,
    ) -> BufferReferenceId {
        let size = size.to_u64();
        if size > self.max_memory_size {
            panic!(
                "tried to create too large a buffer: {}, max: {}",
                size, self.max_memory_size
            );
        }

        let buffer_reference = BufferReference::new(size, referenced_by_candle_storage);
        self.buffer_reference.insert(buffer_reference)
    }

    #[instrument(skip(self, dev, data))]
    pub fn create_buffer_reference_init<T: bytemuck::Pod>(
        &mut self,
        dev: &WgpuDevice,
        data: &[T],
        referenced_by_candle_storage: bool,
    ) -> BufferReferenceId {
        let data = bytemuck::cast_slice(data);
        let length = data.len().div_ceil(4) * 4;

        if length as u64 > self.max_memory_size {
            panic!(
                "tried to create_init too large a buffer: {}, max: {}",
                length, self.max_memory_size
            );
        }

        let buffer = self
            .buffers
            .search_buffer(dev, length as u64, length as u64, 0, u32::MAX - 1); //TODO use exact size?

        if data.len() % 4 == 0 {
            dev.queue
                .write_buffer(self.buffers.get_buffer(&buffer).unwrap().buffer(), 0, data);
        } else {
            let mut padded = Vec::with_capacity(length);
            padded.extend_from_slice(data); // Copy original data
            padded.resize(length, 0); // Fill the rest with zeros
            dev.queue.write_buffer(
                self.buffers.get_buffer(&buffer).unwrap().buffer(),
                0,
                &padded,
            );
        }

        let buffer_reference =
            BufferReference::new_with_storage(length as u64, buffer, referenced_by_candle_storage);
        self.buffer_reference.insert(buffer_reference)
    }

    /// returns, wheter we should stop the command_queue and delete not used buffers
    #[instrument(skip(self))]
    pub fn should_delete_unused(&mut self) -> bool {
        let current_memory = self.buffers.buffer_memory();
        let memory_margin = self.buffers.max_memory_allowed();
        if current_memory > memory_margin {
            return self.buffers.has_free_buffers();
        }
        false
    }

    #[instrument(skip(self))]
    pub fn remove_unused(&mut self) -> bool {
        let remove_older_then = *self
            .mappings
            .last_command_indexes
            .deque
            .front()
            .unwrap_or(&u32::MAX);
        let current_memory = self.buffers.buffer_memory();
        let memory_margin = self.buffers.max_memory_allowed();
        let delete_until_margin = (memory_margin * 4) / 5;
        const CHECK_MEMORY_EVERY: u32 = 10;

        if current_memory <= memory_margin
            && !self
                .buffers
                .inc_remove_test_counter()
                .is_multiple_of(CHECK_MEMORY_EVERY)
        {
            return false;
        }

        //remove buffers, that
        // 1. were not used for a long time
        // 2. have a big memory diff (the actual buffer size vs the average size the buffer is used with)
        let mut check_bindgroups = false;
        if current_memory > memory_margin {
            log::debug!(
                "deleting buffers: ({}) current {current_memory}/{memory_margin}",
                self.buffers.get_buffer_count()
            );

            tracing::info!(
                "deleting buffers: ({}) current {current_memory}/{memory_margin}",
                self.buffers.get_buffer_count()
            );

            //every entry in self.buffers.order will be free and can be potentially deleted
            //this is ordered from small to big.

            let mut buffers = self.buffers.get_free_buffers();

            buffers.sort_by_key(|f| f.1);

            for (id, last_used_counter) in buffers {
                if last_used_counter > remove_older_then
                    && self.buffers.buffer_memory() <= delete_until_margin
                {
                    break;
                }
                check_bindgroups = true;
                self.buffers.delete_buffer(&id);
            }
            let current_memory = self.buffers.buffer_memory();
            log::debug!(
                "after deleting: ({}) current {current_memory}/{}",
                self.buffers.get_buffer_count(),
                self.buffers.max_memory_allowed()
            );

            tracing::info!(
                "after deleting: ({}) current {current_memory}/{}",
                self.buffers.get_buffer_count(),
                self.buffers.max_memory_allowed()
            );
        }

        //remove bindgroups:
        //1. if we removed a buffer, we should also remove the bindgroup
        //2. bindgroups that werent used for a long time may be deleted

        if check_bindgroups {
            self.bindgroups.retain_bindgroups(|bindgroup| {
                let span1 = span!(tracing::Level::INFO, "Calc sould keep bindgroup");
                let _enter1 = span1.enter();

                let check_buffer =
                    |buffer_reference| self.buffers.get_buffer(buffer_reference).is_some();

                let is_valid = check_buffer(bindgroup.buffer().get_dest())
                    && match &bindgroup.buffer().get_input() {
                        BindgroupInputBase::Bindgroup0(_) => true,
                        BindgroupInputBase::Bindgroup1(v1, _) => check_buffer(v1),
                        BindgroupInputBase::Bindgroup2(v1, v2, _) => {
                            check_buffer(v1) && check_buffer(v2)
                        }
                        BindgroupInputBase::Bindgroup3(v1, v2, v3, _) => {
                            check_buffer(v1) && check_buffer(v2) && check_buffer(v3)
                        }
                    };
                drop(_enter1);
                //check if all buffers for this bindgroup still exist!
                if !is_valid {
                    return false;
                }

                true
            });
        }
        false
    }

    #[instrument(skip(self, dev, bindgroup_reference, command_id, pipeline))]
    pub(crate) fn get_bind_group(
        &mut self,
        dev: &WgpuDevice,
        bindgroup_reference: &BindgroupReferenceFull,
        pipeline: PipelineReference,
        command_id: u32,
    ) -> CachedBindgroupId {
        fn check_buffer_reference(
            cache: &mut ModelCache,
            bindgroup_reference: &BindgroupReferenceFull,
            pipeline: PipelineReference,
        ) {
            let check_buffer = |buffer_reference_id| {
                if let Some(buffer_reference) = cache.buffer_reference.get(&buffer_reference_id) {
                    if !buffer_reference.cached_buffer_id().is_valid() {
                        panic!("input buffer {:?}({:?}) in {:?} had no cached_storage set for pipeline {:?}", buffer_reference,buffer_reference_id, bindgroup_reference, pipeline);
                    } else if cache
                        .buffers
                        .get_buffer(buffer_reference.cached_buffer_id())
                        .is_none()
                    {
                        if let Some(buffer_reference) =
                            cache.buffer_reference.get(&buffer_reference_id)
                        {
                            panic!("input buffer {:?}({:?}) in {:?} had no cached_storage set to {:?} widch could not be found for pipeline {:?}", buffer_reference,buffer_reference_id, bindgroup_reference,buffer_reference.cached_buffer_id(), pipeline);
                        }
                    }
                } else if let Some(val) = cache
                    .buffer_reference
                    .get_reference(buffer_reference_id.id())
                {
                    panic!("Reference {:?} inside Bindgroup {:?} invalid for pipeline {:?}, Reference was replaced, current: {:?}", buffer_reference_id, bindgroup_reference, pipeline, val.0);
                } else {
                    panic!("Reference {:?} inside Bindgroup {:?} invalid for pipeline {:?} (Reference was deleted)", buffer_reference_id, bindgroup_reference, pipeline);
                }
            };

            bindgroup_reference.1.fold_owned(|v| {
                check_buffer(v);
            });
        }

        check_buffer_reference(self, bindgroup_reference, pipeline.clone());

        fn get_storage(cache: &ModelCache, id: &BufferReferenceId) -> CachedBufferId {
            *cache.buffer_reference.get(id).unwrap().cached_buffer_id()
        }

        fn get_buffer_referece_key(
            cache: &ModelCache,
            dest_buffer: CachedBufferId,
            bindgroup_reference: &BindgroupReferenceFull,
        ) -> CachedBindgroupFull {
            BindgroupFullBase(
                dest_buffer,
                bindgroup_reference.1.fold(|v| get_storage(cache, v)),
            )
        }

        let buf_dest_id = bindgroup_reference.get_dest();
        let buf_dest_dur;
        let buf_dest_size;
        let buf_dest_cached_id;
        {
            let buf_dest_reference = self.buffer_reference.get(buf_dest_id).unwrap();
            buf_dest_cached_id = *buf_dest_reference.cached_buffer_id();
            buf_dest_size = buf_dest_reference.size();
            buf_dest_dur = buf_dest_reference.last_used() - buf_dest_reference.first_used();
            if buf_dest_reference.last_used() < buf_dest_reference.first_used() {
                panic!("buffer {:?}({:?})", buf_dest_reference, buf_dest_id);
            }
        }
        let minimum_size = buf_dest_size; //minimum size needed
        let mut optimal_size = buf_dest_size; //size we use if we need to create a new buffer (may be a little bit bigger for growing buffers)

        if dev.configuration.use_cache {
            let current_mapping_index = self.mappings.current_index;
            let current_mapping = self.mappings.get_current_mapping();

            let buffer_already_set = self.buffers.get_buffer(&buf_dest_cached_id).is_some();

            //search in buffer mapping, and use the same buffer as last run:
            if let Some(buffer_mapping) =
                current_mapping.get_buffer_mapping(&pipeline, current_mapping_index)
            {
                if buffer_already_set {
                    let bindgroup_inputs =
                        get_buffer_referece_key(self, buf_dest_cached_id, bindgroup_reference);

                    self.mappings.reuse_buffer(buf_dest_size);

                    if let Some(bg) = self
                        .bindgroups
                        .get_bindgroup_reference_by_description(&bindgroup_inputs)
                        .cloned()
                    {
                        self.bindgroups.cached_bindgroup_use_counter_inc();
                        return bg;
                    } else {
                        //create new bindgroup:
                        let bindgroup_reference =
                            get_buffer_referece_key(self, buf_dest_cached_id, bindgroup_reference);
                        let bindgroup_id = self.create_bindgroup(dev, bindgroup_reference);
                        return bindgroup_id;
                    }
                } else {
                    let buffer_id = buffer_mapping.used_buffer;
                    let buffer: Option<&CachedBuffer> = self.buffers.get_buffer(&buffer_id);
                    let buffer_last_size = buffer_mapping.last_size;

                    if let Some(buffer) = buffer {
                        if buffer.is_free() {
                            if buffer.buffer().size() >= minimum_size {
                                tracing::info!("mapping: use buffer {:?}", buffer_id);
                                let buf_dest_reference =
                                    self.buffer_reference.get_mut(buf_dest_id).unwrap();
                                //use this buffer for the buffer reference:
                                buf_dest_reference.set_cached_buffer_id(buffer_id);
                                self.buffers.use_buffer(&buffer_id, command_id);

                                self.mappings.reuse_buffer(buf_dest_size);

                                //reuse a bindgroup, if we could find one:
                                let bindgroup_inputs =
                                    get_buffer_referece_key(self, buffer_id, bindgroup_reference);
                                if let Some(bg) = self
                                    .bindgroups
                                    .get_bindgroup_reference_by_description(&bindgroup_inputs)
                                    .cloned()
                                {
                                    self.bindgroups.cached_bindgroup_use_counter_inc();
                                    return bg;
                                } else {
                                    //create new bindgroup:
                                    let bindgroup_reference = get_buffer_referece_key(
                                        self,
                                        buffer_id,
                                        bindgroup_reference,
                                    );
                                    let bindgroup_id =
                                        self.create_bindgroup(dev, bindgroup_reference);
                                    return bindgroup_id;
                                }
                            } else {
                                tracing::info!(
                                    "mapping: buffer was not big enaugh {:?}",
                                    buffer_id
                                );
                            }
                        } else {
                            tracing::info!("mapping: buffer was not free {:?}", buffer_id);
                        }
                    } else {
                        tracing::info!("mapping: buffer was deleted {:?}", buffer_id);
                    }
                    //replace buffer:
                    if optimal_size > buffer_last_size {
                        let delta_size = optimal_size - buffer_last_size;
                        let new_size = optimal_size
                            + delta_size * self.mappings.get_current_mapping_count() as u64;

                        //We try to determine a good new buffer size for constantly growing buffers,
                        //but do not use more than 2 * optimal_size:
                        let new_size = new_size.min(2 * optimal_size);

                        if new_size != optimal_size {
                            tracing::info!(
                                "increase required size: {} -> {}",
                                optimal_size,
                                new_size
                            );
                        }
                        optimal_size = new_size;
                    }
                }
            }

            //the destination buffer of this bindgroup already has a buffer set
            if buffer_already_set {
                let bindgroup_inputs =
                    get_buffer_referece_key(self, buf_dest_cached_id, bindgroup_reference);
                if let Some(bg) = self
                    .bindgroups
                    .get_bindgroup_reference_by_description(&bindgroup_inputs)
                    .cloned()
                {
                    self.bindgroups.cached_bindgroup_use_counter_inc();
                    self.mappings
                        .add_new_buffer(buf_dest_cached_id, pipeline, buf_dest_size);
                    return bg;
                }
            }
        }

        let buf_dest_reference = self.buffer_reference.get_mut(buf_dest_id).unwrap();

        //create new buffer, if buffer was not already set:
        let dest_buffer_id;
        if buf_dest_reference.cached_buffer_id().is_valid() {
            //this buffer reference already has a buffer connected,use this buffer
            dest_buffer_id = *buf_dest_reference.cached_buffer_id();
        } else {
            //create a new buffer
            dest_buffer_id = self.buffers.search_buffer(
                dev,
                minimum_size,
                optimal_size,
                command_id,
                buf_dest_dur,
            );
            //use this buffer for the buffer reference:
            buf_dest_reference.set_cached_buffer_id(dest_buffer_id);
        }

        let bindgroup_inputs = get_buffer_referece_key(self, dest_buffer_id, bindgroup_reference);

        if dev.configuration.use_cache {
            self.mappings
                .add_new_buffer(dest_buffer_id, pipeline, buf_dest_size);
        }

        if let Some(bg) = self
            .bindgroups
            .get_bindgroup_reference_by_description(&bindgroup_inputs)
            .cloned()
        {
            self.bindgroups.cached_bindgroup_use_counter_inc();
            bg
        } else {
            //create new bindgroup:
            let bindgroup_reference =
                get_buffer_referece_key(self, dest_buffer_id, bindgroup_reference);
            let bindgroup_id = self.create_bindgroup(dev, bindgroup_reference);
            bindgroup_id
        }
    }

    //creats a Bindgroup
    #[instrument(skip(self, dev, bindgroup_d))]
    fn create_bindgroup(
        &mut self,
        dev: &WgpuDevice,
        bindgroup_d: CachedBindgroupFull,
    ) -> CachedBindgroupId {
        let bindgroup = wgpu_functions::create_bindgroup(dev, bindgroup_d.clone(), self);
        let bindgroup = CachedBindgroup::new(bindgroup, bindgroup_d.clone());

        self.bindgroups.insert_bindgroup(bindgroup)
    }
}
