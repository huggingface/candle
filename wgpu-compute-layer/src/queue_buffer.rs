/// Helpers for queuing compute dispatches and building virtual bindgroups.
///
/// The `QueueBuffer`/`QueueBufferInner` types provide utilities to:
/// - add parameters to a `MetaArray`,
/// - collect pipeline constants for the next pipeline call,
/// - create pipeline references and virtual bindgroup descriptors,
/// - enqueue the prepared dispatches.
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, MutexGuard};

use std::hash::Hash;

use crate::cache::BindGroupReference;
use crate::shader_loader;
use crate::wgpu_functions::KernelConstId;

use super::cache::{
    BindgroupAlignment, BindgroupAlignmentLayout, BindgroupInputBase, BufferReferenceId,
    CachedBindgroupId,
};
use super::util::{ObjectToIdMapper, ToU32};
use super::wgpu_functions::{ConstArray, MetaArray, ToKernelParameterMeta};

pub const MAX_DISPATCH_SIZE: u32 = 65535;

#[derive(Debug)]
pub(crate) enum MlQueue {
    Dispatch(MlQueueDispatch),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default, Copy)]
#[cfg_attr(feature = "wgpu_debug", derive(serde::Serialize, serde::Deserialize))]
pub struct OpIsInplaceable {
    pub input1_inplaceable: bool,
    pub input2_inplaceable: bool,
}

impl OpIsInplaceable {
    pub fn new() -> Self {
        Self {
            input1_inplaceable: false,
            input2_inplaceable: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(
    any(feature = "wgpu_debug_serialize", feature = "wgpu_debug"),
    derive(serde::Serialize, serde::Deserialize)
)]
/// Pipeline, pipeline constants, and in-place information.
pub struct PipelineReference {
    pub(crate) index: shader_loader::PipelineIndex,
    pub(crate) const_index: usize, // Index into an array with pipeline constants
    pub(crate) defines_index: usize,
    #[cfg_attr(
        any(feature = "wgpu_debug_serialize", feature = "wgpu_debug"),
        serde(skip)
    )]
    pub(crate) inplaceable: OpIsInplaceable,
}

impl PipelineReference {
    pub fn new(
        index: shader_loader::PipelineIndex,
        const_index: usize,
        defines_index: usize,
        inplaceable: OpIsInplaceable,
    ) -> Self {
        Self {
            index,
            const_index,
            defines_index,
            inplaceable,
        }
    }

    pub fn get_index(&self) -> shader_loader::PipelineIndex {
        self.index
    }
}

#[derive(Debug)]
pub(crate) struct MlQueueDispatch {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) z: u32,
    pub(crate) pipeline: PipelineReference,
    pub(crate) pipeline_cached: Option<Arc<wgpu::ComputePipeline>>,
    pub(crate) bindgroup: BindGroupReference,
    pub(crate) bindgroup_cached: Option<CachedBindgroupId>,
    pub(crate) meta: u32,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) meta_length: u32,
    pub(crate) workload_size: usize, // the total size needed to calculate; prevents queuing too many operations at once.
    #[cfg(feature = "wgpu_debug")]
    pub(crate) debug: Option<String>,
}

pub type DefineSymbol = usize;

#[derive(Debug)]
struct DefineMultiMap<V> {
    key_cache: ObjectToIdMapper<V>,
    id_to_key: Vec<V>,
}

impl<V: std::cmp::Eq + std::hash::Hash + std::clone::Clone> DefineMultiMap<V> {
    fn new() -> Self {
        Self {
            key_cache: ObjectToIdMapper::new(),
            id_to_key: Vec::new(),
        }
    }

    fn get_index(&mut self, key: &V) -> usize {
        let (index, is_new) = self.key_cache.get_or_insert(key);
        if is_new {
            self.id_to_key.push(key.to_owned())
        }
        index
    }
}

#[derive(Debug)]
pub(crate) struct DefinesCache {
    key_cache: DefineMultiMap<&'static str>,
    value_cache: DefineMultiMap<String>,
    defines_cache: DefineMultiMap<Vec<(DefineSymbol, DefineSymbol)>>,
}

impl DefinesCache {
    fn new() -> Self {
        Self {
            key_cache: DefineMultiMap::new(),
            value_cache: DefineMultiMap::new(),
            defines_cache: DefineMultiMap::new(),
        }
    }

    pub(crate) fn get_define(&self, index: usize) -> Vec<(&'static str, String)> {
        let test = &self.defines_cache.id_to_key[index];
        test.iter()
            .map(|c| {
                (
                    self.key_cache.id_to_key[c.0],
                    self.value_cache.id_to_key[c.1].clone(),
                )
            })
            .collect()
    }
}

/// Core fields that are frequently drained for flushing.
#[derive(Debug)]
pub(crate) struct QueueBufferCore {
    /// All queued commands.
    pub(crate) command_queue: Vec<MlQueue>,

    /// u32 `MetaArray` for parameters of kernels.
    pub(crate) meta_array: MetaArray,

    pub(crate) global_command_index: u32,
}

impl QueueBufferCore {
    pub fn get_meta(&self) -> &Vec<u32> {
        &self.meta_array.0
    }

    pub fn get_meta_mut(&mut self) -> &mut Vec<u32> {
        &mut self.meta_array.0
    }
}

/// Shared fields that should remain inside the locked area.
#[derive(Debug)]
pub(crate) struct QueueBufferShared {
    /// `ConstArray` used to store the pipeline constants for the next pipeline call.
    pub(crate) const_array: ConstArray,

    /// `DefinesArray` used to store the pipeline defines for the next pipeline call.
    pub(crate) defines_array: Vec<(DefineSymbol, DefineSymbol)>,

    /// `ConstArray` -> id mapper: maps a set of constants to a unique id.
    pub(crate) const_id_map: ObjectToIdMapper<ConstArray>,
    /// Id -> `ConstArray` mapping: maps a unique id to a set of constants.
    pub(crate) id_to_const_array: Vec<Vec<(&'static str, f64)>>,

    pub(crate) define_cache: DefinesCache,

    /// Current position inside the `MetaArray`.
    pub(crate) current_meta: u32,
}

/// A struct where all operations are cached.
#[derive(Debug)]
pub struct QueueBufferInner {
    pub(crate) core: QueueBufferCore,
    pub(crate) shared: QueueBufferShared,
}

impl QueueBufferInner {
    pub fn new(size: u32) -> Self {
        Self {
            core: QueueBufferCore {
                command_queue: vec![],
                meta_array: MetaArray::new(size),
                global_command_index: 1,
            },
            shared: QueueBufferShared {
                const_array: ConstArray::new(),
                defines_array: Vec::new(),
                const_id_map: ObjectToIdMapper::new(),
                define_cache: DefinesCache::new(),
                id_to_const_array: Vec::new(),
                current_meta: 0,
            },
        }
    }

    /// Access to the `MetaArray`.
    pub fn get_meta(&self) -> &Vec<u32> {
        self.core.get_meta()
    }

    /// Mutable access to the `MetaArray`.
    pub fn get_meta_mut(&mut self) -> &mut Vec<u32> {
        self.core.get_meta_mut()
    }

    /// Resets the `ConstArray` for the next pipeline.
    pub(crate) fn init(&mut self) {
        self.shared.const_array.0.clear();
        self.shared.defines_array.clear();
    }

    /// Drains all operations from the queue and returns them.
    pub(crate) fn drained(&mut self) -> QueueBufferCore {
        let core = QueueBufferCore {
            command_queue: std::mem::take(&mut self.core.command_queue),
            meta_array: std::mem::take(&mut self.core.meta_array),
            global_command_index: self.core.global_command_index,
        };

        self.core.global_command_index += core.command_queue.len() as u32;

        self.init();
        self.shared.current_meta = 0;

        core
    }

    fn finalize_internal_arrays(&mut self) -> (usize, usize) {
        let (index_const, is_new) = self
            .shared
            .const_id_map
            .get_or_insert(&self.shared.const_array);
        if is_new {
            self.shared
                .id_to_const_array
                .push(self.shared.const_array.to_vec())
        }

        self.shared
            .defines_array
            .sort_unstable_by(|a, b| a.0.cmp(&b.0));
        let index_defines = self
            .shared
            .define_cache
            .defines_cache
            .get_index(&self.shared.defines_array);

        self.init();
        (index_const, index_defines)
    }

    pub fn get_pipeline(
        &mut self,
        pipeline: impl Into<shader_loader::PipelineIndex>,
    ) -> PipelineReference {
        let (index_const, index_defines) = self.finalize_internal_arrays();
        PipelineReference::new(
            pipeline.into(),
            index_const,
            index_defines,
            OpIsInplaceable::new(),
        )
    }

    pub fn get_pipeline_inplace(
        &mut self,
        pipeline: impl Into<shader_loader::PipelineIndex>,
        inplaceable: OpIsInplaceable,
    ) -> PipelineReference {
        let (index_const, index_defines) = self.finalize_internal_arrays();
        PipelineReference::new(pipeline.into(), index_const, index_defines, inplaceable)
    }

    /// Adds the parameter `value` to the `MetaArray`.
    pub fn add<T: ToKernelParameterMeta>(&mut self, value: T) {
        self.core.meta_array.add(value);
    }

    pub fn add_const<K: Into<KernelConstId>, T: ToU32>(&mut self, key: K, value: T) {
        self.shared.const_array.insert(key.into(), value);
    }

    pub fn add_define(&mut self, key: &'static str, value: impl ToString) {
        let key = self.shared.define_cache.key_cache.get_index(&key);
        let value = self
            .shared
            .define_cache
            .value_cache
            .get_index(&value.to_string());
        self.shared.defines_array.push((key, value));
    }

    // Allows loading const debug info (for simulating calls).
    pub fn load_simulation_consts(&mut self, consts: Vec<Vec<(&'static str, f64)>>) {
        self.shared.id_to_const_array = consts;
        self.shared.const_id_map.next_id = self.shared.id_to_const_array.len();
    }
}

pub struct QueueBuffer<'a>(MutexGuard<'a, QueueBufferInner>);

impl<'a> QueueBuffer<'a> {
    pub fn new(inner: MutexGuard<'a, QueueBufferInner>) -> Self {
        QueueBuffer(inner)
    }

    /**************** Enqueue Helper: ****************/
    /// Enqueues a command with a workgroup size of 64 on the X dimension.
    pub fn enqueue_64(
        self,
        pipeline: PipelineReference,
        bind_group: BindGroupReference,
        length: u32,
        // workload_size is the total size needed to calculate. Needed so we do not queue too many operations at once.
        workload_size: usize,
    ) {
        self.enqueue_64_extra(
            pipeline,
            bind_group,
            length,
            workload_size,
            #[cfg(feature = "wgpu_debug")]
            None,
        )
    }

    /// Enqueues a command with a workgroup size of 64 on the X dimension.
    /// With extra debug info when `wgpu_debug` is enabled.
    pub fn enqueue_64_extra(
        self,
        pipeline: PipelineReference,
        bind_group: BindGroupReference,
        length: u32,
        // workload_size is the total size needed to calculate. Needed so we do not queue too many operations at once.
        workload_size: usize,
        #[cfg(feature = "wgpu_debug")] _debug: Option<String>,
    ) {
        self.enqueue_workgroups_extra(
            pipeline,
            bind_group,
            length.div_ceil(64),
            1,
            1,
            workload_size,
            #[cfg(feature = "wgpu_debug")]
            _debug,
        )
    }

    /// Enqueues a command with a workgroup size of 64 on the X dimension.
    /// If the length is greater than 65535, additional elements will be enqueued in the Y dimension.
    pub fn enqueue_64_big(
        self,
        pipeline: PipelineReference,
        bind_group: BindGroupReference,
        length: u32,
    ) {
        self.enqueue_64_big_extra(
            pipeline,
            bind_group,
            length,
            #[cfg(feature = "wgpu_debug")]
            None,
        )
    }

    /// Enqueues a command with a workgroup size of 64 on the X dimension.
    /// If the length is greater than 65535, additional elements will be enqueued in the Y dimension.
    /// With extra debug info when `wgpu_debug` is enabled.
    pub fn enqueue_64_big_extra(
        self,
        pipeline: PipelineReference,
        bind_group: BindGroupReference,
        length: u32,
        #[cfg(feature = "wgpu_debug")] _debug: Option<String>,
    ) {
        let id = length.div_ceil(64);
        let x = id.min(65535);
        let y = id.div_ceil(65535);
        self.enqueue_workgroups_extra(
            pipeline,
            bind_group,
            x,
            y,
            1,
            length as usize,
            #[cfg(feature = "wgpu_debug")]
            _debug,
        )
    }

    /// Enqueues a command with x, y and z dimensions.
    pub fn enqueue_workgroups(
        self,
        pipeline: PipelineReference,
        bind_group: BindGroupReference,
        x: u32,
        y: u32,
        z: u32,
        // workload_size is the total size needed to calculate. Needed so we do not queue too many operations at once.
        workload_size: usize,
    ) {
        self.enqueue_workgroups_extra(
            pipeline,
            bind_group,
            x,
            y,
            z,
            workload_size,
            #[cfg(feature = "wgpu_debug")]
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    /// Enqueues a command with x, y and z dimensions.
    /// With extra debug info when `wgpu_debug` is enabled.
    pub fn enqueue_workgroups_extra(
        mut self,
        pipeline: PipelineReference,
        bind_group: BindGroupReference,
        x: u32,
        y: u32,
        z: u32,
        // workload_size is the total size needed to calculate. Needed so we do not queue too many operations at once.
        workload_size: usize,
        #[cfg(feature = "wgpu_debug")] _debug: Option<String>,
    ) {
        if y > MAX_DISPATCH_SIZE || z > MAX_DISPATCH_SIZE || x > MAX_DISPATCH_SIZE {
            panic!(
                "can not queue y or z higher than 65535 x:{x}, y:{y}, z:{z}, pipeline: {:?}",
                pipeline
            );
        }
        #[cfg(not(target_arch = "wasm32"))]
        let meta_length = self.core.get_meta().len() as u32 - self.shared.current_meta;
        let q = MlQueue::Dispatch(MlQueueDispatch {
            x,
            y,
            z,
            pipeline: pipeline.clone(),
            pipeline_cached: None,
            bindgroup: bind_group,
            bindgroup_cached: None,
            meta: self.shared.current_meta,
            #[cfg(not(target_arch = "wasm32"))]
            meta_length,
            workload_size,
            #[cfg(feature = "wgpu_debug")]
            debug: _debug,
        });
        self.core.command_queue.push(q);
    }

    /**************** Virtual Bindgroups: ****************/
    pub fn create_bind_group_input0(
        buffer_dest: BufferReferenceId,
        alignment: BindgroupAlignment,
    ) -> BindGroupReference {
        let alignment = BindgroupAlignmentLayout::Bindgroup0(alignment);
        alignment.validate();
        BindGroupReference::new(buffer_dest, BindgroupInputBase::Bindgroup0(alignment))
    }

    pub fn create_bind_group_input1(
        buffer_dest: BufferReferenceId,
        buffer_input1: BufferReferenceId,
        alignment: BindgroupAlignment,
    ) -> BindGroupReference {
        Self::create_bind_group_input1_with_alignment(
            buffer_dest,
            buffer_input1,
            BindgroupAlignmentLayout::Bindgroup1(alignment, alignment),
        )
    }

    pub fn create_bind_group_input1_with_alignment(
        buffer_dest: BufferReferenceId,
        buffer_input1: BufferReferenceId,
        alignment: BindgroupAlignmentLayout,
    ) -> BindGroupReference {
        alignment.validate();
        BindGroupReference::new(
            buffer_dest,
            BindgroupInputBase::Bindgroup1(buffer_input1, alignment),
        )
    }

    pub fn create_bind_group_input2(
        buffer_dest: BufferReferenceId,
        buffer_input1: BufferReferenceId,
        buffer_input2: BufferReferenceId,
        alignment: BindgroupAlignment,
    ) -> BindGroupReference {
        Self::create_bind_group_input2_with_alignment(
            buffer_dest,
            buffer_input1,
            buffer_input2,
            BindgroupAlignmentLayout::Bindgroup2(alignment, alignment, alignment),
        )
    }

    pub fn create_bind_group_input2_with_alignment(
        buffer_dest: BufferReferenceId,
        buffer_input1: BufferReferenceId,
        buffer_input2: BufferReferenceId,
        alignment: BindgroupAlignmentLayout,
    ) -> BindGroupReference {
        alignment.validate();
        BindGroupReference::new(
            buffer_dest,
            BindgroupInputBase::Bindgroup2(buffer_input1, buffer_input2, alignment),
        )
    }

    pub fn create_bind_group_input3(
        buffer_dest: BufferReferenceId,
        buffer_input1: BufferReferenceId,
        buffer_input2: BufferReferenceId,
        buffer_input3: BufferReferenceId,
        alignment: BindgroupAlignment,
    ) -> BindGroupReference {
        Self::create_bind_group_input3_with_alignment(
            buffer_dest,
            buffer_input1,
            buffer_input2,
            buffer_input3,
            BindgroupAlignmentLayout::Bindgroup3(alignment, alignment, alignment, alignment),
        )
    }

    pub fn create_bind_group_input3_with_alignment(
        buffer_dest: BufferReferenceId,
        buffer_input1: BufferReferenceId,
        buffer_input2: BufferReferenceId,
        buffer_input3: BufferReferenceId,
        alignment: BindgroupAlignmentLayout,
    ) -> BindGroupReference {
        alignment.validate();
        BindGroupReference::new(
            buffer_dest,
            BindgroupInputBase::Bindgroup3(buffer_input1, buffer_input2, buffer_input3, alignment),
        )
    }
}

impl Deref for QueueBuffer<'_> {
    type Target = QueueBufferInner;

    fn deref(&self) -> &Self::Target {
        self.0.deref()
    }
}

impl DerefMut for QueueBuffer<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.deref_mut()
    }
}

#[test]
fn queue_buffer_const_mapping_roundtrip() {
    let mut qb = QueueBufferInner::new(256);

    // Insert a kernel constant
    qb.add_const(KernelConstId("K1"), 42u32);

    // Request a pipeline which should cause the const array to be stored
    let pipeline_ref = qb.get_pipeline(crate::PipelineIndex::new(
        crate::ShaderIndex::new(crate::LoaderIndex(0), 0),
        0,
    ));

    // id_to_const_array should have one entry and contain our constant
    assert_eq!(qb.shared.id_to_const_array.len(), 1);
    let entry = &qb.shared.id_to_const_array[0];
    assert_eq!(entry[0].0, "K1");
    assert_eq!(entry[0].1, 42.0);

    // The returned pipeline_ref should reference the stored constants
    assert_eq!(pipeline_ref.const_index, 0);
}
