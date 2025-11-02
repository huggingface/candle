/// The `QueueBuffer` struct is responsible for queuing a kernel to the GPU.
/// It provides functionality to:
/// - Add parameters to a `MetaBuffer`.
/// - Specify pipeline constants for the next pipeline.
/// - Create a reference to the pipeline.
/// - Enqueue the defined pipeline.
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, MutexGuard};

use std::hash::Hash;

use candle_wgpu_kernels::Constants;

use crate::Layout;

#[cfg(feature = "wgpu_debug")]
use super::debug_info::{DebugInfo, MInfo, Measurements, ShaderInfo};

use super::cache::{BindgroupAlignment, BindgroupAlignmentLayout,BufferReferenceId, CachedBindgroupId, CachedBufferId,BindgroupInputBase};
use super::util::{ObjectToIdMapper, ToU32};
use super::wgpu_functions::{MetaArray, ConstArray, ToKernelParameterMeta};

pub const MAX_DISPATCH_SIZE: u32 = 65535;

#[derive(Debug)]
pub(crate) enum MlQueue {
    Dispatch(MlQueueDispatch),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
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
///Pipeline, Pipeline Constants, and Inplaceable Information
pub struct PipelineReference(
    pub candle_wgpu_kernels::PipelineIndex,
    pub usize, //Index into an Array with Pipeline Constants
    #[cfg_attr(
        any(feature = "wgpu_debug_serialize", feature = "wgpu_debug"),
        serde(skip)
    )]
    pub OpIsInplaceable, 
);

pub(crate) type BindGroupReference = crate::wgpu_backend::cache::BindgroupReferenceFull;

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
    pub(crate) workload_size: usize, //the total size needed to calculate. Needed so we do not queue to many operations at once.
    #[cfg(feature = "wgpu_debug")]
    pub(crate) debug: Option<String>,
}



///a struct, where all operations are cached
#[derive(Debug)]
pub struct QueueBufferInner {
    ///All quened commands
    pub(crate) command_queue: Vec<MlQueue>,

    ///u32 MetaArray for parameters of kernels
    meta_array: MetaArray,

    ///ConstArray is used to store the pipeline constants for the next pipeline call 
    const_array: ConstArray,

    ///ConstArray To Id, maps a set of constants of a pipeline to an unique id.
    const_id_map: ObjectToIdMapper<ConstArray>,

    ///Id to ConstArray, maps a a unique Id to a set of Constants
    pub(crate) id_to_const_array: Vec<Vec<(&'static str, f64)>>,

    global_command_index: u32,

    ///Current position inside the MetaArray
    pub(crate) current_meta: u32,
    
    ///The last destination bufffer, of the last call 
    ///will be used as a workaround to wait for the last command queue
    pub(crate) last_buffer: Option<CachedBufferId>, 
}



impl QueueBufferInner {
    pub fn new(size: u32) -> Self {
        Self {
            command_queue: vec![],
            meta_array: MetaArray::new(size),
            current_meta: 0,
            const_array: ConstArray::new(),
            const_id_map: ObjectToIdMapper::new(),
            id_to_const_array: Vec::new(),
            last_buffer: None,
            global_command_index: 1,
        }
    }

    ///Resets the ConstArray for the next pipeline
    pub fn init(&mut self) {
        self.const_array.0.clear();
    }

    ///Removes all Operations from the queue
    pub fn clear(&mut self) {
        self.command_queue.clear();
        self.meta_array.0.clear();
        self.init();
        self.current_meta = 0;
    }

    pub fn get_meta(&self) -> &Vec<u32> {
        &self.meta_array.0
    }

    pub fn get_meta_mut(&mut self) -> &mut Vec<u32> {
        &mut self.meta_array.0
    }

    pub fn add_layout(
        &mut self,
        layout: &Layout,
        is_contiguous: bool,
        constant_dims: Constants,
        constant_is_startofsset_zero: Constants,
        constant_is_contiguous: Constants,
    ) {
        let shape = layout.shape().dims();
        let stride = layout.stride();

        self.add_const(constant_dims, shape.len());
        if layout.start_offset() != 0 {
            self.add_const(constant_is_startofsset_zero, false);
            self.add(layout.start_offset());
        }

        if is_contiguous {
            self.add(layout.shape().elem_count());
        } else {
            self.add_const(constant_is_contiguous, false);

            self.get_meta_mut().extend(shape.iter().map(|&x| x as u32));
            self.get_meta_mut().extend(stride.iter().map(|&x| x as u32));
        }
    }

    pub fn add_layout1(&mut self, layout: &Layout) {
        self.add_layout(
            layout,
            layout.is_contiguous(),
            Constants::ConstDims1,
            Constants::ConstIsStartoffsetZero1,
            Constants::ConstIsContiguous1,
        );
    }

    pub fn add_layout2(&mut self, layout: &Layout) {
        self.add_layout(
            layout,
            layout.is_contiguous(),
            Constants::ConstDims2,
            Constants::ConstIsStartoffsetZero2,
            Constants::ConstIsContiguous2,
        );
    }

    pub fn add_layout3(&mut self, layout: &Layout) {
        self.add_layout(
            layout,
            layout.is_contiguous(),
            Constants::ConstDims3,
            Constants::ConstIsStartoffsetZero3,
            Constants::ConstIsContiguous3,
        );
    }

    //forces to write the shapes and strides
    pub fn add_layout1_non_contiguous(&mut self, layout: &Layout) {
        self.add_layout(
            layout,
            false,
            Constants::ConstDims1,
            Constants::ConstIsStartoffsetZero1,
            Constants::ConstIsContiguous1,
        );
    }

    pub fn add_layout2_non_contiguous(&mut self, layout: &Layout) {
        self.add_layout(
            layout,
            false,
            Constants::ConstDims2,
            Constants::ConstIsStartoffsetZero2,
            Constants::ConstIsContiguous2,
        );
    }

    pub fn add_layout3_non_contiguous(&mut self, layout: &Layout) {
        self.add_layout(
            layout,
            false,
            Constants::ConstDims3,
            Constants::ConstIsStartoffsetZero3,
            Constants::ConstIsContiguous3,
        );
    }

    pub fn get_pipeline(
        &mut self,
        pipeline: impl Into<candle_wgpu_kernels::PipelineIndex>,
    ) -> PipelineReference {
        let (index, is_new) = self.const_id_map.get_or_insert(&self.const_array);
        if is_new {
            self.id_to_const_array.push(self.const_array.to_vec())
        }
        self.init();
        PipelineReference(pipeline.into(), index, OpIsInplaceable::new())
    }

    pub fn get_pipeline_const<T: ToU32>(
        &mut self,
        pipeline: impl Into<candle_wgpu_kernels::PipelineIndex>,
        const_vec: Vec<T>,
    ) -> PipelineReference {
        for (index, v) in const_vec.into_iter().enumerate() {
            self.const_array
                .0
                .push((candle_wgpu_kernels::Constants::get_const(index), v.to_u32()));
        }

        let (index, is_new) = self.const_id_map.get_or_insert(&self.const_array);
        if is_new {
            self.id_to_const_array.push(self.const_array.to_vec());
        }
        self.init();
        PipelineReference(pipeline.into(), index, OpIsInplaceable::new())
    }

    pub fn get_pipeline_const_inplace<T: ToU32>(
        &mut self,
        pipeline: impl Into<candle_wgpu_kernels::PipelineIndex>,
        const_vec: Vec<T>,
        inplaceable: OpIsInplaceable,
    ) -> PipelineReference {
        for (index, v) in const_vec.into_iter().enumerate() {
            self.const_array
                .0
                .push((candle_wgpu_kernels::Constants::get_const(index), v.to_u32()));
        }

        let (index, is_new) = self.const_id_map.get_or_insert(&self.const_array);
        if is_new {
            self.id_to_const_array.push(self.const_array.to_vec())
        }
        self.init();
        PipelineReference(pipeline.into(), index, inplaceable)
    }

    ///Adds the parameter 'value' to the MetaArray
    pub fn add<T: ToKernelParameterMeta>(&mut self, value: T) {
        self.meta_array.add(value);
    }

    pub fn add_const<T: ToU32>(&mut self, key: candle_wgpu_kernels::Constants, value: T) {
        self.const_array.insert(key, value);
    }

    pub fn global_command_index(&self) -> u32 {
        self.global_command_index
    }

    pub fn set_global_command_index(&mut self, global_command_index: u32) {
        self.global_command_index = global_command_index;
    }

    //allows to load const debug info(for simulating calls)
    pub fn load_debug_info(&mut self, consts: Vec<Vec<(&'static str, f64)>>) {
        self.id_to_const_array = consts;
        self.const_id_map.next_id = self.id_to_const_array.len();
    } 
}

pub struct QueueBuffer<'a>(MutexGuard<'a, QueueBufferInner>);

impl<'a> QueueBuffer<'a> {
    pub fn new(inner: MutexGuard<'a, QueueBufferInner>) -> Self {
        QueueBuffer(inner)
    }

    /**************** Enqueue Helper: ****************/ 
    ///Enqueues a command with a WorkgroupSize of 64 on the X dimension.
    pub fn enqueue_64(
        self,
        pipeline: PipelineReference,
        bind_group: BindGroupReference,
        length: u32,
        //workload_size is the total size needed to calculate. Needed so we do not queue to many operations at once.
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

    ///Enqueues a command with a WorkgroupSize of 64 on the X dimension.
    ///With extra debug Info when `wgpu_debug` is enabled 
    pub fn enqueue_64_extra(
        self,
        pipeline: PipelineReference,
        bind_group: BindGroupReference,
        length: u32,
        //workload_size is the total size needed to calculate. Needed so we do not queue to many operations at once.
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

    ///Enqueues a command with a WorkgroupSize of 64 on the X dimension.
    ///If the length is greater than 65535, more elements will be enqueued in the Y dimension.
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

    ///Enqueues a command with a WorkgroupSize of 64 on the X dimension.
    ///If the length is greater than 65535, more elements will be enqueued in the Y dimension.
    ///With extra debug Info when `wgpu_debug` is enabled 
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

    ///Enqueues a command with x, y and z dimension.
    pub fn enqueue_workgroups(
        self,
        pipeline: PipelineReference,
        bind_group: BindGroupReference,
        x: u32,
        y: u32,
        z: u32,
        //workload_size is the total size needed to calculate. Needed so we do not queue to many operations at once.
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
    ///Enqueues a command with x, y and z dimension.
    ///With extra debug Info when `wgpu_debug` is enabled 
    pub fn enqueue_workgroups_extra(
        mut self,
        pipeline: PipelineReference,
        bind_group: BindGroupReference,
        x: u32,
        y: u32,
        z: u32,
        //workload_size is the total size needed to calculate. Needed so we do not queue to many operations at once.
        workload_size: usize,
        #[cfg(feature = "wgpu_debug")] _debug: Option<String>,
    ) {
        if y > MAX_DISPATCH_SIZE || z > MAX_DISPATCH_SIZE || x > MAX_DISPATCH_SIZE {
            panic!(
                "can not queue y or z higher than 65535 x:{x}, y:{y}, z:{z}, pipeline: {:?}",
                pipeline
            );
        }
        let q = MlQueue::Dispatch(MlQueueDispatch {
            x,
            y,
            z,
            pipeline: pipeline.clone(),
            pipeline_cached: None,
            bindgroup: bind_group,
            bindgroup_cached: None,
            meta: self.current_meta,
            workload_size,
            #[cfg(feature = "wgpu_debug")]
            debug: _debug,
        });
        self.command_queue.push(q);
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
            BindgroupAlignmentLayout::Bindgroup3(alignment, alignment, alignment,alignment),
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