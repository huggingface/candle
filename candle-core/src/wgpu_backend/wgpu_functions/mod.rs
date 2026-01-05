pub mod binary;
pub mod cmp;
pub mod conv2d;
pub mod convert;
pub mod copy;
pub mod gather;
pub mod index_select;
pub mod matmul;
pub mod pool2d;
pub mod reduce;
pub mod rms_norm;
pub mod softmax;
pub mod unary;
pub mod upsample;
pub mod where_cond;
pub mod rotary_emb_i;

use rustc_hash::FxHasher;
use std::{
    hash::{Hash, Hasher}, num::NonZeroU64
};

use super::{
    cache::{
        BindgroupAlignmentLayout, BindgroupInputBase, BindgroupReferenceFull, BindgroupReferenceInput, BufferReferenceId, CachedBindgroupFull, CachedBindgroupInput, CachedBufferId, ModelCache
    }, 
    queue_buffer::{BindGroupReference, MlQueue, OpIsInplaceable, QueueBuffer, QueueBufferInner}, 
    util::{FixedArray, ToU32, ToF64}, 
    WgpuDevice
};
use crate::wgpu_backend::{cache::BindgroupAlignment, util::ReferenceTrait};
use tracing::{instrument, span, Level};

pub use candle_wgpu_kernels::DType;
pub use candle_wgpu_kernels::Pipelines;
use candle_wgpu_kernels::EntryPoint;

use crate::{Layout, WgpuError};
use std::borrow::Cow;

/**************** FUNCTIONS ****************/ 
pub use binary::queue_binary_buffer_from_buffer;
pub use cmp::queue_cmp_buffer_from_buffer;
pub use conv2d::{queue_conv1d, queue_conv1d_transpose, queue_conv2d, queue_conv2d_transpose};
pub use convert::{
    queue_convert, queue_convert_f32_to_u8,
    queue_convert_u32_to_u8, queue_convert_u8_to_f32,
    queue_convert_f32_to_f16, queue_convert_f16_to_f32
};
pub use copy::{queue_copy, queue_copy2d, queue_copy3d, queue_copy3d_padded, queue_copy_strided, queue_transpose3d};
pub use gather::{queue_gather, queue_index_add_inplace, queue_scatter_add_inplace, queue_scatter_set_inplace};
pub use index_select::queue_index_select;
pub use matmul::queue_matmul_buffer;
pub use pool2d::{queue_avg_pool2d, queue_max_pool2d};
pub use reduce::queue_reduce_from_buffer_op;
pub use rms_norm::{queue_rms_norm, queue_layer_norm};
pub use softmax::queue_softmax;
pub use unary::{queue_unary_from_buffer_op, queue_unary_inplace_op};
pub use upsample::{queue_upsample1d, queue_upsample2d};
pub use where_cond::queue_where_cond;
pub use rotary_emb_i::queue_rotary_emb_i;

#[derive(Debug, Copy, Clone)]
pub struct WgpuTensor<'a>{
    layout : &'a Layout,
    buffer : BufferReferenceId,
}

impl<'a> WgpuTensor<'a> {
    pub fn new(layout: &'a Layout, buffer: BufferReferenceId) -> Self {
        Self { layout, buffer }
    }
    
    pub fn layout(&self) -> &Layout {
        self.layout
    }
    
    pub fn buffer(&self) -> BufferReferenceId {
        self.buffer
    }
}

///Helper Type MetaArray, for constructing the MetaBuffer
///The MetaBuffer is used to pass Parameters to the Kernel. 
///Paramerters for multiple Commands are grouped together in this MetaArray.
#[derive(Debug)]
pub struct MetaArray(pub Vec<u32>);



#[derive(Debug, Clone, Hash, PartialEq, Eq)]
///Helper Array to Construct Kernel Constants. 
///Kernel Constants are compiled into the kernel. 
pub struct ConstArray(pub FixedArray<(candle_wgpu_kernels::Constants, u32), 32>);

impl ConstArray {
    pub fn to_vec(&self) -> Vec<(&'static str, f64)> {
        Vec::from_iter(
            self
                .0
                .iter()
                .map(|(k, v)| (k.get_entry_point(), v.to_f64())),
        )
    }
}

//Allows objects to be added to the Meta Parameter Array
pub trait ToKernelParameterMeta {
    fn write_meta(&self, meta: &mut MetaArray);
}

impl MetaArray {
    pub fn new(capacity: u32) -> Self {
        MetaArray(Vec::with_capacity(capacity as usize))
    }

    pub fn add<T: ToKernelParameterMeta>(&mut self, value: T) {
        value.write_meta(self);
    }
}

impl<T: ToU32 + Copy> ToKernelParameterMeta for T {
    fn write_meta(&self, meta: &mut MetaArray) {
        meta.0.push((*self).to_u32());
    }
}

impl ConstArray {
    pub fn new() -> Self {
        ConstArray(FixedArray::new())
    }

    pub fn insert<T: ToU32>(&mut self, key: candle_wgpu_kernels::Constants, value: T) {
        self.0.push((key, value.to_u32()));
    }
}

impl Default for ConstArray {
    fn default() -> Self {
        Self::new()
    }
}


fn next_divisible_by_n<T: num_traits::Num + Clone>(value: T, n: T) -> T {
    if n.is_zero() {
        panic!("n must be a non-zero integer");
    }

    if (value.clone() % n.clone()).is_zero() {
        value
    } else {
        value.clone() + (n.clone() - value % n)
    }
}

impl WgpuDevice{
    ///Returns the Meta Array. 
    pub fn get_queue<'a>(&'a self) -> QueueBuffer<'a> {
        let mut command_queue = self
            .command_queue
            .lock()
            .expect("could not get meta command_queue lock");
        let meta_array_length = command_queue.get_meta().len() as i32;
        let meta_offset = next_divisible_by_n(
            meta_array_length,
            self.device_limits.min_storage_buffer_offset_alignment as i32 / 4,
        );
        command_queue.current_meta = meta_offset as u32;
        command_queue
            .get_meta_mut()
            .extend(std::iter::repeat_n(0, (meta_offset - meta_array_length) as usize));

        QueueBuffer::new(command_queue)
    }

    ///Returns the candle-wgpu-kernels::DType for the given dtype if available on this device.
    pub fn get_dtype(&self, dtype: crate::DType) -> crate::Result<DType> {
        match (dtype, self.is_dtype_available(dtype)) {
            (crate::DType::U8, true) => Ok(DType::U8),
            (crate::DType::U32, true) => Ok(DType::U32),
            (crate::DType::F32, true) => Ok(DType::F32),
            (crate::DType::I64, true) => Ok(DType::I64),
            (crate::DType::F64, true) => Ok(DType::F64),
            (crate::DType::F16, true) => Ok(DType::F16),
            (crate::DType::BF16, _) | (crate::DType::U8, _) | (crate::DType::F8E4M3, _) => Err(crate::Error::Wgpu(WgpuError::from(format!(
                "Dtype {:?} not supported on wgpu",
                dtype
            )))),
            (_, false) => Err(crate::Error::Wgpu(WgpuError::from(format!(
                "Dtype {:?} not supported on this wgpu device",
                dtype
            )))),
        }
    }
    
}


/**************** FLUSH COMMANDS TO GPU: ****************/ 
#[instrument(skip(dev, queue_buffer, cache))]
///Prepares Buffers
fn prepare(dev: &WgpuDevice, queue_buffer: &mut QueueBufferInner, cache: &mut ModelCache) {
    let global_index = queue_buffer.global_command_index();
    cache.mappings.set_global_command_index(global_index);
    let queue = &mut queue_buffer.command_queue;
    {
        let mut hasher = FxHasher::default();
        for q in queue.iter() {
            match q {
                MlQueue::Dispatch(q) => {
                    q.pipeline.hash(&mut hasher);
                }
            }
        }

        let current_hash = hasher.finish();

        tracing::info!("current hash: {current_hash}");

        cache.mappings.set_current_buffer_mapping(current_hash);
        let deleted_entries = cache.buffer_reference.get_deletion_entries();

        for entry in deleted_entries.iter() {
            if let Some(buffer_reference) = cache.buffer_reference.get_mut(entry) {
                buffer_reference.set_referenced_by_candle_storage(false);
                buffer_reference.set_last_used(0); //if this buffer will not be used in this queue, we can delete the queue and free the used buffer.
            }
        }

        for (index, q) in queue.iter().enumerate() {
            let command_index = global_index + index as u32;

            let mut check_buffer = |buffer_reference| {
                if let Some(buffer) = cache.buffer_reference.get_mut(&buffer_reference) {
                    if buffer.first_used() == 0 {
                        //not alreaedy set:
                        buffer.set_first_used(command_index);
                    }
                    if buffer.last_used() < command_index {
                        buffer.set_last_used(command_index);
                    }
                }
            };
            match q {
                MlQueue::Dispatch(q) => {
                    let dest = q.bindgroup.get_dest();
                    let input = q.bindgroup.get_input();

                    check_buffer(*dest);

                    input.fold_owned(|v| {
                        check_buffer(v);
                    });
                }
            }
        }

        for entry in deleted_entries.iter() {
            if let Some(buffer_reference) = cache.buffer_reference.get_mut(entry) {
                if buffer_reference.last_used() == 0 {
                    //This buffer was not used in the queue -> we can remove it!
                    let buffer_cached_id = *buffer_reference.cached_buffer_id();
                    if buffer_reference.cached_buffer_id().is_valid() {
                        cache.buffers.free_buffer(&buffer_cached_id);
                    }
                    cache.buffer_reference.delete(entry);
                }
            }
        }

        cache
            .buffers
            .set_max_memory_allowed(dev.configuration.buffer_cached_max_allowed_size);
    }
}


#[instrument(skip(dev, cache, command_queue, waiting_buffer, current_meta, meta_array))]
///Builds Command Buffer and writes meta_array to the gpu 
fn get_command_buffer(
    dev: &WgpuDevice,
    meta_array: &[u32],
    command_queue: &[MlQueue],
    current_meta: usize,
    waiting_buffer: &Option<CachedBufferId>, //a buffer, we want to wait for, after all commands have been queued
    cache: &mut ModelCache,
) -> wgpu::CommandBuffer {
    #[cfg(feature = "wgpu_debug")]
    let query_set = &dev.debug.query_set;

    #[cfg(feature = "wgpu_debug")]
    let global_index = dev.debug.counter.load(std::sync::atomic::Ordering::Relaxed);

    #[cfg(feature = "wgpu_debug")]
    let mut debug_index = 0;

    let span1 = span!(Level::INFO, "Write Metabuffer");
    let _enter1 = span1.enter();

    let data = bytemuck::cast_slice(meta_array);
    if data.len() as u32 + 256 > dev.configuration.meta_buffer_size {
        panic!("Meta Buffer was to big, length was: {}", data.len());
    }

    //write Meta Buffer
    dev.queue.write_buffer(&dev.meta_buffer, 0, data);
    drop(_enter1);

    let span1 = span!(Level::INFO, "Create Encoder");
    let _enter1 = span1.enter();

    let mut encoder = dev
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });

    for q in command_queue.iter() {
        match q {
            MlQueue::Dispatch(q) => {
                if let Some(bindgroup) = &q.bindgroup_cached {
                    if let Some(pipeline) = &q.pipeline_cached {
                        let qx = q.x;
                        let qy = q.y;
                        let qz = q.z;
                        let meta = q.meta - current_meta as u32;

                        #[cfg(feature = "wgpu_debug")]
                        cpass.write_timestamp(query_set, debug_index);
                        let span1 = span!(Level::INFO, "Set Pipeline");
                        let _enter1 = span1.enter();
                        cpass.set_pipeline(pipeline);
                        drop(_enter1);

                        if meta * 4 >= dev.configuration.meta_buffer_size - 256 {
                            panic!(
                                "meta is to big!: meta was {meta}, q.meta: {}/{current_meta}",
                                q.meta
                            );
                        }

                        let span1 = span!(Level::INFO, "Set Bindgroup");
                        let _enter1 = span1.enter();

                        let bindgroup = cache
                            .bindgroups
                            .get_bindgroup(bindgroup)
                            .expect("bindgroup could not be found!");

                        let buffers = bindgroup.buffer();
                        let vd = *buffers.get_dest();
                        match buffers.get_input() {
                            BindgroupInputBase::Bindgroup0(_) => {}
                            BindgroupInputBase::Bindgroup1(v1, _) => {
                                if v1 == &vd {
                                    panic!("B1: output and input are equal");
                                }
                            }
                            BindgroupInputBase::Bindgroup2(v1, v2, _) => {
                                if v1 == &vd {
                                    panic!("B2: output and input1 are equal");
                                }
                                if v2 == &vd {
                                    panic!("B2: output and input2 are equal");
                                }
                            }
                            BindgroupInputBase::Bindgroup3(v1, v2, v3, _) => {
                                if v1 == &vd {
                                    panic!("B3: output and input1 are equal");
                                }

                                if v2 == &vd {
                                    panic!("B3: output and input2 are equal");
                                }

                                if v3 == &vd {
                                    panic!("B3: input3 and output are equal");
                                }
                            }
                        }

                        cpass.set_bind_group(0, bindgroup.bindgroup(), &[meta * 4]);
                        drop(_enter1);

                        let span1 = span!(Level::INFO, "Dispatch Workgroups");
                        let _enter1 = span1.enter();
                        cpass.dispatch_workgroups(qx, qy, qz);
                        drop(_enter1);

                        #[cfg(feature = "wgpu_debug")]
                        {
                            cpass.write_timestamp(query_set, debug_index + 1);
                            dev.debug.insert_info(
                                global_index + debug_index * 8,
                                crate::wgpu_backend::debug_info::ShaderDebugInfo{
                                    pipeline: format!(
                                        "Shader: '{}', Pipeline: '{}', {}",
                                        cache.shader.loader_cache.get_shader_name(q.pipeline.0.get_shader()),
                                        cache.shader.loader_cache.get_entry_point(q.pipeline.0),
                                        q.debug.to_owned().map_or("".to_string(), |s| s)
                                    ),
                                    workload_size: q.workload_size as u64,
                                    x : q.x,
                                    y : q.y,
                                    z : q.z,
                                },
                            );
                            debug_index += 2;
                        
                        
                            if cache.full_recording.should_record
                            {
                                use crate::wgpu_backend::wgpu_functions;

                                let debug_info = crate::wgpu_backend::device::DebugPipelineRecording {
                                    x: q.x,
                                    y: q.y,
                                    z: q.z,
                                    pipeline: q.pipeline.clone(),
                                    meta : meta_array[meta as usize..].to_vec(),
                                    bindgroup: q.bindgroup.clone(),
                                    count: 1,
                                };

                                fn get_buffer_data(dev: &WgpuDevice, buffer_reference : CachedBufferId, cache: &mut ModelCache) -> crate::Result<super::debug_info::NumericArray>{
                                    #[cfg(not(target_arch = "wasm32"))]{
                                        let staging_buffer;
                                        if buffer_reference.is_valid() {
                                            if let Some(buffer) = cache.buffers.get_buffer(&buffer_reference) {
                                                staging_buffer =  wgpu_functions::copy_buffer_to_staging_buffer(dev, buffer.buffer());
                                            } else {
                                                panic!("Unespected error at read_data from gpu. Tensor WgpuStorage did not Point to a wgpu Buffer")
                                            }
                                        } else {
                                            panic!("Unespected error at read_data from gpu. Tensor WgpuStorage did not Point to a wgpu Buffer")
                                        }
                                    

                                        let data = pollster::block_on(wgpu_functions::read_from_staging_buffer_async::<u8>(dev, staging_buffer))?;
                                        //let data = pollster::block_on(wgpu_functions::read_from_buffer_reference_async::<u32>(dev, buffer_reference))?;
                                        Ok(super::debug_info::NumericArray::U8(data))
                                    }
                                    #[cfg(target_arch = "wasm32")]{
                                        crate::bail!("Synchronous read not supported on wasm32");
                                    }
                                }

                                let buffer_input1 = buffers.get_input().get_input1().cloned();
                                let buffer_input2 = buffers.get_input().get_input2().cloned();
                                let buffer_input3 = buffers.get_input().get_input3().cloned();
                                let vd1 = get_buffer_data(dev, vd, cache).expect("Expect to Read the Buffer");
                                let v_input1 = buffer_input1.map(|buffer| get_buffer_data(dev, buffer, cache).expect("Expect to Read the Buffer"));
                                let v_input2 = buffer_input2.map(|buffer| get_buffer_data(dev, buffer, cache).expect("Expect to Read the Buffer"));
                                let v_input3 = buffer_input3.map(|buffer| get_buffer_data(dev, buffer, cache).expect("Expect to Read the Buffer"));


                                let data = super::debug_info::DebugPipelineRecordingWithData{
                                    recording : debug_info,
                                    v_dest : vd1,
                                    v_input1,
                                    v_input2,
                                    v_input3
                                };

                                cache.full_recording.recordings.push(data);
                            }
                        }
                    }
                }
            }
        }
    }

    let span2 = span!(Level::INFO, "Drop Cpass");
    let _enter2 = span2.enter();
    drop(cpass);
    drop(_enter2);
    drop(_enter1);

    if let Some(waiting_buffer) = waiting_buffer {
        let staging_buffer = &dev.staging_probe_buffer;
        if let Some(buffer) = cache.buffers.get_buffer(waiting_buffer) {
            encoder.copy_buffer_to_buffer(buffer.buffer(), 0, staging_buffer, 0, 4);
        }
    }

    #[cfg(feature = "wgpu_debug")]
    end_debug_queue(
        dev,
        command_queue.len() as u32 * 2,
        global_index,
        &mut encoder,
        query_set,
    );

    let span1 = span!(Level::INFO, "Encoder Finish");
    let _enter1 = span1.enter();
    let result = encoder.finish();
    drop(_enter1);
    result
}


#[instrument(skip(dev, command_buffer, cache, index, last_meta, current_meta))]
///Maps Virtual Compute Graph Buffers to actual wgpu Buffers
fn set_buffers(
    dev: &WgpuDevice,
    command_buffer: &mut QueueBufferInner,
    index: &mut usize,
    current_meta: usize,
    last_meta: &mut usize,
    cache: &mut ModelCache,
) -> crate::Result<(bool, u64)> {
    let global_index = command_buffer.global_command_index();
    
    #[cfg(feature = "wgpu_debug")]
    {
        use crate::wgpu_backend::cache::AverageBufferInfo;

        let mut buffers: std::collections::HashMap<(u32,bool),u32>  = std::collections::HashMap::new();
        for (_id, cached_buffer) in cache.buffers.iter_buffers() {
            let size = cached_buffer.buffer().size() as u32;

            // Adjust this depending on how you detect "free"
            let is_free = cached_buffer.is_free();

            *buffers.entry((size, is_free)).or_insert(0) += 1;
        }

        let debug_buffer_info = crate::wgpu_backend::cache::DebugBufferUsage{ 
            memory_alloc : cache.buffers.buffer_memory(),
            memory_free : cache.buffers.buffer_free_memory(),
            buffers : buffers.into_iter().map(|(key, value)| AverageBufferInfo{count : value, is_free : key.1, size: key.0}).collect(),
            command_buffer_id : global_index
        };
        cache.debug_buffer_info.push(debug_buffer_info);
    }
    let queue = &mut command_buffer.command_queue;
    let mut cache_limit = false;
    let mut total_workload = 0u64; //we only allow a certain amount of workload per commandBuffer
    let start_index = *index;

    for q in queue[*index..].iter_mut() {
        #[cfg(feature = "wgpu_debug")]
        {
            let ele_size = *index - start_index;
            if ele_size >= wgpu::QUERY_SET_MAX_QUERIES as usize / 2 - 1{
                break;
            } 
            if cache.full_recording.should_record && ele_size > 1{
                break;
            }
        }

        *index += 1;
        match q {
            MlQueue::Dispatch(q) => {
                let command_index = (*index - 1) as u32 + global_index;

                let ele_size = *index - start_index;
                if (total_workload + q.workload_size as u64) > dev.configuration.max_workload_size
                    && ele_size > 1
                {
                    *index -= 1;
                    break;
                } else {
                    total_workload += q.workload_size as u64;
                }

                let span1 = span!(Level::INFO, "SetBuffers_Analyse UnaryBuffer ");
                let _enter1 = span1.enter();
                let mut optimize_copy_inplace = false;
                let mut input_replaced_buffer = BufferReferenceId::new(0, 0); //input buffer, that was replaced

                fn should_optimize_inplace(cache : &mut ModelCache,vdest_id: &BufferReferenceId, v1_id: &BufferReferenceId, command_index : u32) -> bool{
                    let vdest = cache.buffer_reference.get(vdest_id);
                    let v1 = cache.buffer_reference.get(v1_id);
                    if let Some(vdest) = vdest {
                        if let Some(v1) = v1 {
                            if !v1.cached_buffer_id().is_valid() {
                                panic!("while optimizing: input buffer {:?}({:?}) storage was not set in {command_index}", v1, v1_id)
                            }

                            //this buffer was last used in this pipeline
                            if v1.last_used() == command_index &&  vdest.size() <= v1.size() && !vdest.cached_buffer_id().is_valid() {
                                let input_cached_buffer_id = *v1.cached_buffer_id();
                                let vdest = cache.buffer_reference.get_mut(vdest_id);
                                if let Some(vdest) = vdest {
                                    vdest.set_cached_buffer_id(input_cached_buffer_id);
                                }
                                return true;
                            }
                        }
                    }
                    false
                }

                let bindgroup_reference = &q.bindgroup;

                if q.pipeline.0.get_shader().get_loader() == candle_wgpu_kernels::DefaultWgpuShader::LOADER_INDEX{
                    let pipeline : Pipelines = q.pipeline.0.into();
                    if let Pipelines::Unary(
                        dtype,
                        candle_wgpu_kernels::unary::Functions::UnaryFromBufferContiguous,
                    ) = &pipeline
                    {
                        if q.pipeline.2.input1_inplaceable {
                            if let BindgroupReferenceInput::Bindgroup1(
                                v1_id,
                                BindgroupAlignmentLayout::Bindgroup1(dest_alignment, _),
                            ) = bindgroup_reference.get_input()
                            {
                                if should_optimize_inplace(cache, bindgroup_reference.get_dest(), v1_id, command_index) {
                                    cache.unary_inplace_counter += 1;
                                    input_replaced_buffer = *v1_id;
                                    q.pipeline.0 = Pipelines::Unary(
                                        dtype.clone(),
                                        candle_wgpu_kernels::unary::Functions::UnaryInplaceContiguous,
                                    ).into();
                                    q.bindgroup = BindGroupReference::new(
                                        *v1_id,
                                        BindgroupInputBase::Bindgroup0(
                                            BindgroupAlignmentLayout::Bindgroup0(*dest_alignment),
                                        ),
                                    );
    
                                }
                            }
                        }
                    } else if let Pipelines::Binary(
                        dtype,
                        candle_wgpu_kernels::binary::Functions::BinaryBufferFromBufferContiguousBoth,
                    ) = &pipeline
                    {
                        if let BindgroupReferenceInput::Bindgroup2(
                            v1_id,
                            v2_id,
                            BindgroupAlignmentLayout::Bindgroup2(dest_alignment, _, _),
                        ) = bindgroup_reference.get_input()
                        {
                            if !cache
                                .buffer_reference
                                .get(v1_id)
                                .expect("buffer_reference v1 not found")
                                .cached_buffer_id()
                                .is_valid()
                            {
                                panic!("input buffer v1 {:?}({:?}) has not input cache storage set {command_index}", cache.buffer_reference.get(v1_id).unwrap(), v1_id);
                            }
                            if !cache
                                .buffer_reference
                                .get(v2_id)
                                .expect("buffer_reference v2 not found")
                                .cached_buffer_id()
                                .is_valid()
                            {
                                panic!("input buffer v2 {:?}({:?}) has not input cache storage set {command_index}", cache.buffer_reference.get(v2_id).unwrap(), v2_id);
                            }
    
                            if q.pipeline.2.input1_inplaceable {
                                if should_optimize_inplace(cache, bindgroup_reference.get_dest(), v1_id, command_index) {
                                    input_replaced_buffer = *v1_id;
                                    cache.binary_inplace_counter += 1;
    
                                    q.pipeline.0 = Pipelines::Binary(dtype.clone(), candle_wgpu_kernels::binary::Functions::BinaryBufferInplace1ContiguousBoth).into();
                                    q.bindgroup = BindGroupReference::new(
                                        *v1_id,
                                        BindgroupInputBase::Bindgroup1(
                                            *v2_id,
                                            BindgroupAlignmentLayout::Bindgroup1(
                                                *dest_alignment,
                                                *dest_alignment,
                                            ),
                                        ),
                                    );
                                }
                            } else if q.pipeline.2.input2_inplaceable && should_optimize_inplace(cache, bindgroup_reference.get_dest(), v2_id, command_index){
                                input_replaced_buffer = *v2_id;
                                cache.binary_inplace_counter += 1;
                                q.pipeline.0 = Pipelines::Binary(dtype.clone(), candle_wgpu_kernels::binary::Functions::BinaryBufferInplace2ContiguousBoth).into();
                                q.bindgroup = BindGroupReference::new(
                                    *v2_id,
                                    BindgroupInputBase::Bindgroup1(
                                        *v1_id,
                                        BindgroupAlignmentLayout::Bindgroup1(
                                            *dest_alignment,
                                            *dest_alignment,
                                        ),
                                    ),
                                );
                            }
                        }
                    } else if let Pipelines::Copy(_, candle_wgpu_kernels::copy::Functions::Copy) =
                        &pipeline
                    {
                        if q.pipeline.2.input1_inplaceable {
                            if let BindgroupReferenceInput::Bindgroup1(v1_id, _) =
                                bindgroup_reference.get_input()
                            {
                                let v1 = cache.buffer_reference.get(v1_id);
                                if let Some(v1) = v1 {
                                    let vdest_id = bindgroup_reference.get_dest();
    
                                    let v1_cached_id = *v1.cached_buffer_id();
                                    let v1_size = v1.size();
                                    //this buffer was last used in this pipeline
                                    if v1.last_used() == command_index {
                                        let vdest = cache.buffer_reference.get_mut(vdest_id);
                                        if let Some(vdest) = vdest {
                                            if vdest.size() <= v1_size && !vdest.cached_buffer_id().is_valid() {
                                                vdest.set_cached_buffer_id(v1_cached_id);

                                                cache.copy_inplace_counter += 1;
                                                optimize_copy_inplace = true;
                                            }
                                        }
    
                                        if optimize_copy_inplace {
                                            let v1 = cache.buffer_reference.get_mut(v1_id);
                                            if let Some(v1) = v1 {
                                                v1.set_cached_buffer_id(CachedBufferId::new(0, 0));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                drop(_enter1);
                
                
                #[instrument(skip(cache, bindgroup_reference, command_index, input_replaced_buffer))]
                fn check_for_removal(
                    bindgroup_reference: &BindgroupReferenceFull,
                    command_index: u32,
                    cache: &mut ModelCache,
                    input_replaced_buffer : &BufferReferenceId
                ) {
                    let check_buffer = |buffer_reference: &BufferReferenceId,
                                       cache: &mut ModelCache,
                                       command_index: u32| {
                        if let Some(buffer) = cache.buffer_reference.get_mut(buffer_reference) {
                            if buffer.last_used() <= command_index {
                                //this buffer reference is not used after this:
                                let cached_buffer_id = *buffer.cached_buffer_id();
                                cache.buffer_reference.delete(buffer_reference);
                                if cached_buffer_id.is_valid() && buffer_reference != input_replaced_buffer { //if this buffer was replaced by another buffer, the referenced buffer is still not free
                                    cache.buffers.free_buffer(&cached_buffer_id);
                                }
                            }
                        }
                    };

                    let dest = bindgroup_reference.get_dest();
                    let input = bindgroup_reference.get_input();

                    if cache.buffer_reference.get(dest).is_some() {
                    } else {
                        panic!("dest was not set!");
                    }

                    check_buffer(dest, cache, command_index);

                    input.fold(|v| check_buffer(v, cache, command_index));
                }

                if !optimize_copy_inplace {
                    let pl: &wgpu::PipelineLayout = match q.bindgroup.get_input() {
                        BindgroupReferenceInput::Bindgroup0(alignment) => {
                            &dev.bindgroup_layouts[*alignment].1
                        }
                        BindgroupReferenceInput::Bindgroup1(_, alignment) => {
                            &dev.bindgroup_layouts[*alignment].1
                        }
                        BindgroupReferenceInput::Bindgroup2(_, _, alignment) => {
                            &dev.bindgroup_layouts[*alignment].1
                        }
                        BindgroupReferenceInput::Bindgroup3(_, _, _, alignment) => {
                            &dev.bindgroup_layouts[*alignment].1
                        }
                    };

                    let consts = &command_buffer.id_to_const_array[q.pipeline.1];
                    let pipeline = cache.shader.get_pipeline(&dev.device,&q.pipeline, pl, consts)?;

                    let bindgroup =
                        cache.get_bind_group(dev, &q.bindgroup, q.pipeline.clone(), command_index);

                    let span1 = span!(Level::INFO, "SetBuffers_Optimize implace: ");
                    let _enter1 = span1.enter();   
                    check_for_removal(&q.bindgroup, command_index, cache, &input_replaced_buffer);
                    if cache.should_delete_unused() {
                        //we hit the max cache size
                        cache_limit = true;
                    }
                    q.bindgroup_cached = Some(bindgroup);
                    q.pipeline_cached = Some(pipeline);
                } else {
                    check_for_removal(&q.bindgroup, command_index, cache, &input_replaced_buffer);
                    q.bindgroup_cached = None;
                }

                *last_meta = q.meta as usize;

                let meta_size = (*last_meta - current_meta) * 4 + 256 * 3;
                if meta_size > dev.configuration.meta_buffer_size as usize {
                    break;
                }
                if cache_limit {
                    break;
                }
                if total_workload > dev.configuration.max_workload_size {
                    break;
                }
            }
        }
    }
    
    #[cfg(feature = "wgpu_debug")]
    {
        use crate::wgpu_backend::cache::AverageBufferInfo;

        let mut buffers: std::collections::HashMap<(u32,bool),u32>  = std::collections::HashMap::new();
        for (_id, cached_buffer) in cache.buffers.iter_buffers() {
            let size = cached_buffer.buffer().size() as u32;

            // Adjust this depending on how you detect "free"
            let is_free = cached_buffer.is_free();

            *buffers.entry((size, is_free)).or_insert(0) += 1;
        }

        let debug_buffer_info = crate::wgpu_backend::cache::DebugBufferUsage{ 
            memory_alloc : cache.buffers.buffer_memory(),
            memory_free : cache.buffers.buffer_free_memory(),
            buffers : buffers.into_iter().map(|(key, value)| AverageBufferInfo{count : value, is_free : key.1, size: key.0}).collect(),
            command_buffer_id : global_index
        };
        cache.debug_buffer_info.push(debug_buffer_info);
    }

    let meta_size = (*last_meta - current_meta) * 4 + 256 * 3;
    let ele_size = *index - start_index;
    log::trace!("queue {ele_size}, Meta: {meta_size}, workload: {total_workload}, cache_limit: {cache_limit}");

    Ok((cache_limit, total_workload))
}

#[instrument(skip(dev, queue_buffer))]
///Send queued commands to the GPU
pub(crate) fn flush_gpu_command(
    dev: &WgpuDevice,
    queue_buffer: &mut QueueBufferInner,
) -> crate::Result<()> {
    if !queue_buffer.command_queue.is_empty() {
        let mut cache = dev
            .cache
            .lock().expect("");
        prepare(dev, queue_buffer, &mut cache);
        {
            let mut start_index = 0;
            let mut index = 0;
            let mut current_meta: usize = 0;
            let mut last_meta: usize = 0;

            while index < queue_buffer.command_queue.len() {
                let (should_reuse_unused, _) = set_buffers(
                    dev,
                    queue_buffer,
                    &mut index,
                    current_meta,
                    &mut last_meta,
                    &mut cache,
                )?;

                let last_meta_index = (last_meta + 256 / 4).min(queue_buffer.get_meta().len());
                let cb = get_command_buffer(
                    dev,
                    &queue_buffer.get_meta()[current_meta..last_meta_index],
                    &queue_buffer.command_queue[start_index..index],
                    current_meta,
                    &None,
                    &mut cache,
                );

                if should_reuse_unused {
                    cache.remove_unused();
                }

                #[cfg(not(target_arch = "wasm32"))]
                {
                    let span1 = span!(Level::INFO, "Device Poll");
                    let _enter1 = span1.enter();
                    dev.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                }

                if index > 0 {
                    //set last buffer, so we can wait for it to finish in the future
                    match &queue_buffer.command_queue[index - 1] {
                        MlQueue::Dispatch(d) => if let Some(c) = &d.bindgroup_cached {
                            if let Some(c) = cache.bindgroups.get_bindgroup(c) {
                                queue_buffer.last_buffer = Some(*c.buffer().get_dest())
                            }
                        },
                    }
                }

                let span1 = span!(Level::INFO, "Submit");
                let _enter1 = span1.enter();
                dev.queue.submit(Some(cb));
                drop(_enter1);

                start_index = index;
                current_meta = last_meta;
            }
            finish_commands(queue_buffer, index, &mut cache);
        }
        queue_buffer.clear();
        {
            log::debug!(
                "current memory {} / {}",
                cache.buffers.buffer_memory(),
                cache.buffers.max_memory_allowed()
            );
            cache.mappings.finish();
            cache.remove_unused();
        }
    }
    Ok(())
}

#[instrument(skip(dev, queue_buffer))]
///Send queued commands to the GPU, waits asynchronously
pub(crate) async fn flush_gpu_command_async(
    dev: &WgpuDevice,
    queue_buffer: &mut QueueBufferInner,
) -> crate::Result<()> {
    if !queue_buffer.command_queue.is_empty() {
        log::debug!("flush_gpu_command_async");
        let mut cache = dev
            .cache
            .lock().expect("");
        prepare(dev, queue_buffer, &mut cache);
        {
            let mut start_index = 0;
            let mut index = 0;
            let mut current_meta: usize = 0;
            let mut last_meta: usize = 0;

            while index < queue_buffer.command_queue.len() {
                let (should_reuse_unused, _) = set_buffers(
                    dev,
                    queue_buffer,
                    &mut index,
                    current_meta,
                    &mut last_meta,
                    &mut cache,
                )?;
                let last_meta_index = (last_meta + 256 / 4).min(queue_buffer.get_meta().len());
                let cb = get_command_buffer(
                    dev,
                    &queue_buffer.get_meta()[current_meta..last_meta_index],
                    &queue_buffer.command_queue[start_index..index],
                    current_meta,
                    &queue_buffer.last_buffer,
                    &mut cache,
                );

                if should_reuse_unused {
                    cache.remove_unused();
                }

                synchronize_device(dev).await?;

                if index > 0 {
                    //set last buffer, so we can wait for it to finish in the future
                    match &queue_buffer.command_queue[index - 1] {
                        MlQueue::Dispatch(d) => if let Some(c) = &d.bindgroup_cached {
                            if let Some(c) = cache.bindgroups.get_bindgroup(c) {
                                queue_buffer.last_buffer = Some(*c.buffer().get_dest())
                            }
                        },
                    }
                }

                let span1 = span!(Level::INFO, "Submit");
                let _enter1 = span1.enter();
                dev.queue.submit(Some(cb));
                drop(_enter1);

                start_index = index;
                current_meta = last_meta;
            }
            finish_commands(queue_buffer, index, &mut cache);
        }

        queue_buffer.clear();
        {
            log::debug!(
                "current memory {} / {}",
                cache.buffers.buffer_memory(),
                cache.buffers.max_memory_allowed()
            );
            cache.mappings.finish();
            cache.remove_unused();
        }
    }
    Ok(())
}


fn finish_commands(command_buffer: &mut QueueBufferInner, index: usize, _cache: &mut ModelCache) {
    let global_index = command_buffer.global_command_index();
    command_buffer.set_global_command_index(global_index + index as u32);

    #[cfg(feature = "wgpu_debug")]
    {
        for i in 0..command_buffer.command_queue.len() {
            let current = &command_buffer.command_queue[i];
            let next = &command_buffer.command_queue.get(i + 1);
            match current {
                MlQueue::Dispatch(q) => {
                    let mut next_meta = command_buffer.get_meta().len();
                    if let Some(next) = next {
                        match next {
                            MlQueue::Dispatch(q) => {
                                next_meta = q.meta as usize;
                            }
                        }
                    }

                    let new_bindgroup = BindgroupReferenceFull::new(
                        Default::default(),
                        match q.bindgroup.get_input() {
                            BindgroupInputBase::Bindgroup0(alginment) => BindgroupInputBase::Bindgroup0(*alginment),
                            BindgroupInputBase::Bindgroup1(_, alginment) => {
                                BindgroupInputBase::Bindgroup1(Default::default(), *alginment)
                            }
                            BindgroupInputBase::Bindgroup2(_, _, alginment) => {
                                BindgroupInputBase::Bindgroup2(
                                    Default::default(),
                                    Default::default(),
                                    *alginment,
                                )
                            }
                            BindgroupInputBase::Bindgroup3(_, _, _, alginment) => {
                                BindgroupInputBase::Bindgroup3(
                                    Default::default(),
                                    Default::default(),
                                    Default::default(),
                                    *alginment
                                )
                            }
                        },
                    );

                    let mut meta: Vec<u32> =
                        command_buffer.get_meta()[q.meta as usize..next_meta].into();

                    //the scalar and randstate on unary should have no performance effect:
                    if let Pipelines::Unary(_, candle_wgpu_kernels::unary::Functions::RandInplaceContiguous) = q.pipeline.0.into() {
                        meta[1] = f32::to_bits(1.0);
                        meta[2] = f32::to_bits(1.0);
                        meta[3] = 0; //rand state
                    }
                    else if let Pipelines::Unary(_, _) = q.pipeline.0.into() {
                        meta[1] = f32::to_bits(1.0);
                        meta[2] = f32::to_bits(1.0);
                    }

                    let debug_info = crate::wgpu_backend::device::DebugPipelineRecording {
                        x: q.x,
                        y: q.y,
                        z: q.z,
                        pipeline: q.pipeline.clone(),
                        meta,
                        bindgroup: new_bindgroup,
                        count: 1,
                    };

                    if let Some(debug) = _cache.debug.get_mut(&debug_info) {
                        debug.count += 1;
                    } else {
                        _cache.debug.insert(debug_info.clone(), debug_info);
                    }
                }
            }
        }
    }
}

#[cfg(feature = "wgpu_debug")]
fn end_debug_queue(
    dev: &WgpuDevice,
    length: u32,
    global_index: u32,
    encoder: &mut wgpu::CommandEncoder,
    query_set: &wgpu::QuerySet,
) {
    if global_index % 256 != 0 {
        panic!("global_index was:{global_index}")
    }

    encoder.resolve_query_set(
        query_set,
        0..length,
        &dev.debug.query_set_buffer,
        global_index as u64,
    );
    let global_index = global_index + (length * 8);

    let remainder = global_index % 256;
    let global_index = if remainder == 0 {
        global_index
    } else {
        global_index + (256 - remainder)
    };
    dev.debug
        .counter
        .store(global_index, std::sync::atomic::Ordering::Relaxed);
}


/**************** WGPU FUNCTIONS: ****************/ 

#[instrument(skip(device, shader))]
///Creates a wgpu shaderModule
pub fn get_shader(device: &wgpu::Device, shader: &str) -> wgpu::ShaderModule {
    //since wgpu v25.x the performance drastically decreased (e.g. llama2-c 15-m model has:
    // wgpu v24.0.5: (~332,284 token/sec)
    // wgpu v25.0.2: (~215,598 token/sec)
    // wgpu v27.0.1: (~211,38 token/sec)
    // in the release notes of wgpu v25.x there is a note:
    //  "Ensure loops generated by SPIR-V and HLSL Naga backends are bounded...
    //     Note that this may have a performance cost. ... this can be disabled by using `Device::create_shader_module_trusted()`"

    //to mitigate this performance drop, we use create_shader_module_trusted without bounds checks:
    //wgpu v27.0.1(bounds_checks: false, force_loop_bounding: false): (~330,496 token/sec)

    unsafe{
        let cs_module = device.create_shader_module_trusted(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader)),
        }, wgpu::ShaderRuntimeChecks { bounds_checks: false, force_loop_bounding: false });
        cs_module
    }
}


#[instrument(skip(dev, size))]
///Creates a wgpu buffer
pub fn create_buffer(dev: &WgpuDevice, size: u64) -> wgpu::Buffer {
    dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

#[instrument(skip(dev, cache, bindgroup))]
///Creates a wgpu bindgroup
pub fn create_bindgroup(
    dev: &WgpuDevice,
    bindgroup: CachedBindgroupFull,
    cache: &ModelCache,
) -> wgpu::BindGroup {
    let buffer_meta = &dev.meta_buffer;

    let meta_binding = wgpu::BufferBinding {
        buffer: buffer_meta,
        offset: 0,
        size: Some(NonZeroU64::new(256).unwrap()),
    };
    let meta_binding = wgpu::BindingResource::Buffer(meta_binding);

    let meta_entry = wgpu::BindGroupEntry {
        binding: 1,
        resource: meta_binding,
    };

    let bind_group_layout: &wgpu::BindGroupLayout = match bindgroup.get_input() {
        BindgroupInputBase::Bindgroup0(alignment) => &dev.bindgroup_layouts[*alignment].0,
        BindgroupInputBase::Bindgroup1(_, alignment) => &dev.bindgroup_layouts[*alignment].0,
        BindgroupInputBase::Bindgroup2(_, _, alignment) => &dev.bindgroup_layouts[*alignment].0,
        BindgroupInputBase::Bindgroup3(_, _, _, alignment) => &dev.bindgroup_layouts[*alignment].0,
    };

    let buffer_dest = bindgroup.get_dest();

    let buffer_resource = cache
        .buffers
        .get_buffer(buffer_dest)
        .expect("buffer_dest could not be found")
        .buffer()
        .as_entire_binding();

    let dest_buffer_bingdgroup_entry = wgpu::BindGroupEntry {
        binding: 0,
        resource: buffer_resource,
    };

    let create_buffer_entry = |binding: u32, buffer_id: &CachedBufferId| -> wgpu::BindGroupEntry {
        wgpu::BindGroupEntry {
            binding,
            resource: cache
                .buffers
                .get_buffer(buffer_id)
                .unwrap_or_else(|| panic!("Buffer with ID {:?} could not be found", buffer_id))
                .buffer()
                .as_entire_binding(),
        }
    };

    match bindgroup.get_input() {
        CachedBindgroupInput::Bindgroup0(_) => {
            let entries = &[
                dest_buffer_bingdgroup_entry,
                meta_entry,
            ];
            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: bind_group_layout,
                entries,
            })
        }
        CachedBindgroupInput::Bindgroup1(buffer_input1, _) => {
            let entries = &[
                dest_buffer_bingdgroup_entry,
                meta_entry,
                create_buffer_entry(2, buffer_input1)
            ];
            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: bind_group_layout,
                entries,
            })
        }
        CachedBindgroupInput::Bindgroup2(buffer_input1, buffer_input2, _) => {
            let entries = &[
                dest_buffer_bingdgroup_entry,
                meta_entry,
                create_buffer_entry(2, buffer_input1),
                create_buffer_entry(3, buffer_input2)
            ];
            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: bind_group_layout,
                entries,
            })
        }
        CachedBindgroupInput::Bindgroup3(buffer_input1, buffer_input2, buffer_input3, _) => {
            let entries = &[
                dest_buffer_bingdgroup_entry,
                meta_entry,
                create_buffer_entry(2, buffer_input1),
                create_buffer_entry(3, buffer_input2),
                create_buffer_entry(4, buffer_input3)
            ];
            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: bind_group_layout,
                entries,
            })
        }
    }
}

#[instrument(skip(dev))]
pub fn synchronize(dev: &WgpuDevice) -> crate::Result<()> {
    let mut command_queue = dev.command_queue.lock().unwrap();
    if !command_queue.command_queue.is_empty() {
        flush_gpu_command(dev, &mut command_queue)?;
        if let Some(buffer) = &command_queue.last_buffer {
            let cache = dev.cache.lock().unwrap();
            if let Some(buffer) = cache.buffers.get_buffer(buffer) {
                copy_to_staging_probe(dev, buffer.buffer());
            }
        }

        return pollster::block_on(synchronize_device(dev));
    }
    Ok(())
}

#[instrument(skip(dev))]
pub async fn synchronize_async(dev: &WgpuDevice) -> crate::Result<()> {
    let mut command_queue = dev.command_queue.lock().unwrap();
    if !command_queue.command_queue.is_empty() {
        flush_gpu_command_async(dev, &mut command_queue).await?;
        if let Some(buffer) = &command_queue.last_buffer {
            let cache = dev.cache.lock().unwrap();
            if let Some(buffer) = cache.buffers.get_buffer(buffer) {
                copy_to_staging_probe(dev, buffer.buffer());
            }
        }
        return synchronize_device(dev).await;
    }
    Ok(())
}

#[instrument(skip(dev))]
async fn synchronize_device(dev: &WgpuDevice) -> crate::Result<()> {
    wait_for_gpu_buffer_async(dev).await
}


///Copies 4 bytes of the buffer to a staging buffer
///This is used as a workaround to wait, until all calculation for the buffer has been finished.  
pub fn copy_to_staging_probe(dev: &WgpuDevice, buffer: &wgpu::Buffer) {
    let mut encoder = dev
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    let staging_buffer = &dev.staging_probe_buffer;

    encoder.copy_buffer_to_buffer(buffer, 0, staging_buffer, 0, 4);

    // Submits command encoder for processing
    dev.queue.submit(Some(encoder.finish()));
}

#[instrument(skip(dev))]
//wait for the current staging buffer,
//the buffer one wants to
pub async fn wait_for_gpu_buffer_async(dev: &WgpuDevice) -> crate::Result<()> {
    let staging_buffer = &dev.staging_probe_buffer;

    // Note that we're not calling `.await` here.
    let buffer_slice = staging_buffer.slice(..);
    // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    dev.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    // Awaits until `buffer_future` can be read from
    if let Ok(Ok(())) = receiver.recv_async().await {
        staging_buffer.unmap(); // Unmaps buffer from memory
                                // If you are familiar with C++ these 2 lines can be thought of similarly to:
                                //   delete myPointer;
                                //   myPointer = NULL;
                                // It effectively frees the memory
                                // Returns data from buffer
        Ok(())
    } else {
        panic!("failed to run compute on gpu!")
    }
}

#[instrument(skip(dev, buffer))]
pub async fn read_from_buffer_async<T: bytemuck::Pod>(
    dev: &WgpuDevice,
    buffer: &wgpu::Buffer,
) -> crate::Result<Vec<T>> {
    let dest_size = buffer.size();

    let staging_buffer = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: dest_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = dev
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, dest_size);

    // Submits command encoder for processing
    dev.queue.submit(Some(encoder.finish()));

    read_from_staging_buffer_async(dev, staging_buffer).await
}

fn copy_buffer_to_staging_buffer(dev: &WgpuDevice, buffer : &wgpu::Buffer) -> wgpu::Buffer{
    let dest_size = buffer.size();
    let staging_buffer = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: dest_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = dev
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, dest_size);

    // Submits command encoder for processing
    dev.queue.submit(Some(encoder.finish()));
    staging_buffer
} 

#[instrument(skip(dev))]
pub async fn read_from_buffer_reference_async<T: bytemuck::Pod>(
    dev: &WgpuDevice,
    buffer_reference : BufferReferenceId
) -> crate::Result<Vec<T>> {

    let mut command_queue = dev.command_queue.lock().unwrap();
    flush_gpu_command_async(dev, &mut command_queue).await?; //send all previous commands to the gpu
    let staging_buffer;
    {
        let cache = dev.cache.lock().unwrap();
        if let Some(buffer) = cache.buffer_reference.get(&buffer_reference) {
            let buffer_storage = buffer.cached_buffer_id();
            if buffer_storage.is_valid() {
                if let Some(buffer) = cache.buffers.get_buffer(buffer_storage) {
                    staging_buffer = copy_buffer_to_staging_buffer(dev, buffer.buffer());
                } else {
                    panic!("Unespected error at read_data from gpu. Tensor WgpuStorage did not Point to a wgpu Buffer")
                }
            } else {
                panic!("Unespected error at read_data from gpu. Tensor WgpuStorage did not Point to a wgpu Buffer")
            }
        } else {
            panic!("Unespected error at read_data from gpu. Tensor WgpuStorage did not Point to a wgpu Buffer Reference")
        }
    }
   
    read_from_staging_buffer_async(dev, staging_buffer).await
}


#[instrument(skip(dev, staging_buffer))]
async fn read_from_staging_buffer_async<T: bytemuck::Pod>(
    dev: &WgpuDevice,
    staging_buffer : wgpu::Buffer
) -> crate::Result<Vec<T>> {
    // Note that we're not calling `.await` here.
    let buffer_slice = staging_buffer.slice(..);
    // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
        sender
            .send(v)
            .expect("error in read_data could not send flume")
    });

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    dev.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    // Awaits until `buffer_future` can be read from
    if let Ok(Ok(())) = receiver.recv_async().await {
        // Gets contents of buffer
        let data = buffer_slice.get_mapped_range();
        // Since contents are got in bytes, this converts these bytes back to u32
        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(data);
        staging_buffer.unmap(); // Unmaps buffer from memory
                                // If you are familiar with C++ these 2 lines can be thought of similarly to:
                                //   delete myPointer;
                                //   myPointer = NULL;
                                // It effectively frees the memory

        // Returns data from buffer
        Ok(result)
    } else {
        panic!("failed to run compute on gpu!")
    }
}