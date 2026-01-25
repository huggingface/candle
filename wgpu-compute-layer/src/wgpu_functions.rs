use rustc_hash::FxHasher;
use std::{
    hash::{Hash, Hasher},
    num::NonZeroU64,
};

use super::{
    cache::{
        BindgroupInputBase, BindgroupReferenceFull, BindgroupReferenceInput, BufferReferenceId,
        CachedBindgroupFull, CachedBindgroupInput, CachedBufferId, ModelCache,
    },
    queue_buffer::{MlQueue, QueueBuffer, QueueBufferInner},
    util::{FixedArray, ToF64, ToU32},
    WgpuDevice,
};
use crate::util::ReferenceTrait;
use tracing::{instrument, span, Level};

use crate::DType;
use std::borrow::Cow;

///Helper Type MetaArray, for constructing the MetaBuffer
///The MetaBuffer is used to pass Parameters to the Kernel.
///Paramerters for multiple Commands are grouped together in this MetaArray.
#[derive(Debug)]
pub struct MetaArray(pub Vec<u32>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct KernelConstId(pub &'static str);

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
///Helper Array to Construct Kernel Constants.
///Kernel Constants are compiled into the kernel.
pub struct ConstArray(pub FixedArray<(KernelConstId, u32), 32>);

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

    pub fn insert<T: ToU32>(&mut self, key: KernelConstId, value: T) {
        self.0.push((key, value.to_u32()));
    }

    pub fn to_vec(&self) -> Vec<(&'static str, f64)> {
        Vec::from_iter(self.0.iter().map(|(k, v)| (k.0, v.to_f64())))
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

impl WgpuDevice {
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
        command_queue.get_meta_mut().extend(std::iter::repeat_n(
            0,
            (meta_offset - meta_array_length) as usize,
        ));

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
            (crate::DType::U8, _) => Err(crate::Error::from(format!(
                "Dtype {:?} not supported on wgpu",
                &dtype
            ))),
            (_, false) => Err(crate::Error::from(format!(
                "Dtype {:?} not supported on this wgpu device",
                dtype
            ))),
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

#[instrument(skip(dev, cache, command_queue, current_meta, meta_array))]
///Builds Command Buffer and writes meta_array to the gpu
fn get_command_buffer(
    dev: &WgpuDevice,
    meta_array: &[u32],
    command_queue: &[MlQueue],
    current_meta: usize,
    cache: &mut ModelCache,
    buffer_to_map: Option<(CachedBufferId, &wgpu::Buffer, u64)>,
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
    let mut last_pipeline = None;
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

                        if last_pipeline != Some((q.pipeline.0, q.pipeline.1)) {
                            cpass.set_pipeline(pipeline);
                            last_pipeline = Some((q.pipeline.0, q.pipeline.1));
                        }

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
                            use crate::debug_info;

                            cpass.write_timestamp(query_set, debug_index + 1);
                            dev.debug.insert_info(
                                global_index + debug_index * 8,
                                debug_info::ShaderPerformanceMeasurmentDebugInfo {
                                    pipeline: format!(
                                        "Shader: '{}', Pipeline: '{}', {}",
                                        cache
                                            .shader
                                            .loader_cache
                                            .get_shader_name(q.pipeline.0.get_shader()),
                                        cache.shader.loader_cache.get_entry_point(q.pipeline.0),
                                        q.debug.to_owned().map_or("".to_string(), |s| s)
                                    ),
                                    workload_size: q.workload_size as u64,
                                    x: q.x,
                                    y: q.y,
                                    z: q.z,
                                },
                            );
                            debug_index += 2;

                            if cache.full_recording.should_record {
                                use crate::wgpu_functions;

                                let debug_info = crate::device::DebugPipelineRecording {
                                    x: q.x,
                                    y: q.y,
                                    z: q.z,
                                    pipeline: q.pipeline.clone(),
                                    meta: meta_array[meta as usize..].to_vec(),
                                    bindgroup: q.bindgroup.clone(),
                                    count: 1,
                                };

                                fn get_buffer_data(
                                    dev: &WgpuDevice,
                                    buffer_reference: CachedBufferId,
                                    cache: &mut ModelCache,
                                ) -> crate::Result<super::debug_info::NumericArray>
                                {
                                    #[cfg(not(target_arch = "wasm32"))]
                                    {
                                        let staging_buffer;
                                        if buffer_reference.is_valid() {
                                            if let Some(buffer) =
                                                cache.buffers.get_buffer(&buffer_reference)
                                            {
                                                use crate::wgpu_functions;
                                                staging_buffer = create_staging_buffer(
                                                    dev,
                                                    buffer.buffer().size(),
                                                );
                                                wgpu_functions::copy_buffer(
                                                    dev,
                                                    buffer.buffer(),
                                                    &staging_buffer,
                                                    buffer.buffer().size(),
                                                );
                                            } else {
                                                panic!("Unespected error at read_data from gpu. Tensor WgpuStorage did not Point to a wgpu Buffer")
                                            }
                                        } else {
                                            panic!("Unespected error at read_data from gpu. Tensor WgpuStorage did not Point to a wgpu Buffer")
                                        }

                                        let data = pollster::block_on(
                                            wgpu_functions::read_from_staging_buffer_async::<u8>(
                                                dev,
                                                staging_buffer,
                                            ),
                                        )?;

                                        Ok(super::debug_info::NumericArray::U8(data))
                                    }
                                    #[cfg(target_arch = "wasm32")]
                                    {
                                        return crate::bail!(
                                            "Synchronous read not supported on wasm32"
                                        );
                                    }
                                }

                                let buffer_input1 = buffers.get_input().get_input1().cloned();
                                let buffer_input2 = buffers.get_input().get_input2().cloned();
                                let buffer_input3 = buffers.get_input().get_input3().cloned();
                                let vd1 = get_buffer_data(dev, vd, cache)
                                    .expect("Expect to Read the Buffer");
                                let v_input1 = buffer_input1.map(|buffer| {
                                    get_buffer_data(dev, buffer, cache)
                                        .expect("Expect to Read the Buffer")
                                });
                                let v_input2 = buffer_input2.map(|buffer| {
                                    get_buffer_data(dev, buffer, cache)
                                        .expect("Expect to Read the Buffer")
                                });
                                let v_input3 = buffer_input3.map(|buffer| {
                                    get_buffer_data(dev, buffer, cache)
                                        .expect("Expect to Read the Buffer")
                                });

                                let data = super::debug_info::DebugPipelineRecordingWithData {
                                    recording: debug_info,
                                    v_dest: vd1,
                                    v_input1,
                                    v_input2,
                                    v_input3,
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

    #[cfg(feature = "wgpu_debug")]
    end_debug_queue(
        dev,
        command_queue.len() as u32 * 2,
        global_index,
        &mut encoder,
        query_set,
    );

    if let Some((buffer_to_map, staging_buffer, size)) = buffer_to_map {
        if let Some(buffer) = cache.buffers.get_buffer(&buffer_to_map) {
            encoder.copy_buffer_to_buffer(buffer.buffer(), 0, staging_buffer, 0, size);
        } else {
            panic!("Unespected error at read_data from gpu. Tensor WgpuStorage did not Point to a wgpu Buffer")
        }
    }

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
        use crate::cache::AverageBufferInfo;

        let mut buffers: std::collections::HashMap<(u32, bool), u32> =
            std::collections::HashMap::new();
        for (_id, cached_buffer) in cache.buffers.iter_buffers() {
            let size = cached_buffer.buffer().size() as u32;

            // Adjust this depending on how you detect "free"
            let is_free = cached_buffer.is_free();

            *buffers.entry((size, is_free)).or_insert(0) += 1;
        }

        let debug_buffer_info = crate::cache::DebugBufferUsage {
            memory_alloc: cache.buffers.buffer_memory(),
            memory_free: cache.buffers.buffer_free_memory(),
            buffers: buffers
                .into_iter()
                .map(|(key, value)| AverageBufferInfo {
                    count: value,
                    is_free: key.1,
                    size: key.0,
                })
                .collect(),
            command_buffer_id: global_index,
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
            if ele_size >= wgpu::QUERY_SET_MAX_QUERIES as usize / 2 - 1 {
                break;
            }
            if cache.full_recording.should_record && ele_size > 1 {
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

                fn should_optimize_inplace(
                    cache: &mut ModelCache,
                    vdest_id: &BufferReferenceId,
                    v1_id: &BufferReferenceId,
                    command_index: u32,
                ) -> bool {
                    let vdest = cache.buffer_reference.get(vdest_id);
                    let v1 = cache.buffer_reference.get(v1_id);
                    if let Some(vdest) = vdest {
                        if let Some(v1) = v1 {
                            if !v1.cached_buffer_id().is_valid() {
                                panic!("while optimizing: input buffer {:?}({:?}) storage was not set in {command_index}", v1, v1_id)
                            }

                            //this buffer was last used in this pipeline
                            if v1.last_used() == command_index
                                && vdest.size() <= v1.size()
                                && !vdest.cached_buffer_id().is_valid()
                            {
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

                if q.pipeline.2.input1_inplaceable || q.pipeline.2.input2_inplaceable {
                    let loader = q.pipeline.0.get_shader().get_loader();
                    if let Some(plan) = cache.shader.loader_cache.rewrite_plan(
                        loader,
                        crate::shader_loader::InplaceRewriteDesc {
                            pipeline: q.pipeline.0,
                            bindgroup: &q.bindgroup,
                            inplace_flags: q.pipeline.2,
                        },
                    ) {
                        match plan {
                            crate::shader_loader::RewritePlan::InplaceDispatch {
                                new_pipeline,
                                new_bindgroup,
                                replaced_input,
                            } => {
                                let vinput_id = match replaced_input {
                                    crate::shader_loader::ReplacedInput::Input1 => {
                                        q.bindgroup.get_input().get_input1()
                                    }
                                    crate::shader_loader::ReplacedInput::Input2 => {
                                        q.bindgroup.get_input().get_input2()
                                    }
                                };
                                if let Some(vinput_id) = vinput_id {
                                    if should_optimize_inplace(
                                        cache,
                                        q.bindgroup.get_dest(),
                                        vinput_id,
                                        command_index,
                                    ) {
                                        input_replaced_buffer = *vinput_id;
                                        q.pipeline.0 = new_pipeline;
                                        q.bindgroup = new_bindgroup;
                                    }
                                }
                            }
                            crate::shader_loader::RewritePlan::ElideDispatch { replaced_input } => {
                                let vinput_id = match replaced_input {
                                    crate::shader_loader::ReplacedInput::Input1 => {
                                        q.bindgroup.get_input().get_input1()
                                    }
                                    crate::shader_loader::ReplacedInput::Input2 => {
                                        q.bindgroup.get_input().get_input2()
                                    }
                                };
                                if let Some(vinput_id) = vinput_id {
                                    let v1 = cache.buffer_reference.get(vinput_id);
                                    if let Some(v1) = v1 {
                                        let vdest_id = bindgroup_reference.get_dest();

                                        let v1_cached_id = *v1.cached_buffer_id();
                                        let v1_size = v1.size();
                                        //this buffer was last used in this pipeline
                                        if v1.last_used() == command_index {
                                            let vdest = cache.buffer_reference.get_mut(vdest_id);
                                            if let Some(vdest) = vdest {
                                                if vdest.size() <= v1_size
                                                    && !vdest.cached_buffer_id().is_valid()
                                                {
                                                    vdest.set_cached_buffer_id(v1_cached_id);

                                                    cache.copy_inplace_counter += 1;
                                                    optimize_copy_inplace = true;
                                                }
                                            }

                                            if optimize_copy_inplace {
                                                let v1 = cache.buffer_reference.get_mut(vinput_id);
                                                if let Some(v1) = v1 {
                                                    v1.set_cached_buffer_id(CachedBufferId::new(
                                                        0, 0,
                                                    ));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                drop(_enter1);

                #[instrument(skip(
                    cache,
                    bindgroup_reference,
                    command_index,
                    input_replaced_buffer
                ))]
                fn check_for_removal(
                    bindgroup_reference: &BindgroupReferenceFull,
                    command_index: u32,
                    cache: &mut ModelCache,
                    input_replaced_buffer: &BufferReferenceId,
                ) {
                    let check_buffer = |buffer_reference: &BufferReferenceId,
                                        cache: &mut ModelCache,
                                        command_index: u32| {
                        if let Some(buffer) = cache.buffer_reference.get_mut(buffer_reference) {
                            if buffer.last_used() <= command_index {
                                //this buffer reference is not used after this:
                                let cached_buffer_id = *buffer.cached_buffer_id();
                                cache.buffer_reference.delete(buffer_reference);
                                if cached_buffer_id.is_valid()
                                    && buffer_reference != input_replaced_buffer
                                {
                                    //if this buffer was replaced by another buffer, the referenced buffer is still not free
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
                    let pipeline =
                        cache
                            .shader
                            .get_pipeline(&dev.device, &q.pipeline, pl, consts)?;

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
        use crate::cache::AverageBufferInfo;

        let mut buffers: std::collections::HashMap<(u32, bool), u32> =
            std::collections::HashMap::new();
        for (_id, cached_buffer) in cache.buffers.iter_buffers() {
            let size = cached_buffer.buffer().size() as u32;
            let is_free = cached_buffer.is_free();
            *buffers.entry((size, is_free)).or_insert(0) += 1;
        }

        let debug_buffer_info = crate::cache::DebugBufferUsage {
            memory_alloc: cache.buffers.buffer_memory(),
            memory_free: cache.buffers.buffer_free_memory(),
            buffers: buffers
                .into_iter()
                .map(|(key, value)| AverageBufferInfo {
                    count: value,
                    is_free: key.1,
                    size: key.0,
                })
                .collect(),
            command_buffer_id: global_index,
        };
        cache.debug_buffer_info.push(debug_buffer_info);
    }

    let meta_size = (*last_meta - current_meta) * 4 + 256 * 3;
    let ele_size = *index - start_index;
    log::trace!("queue {ele_size}, Meta: {meta_size}, workload: {total_workload}, cache_limit: {cache_limit}");

    Ok((cache_limit, total_workload))
}

macro_rules! maybe_await {
    ($expr:expr) => {{
        #[cfg(target_arch = "wasm32")]
        {
            $expr.await
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            $expr
        }
    }};
}

macro_rules! platform_fn {
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident( $($args:tt)* ) -> $ret:ty $body:block
    ) => {
        // Native version (sync)
        $(#[$meta])*
        #[cfg(not(target_arch = "wasm32"))]
        $vis fn $name ( $($args)* ) -> $ret $body

        // Wasm version (async)
        $(#[$meta])*
        #[cfg(target_arch = "wasm32")]
        $vis async fn $name ( $($args)* ) -> $ret $body
    };
}

platform_fn! {
    #[instrument(skip(dev))]
    ///Send queued commands to the GPU,
    pub(crate) fn flush_gpu_command(dev: &WgpuDevice, buffer_id_to_map : Option<(BufferReferenceId, &wgpu::Buffer)>) -> crate::Result<Option<WasmSubmissionIndex>> {
        let queue_buffer = &mut dev.command_queue.lock().unwrap();
        if !queue_buffer.command_queue.is_empty() {
            log::debug!("flush_gpu_command");
            let mut submissions = std::collections::VecDeque::<WasmSubmissionIndex> ::new();
            let mut cache = dev.cache.lock().expect("");
            prepare(dev, queue_buffer, &mut cache);
            {
                let mut start_index = 0;
                let mut index = 0;
                let mut current_meta: usize = 0;
                let mut last_meta: usize = 0;

                while index < queue_buffer.command_queue.len() {
                    let (should_remove_unused, _) = set_buffers(
                        dev,
                        queue_buffer,
                        &mut index,
                        current_meta,
                        &mut last_meta,
                        &mut cache,
                    )?;
                    let last_meta_index = (last_meta + 256 / 4).min(queue_buffer.get_meta().len());

                    let is_last = index >= queue_buffer.command_queue.len();
                    let mut buffer_to_map = None;
                    if is_last{
                        if let Some((buffer_reference, staging_buffer)) = buffer_id_to_map{
                            if let Some(buffer) = cache.buffer_reference.get(&buffer_reference) {
                                let buffer_storage = buffer.cached_buffer_id();
                                if buffer_storage.is_valid() {
                                    buffer_to_map = Some((*buffer_storage, staging_buffer, buffer.size()));
                                } else {
                                    panic!("Unespected error at read_data from gpu. Tensor WgpuStorage did not Point to a wgpu Buffer")
                               }
                            } else {
                                panic!("Unespected error at read_data from gpu. Tensor WgpuStorage did not Point to a wgpu Buffer Reference")
                            }
                        }

                    }

                    let cb = get_command_buffer(
                        dev,
                        &queue_buffer.get_meta()[current_meta..last_meta_index],
                        &queue_buffer.command_queue[start_index..index],
                        current_meta,
                        &mut cache,
                        buffer_to_map
                    );

                    if should_remove_unused {
                        cache.remove_unused();
                    }

                    if !submissions.is_empty() {
                        let submission_to_wait_for = submissions.pop_front().unwrap();
                        maybe_await!(wait_for_submission(dev, submission_to_wait_for))?; //TODO: cargo clippy --target wasm32-unknown-unknown
                                                                                          //      this `MutexGuard` is held across an await point
                    }

                    let span1 = span!(Level::INFO, "Submit");
                    let _enter1: span::Entered<'_> = span1.enter();
                    #[cfg(target_arch = "wasm32")]
                    let wasm_submission_id =
                    {
                        let tracker_clone = dev.submission_tracker.clone();
                        let id = tracker_clone.next_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        cb.on_submitted_work_done(move || {
                            // Mark completion
                            tracker_clone
                                .completed_id
                                .store(id, std::sync::atomic::Ordering::Release);
                            let _ = tracker_clone.tx.send(id);
                        });
                        id
                    };

                    let _submission_id = dev.queue.submit(Some(cb));
                    let submission_index = WasmSubmissionIndex {
                        #[cfg(not(target_arch = "wasm32"))]
                        submission_index: _submission_id,
                        #[cfg(target_arch = "wasm32")]
                        submission_index_wasm: wasm_submission_id,
                    };
                    submissions.push_back(submission_index);
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

            return Ok(submissions.pop_back());
        }
        else if let Some((buffer_reference, staging_buffer)) = buffer_id_to_map{
            let cache = dev.cache.lock().expect("");
            if let Some(buffer) = cache.buffer_reference.get(&buffer_reference) {
                let buffer_storage = buffer.cached_buffer_id();
                if buffer_storage.is_valid() {
                    if let Some(buffer_storage) = cache.buffers.get_buffer(buffer_storage) {
                        copy_buffer(dev, buffer_storage.buffer(), staging_buffer, buffer.size());
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
        Ok(None)
    }
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
                            BindgroupInputBase::Bindgroup0(alginment) => {
                                BindgroupInputBase::Bindgroup0(*alginment)
                            }
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
                                    *alginment,
                                )
                            }
                        },
                    );

                    let mut meta: Vec<u32> =
                        command_buffer.get_meta()[q.meta as usize..next_meta].into();

                    _cache
                        .shader
                        .loader_cache
                        .normalize_debug_meta(q.pipeline.0, &mut meta);

                    let debug_info = crate::device::DebugPipelineRecording {
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
    if !global_index.is_multiple_of(256) {
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
pub(crate) fn get_shader(device: &wgpu::Device, shader: &str) -> wgpu::ShaderModule {
    //since wgpu v25.x the performance drastically decreased (e.g. llama2-c 15-m model has:
    // wgpu v24.0.5: (~332,284 token/sec)
    // wgpu v25.0.2: (~215,598 token/sec)
    // wgpu v27.0.1: (~211,38 token/sec)
    // in the release notes of wgpu v25.x there is a note:
    //  "Ensure loops generated by SPIR-V and HLSL Naga backends are bounded...
    //     Note that this may have a performance cost. ... this can be disabled by using `Device::create_shader_module_trusted()`"

    //to mitigate this performance drop, we use create_shader_module_trusted without bounds checks:
    //wgpu v27.0.1(bounds_checks: false, force_loop_bounding: false): (~330,496 token/sec)

    unsafe {
        let cs_module = device.create_shader_module_trusted(
            wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader)),
            },
            wgpu::ShaderRuntimeChecks {
                bounds_checks: false,
                force_loop_bounding: false,
                ray_query_initialization_tracking: true,
            },
        );
        cs_module
    }
}

#[instrument(skip(dev, size))]
///Creates a wgpu buffer
pub(crate) fn create_buffer(dev: &WgpuDevice, size: u64) -> wgpu::Buffer {
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
pub(crate) fn create_bindgroup(
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
            let entries = &[dest_buffer_bingdgroup_entry, meta_entry];
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
                create_buffer_entry(2, buffer_input1),
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
                create_buffer_entry(3, buffer_input2),
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
                create_buffer_entry(4, buffer_input3),
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
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn synchronize(dev: &WgpuDevice) -> crate::Result<()> {
    if let Some(last_submission_index) = flush_gpu_command(dev, None)? {
        wait_for_submission(dev, last_submission_index)?;
    }
    Ok(())
}

#[instrument(skip(dev))]
pub(crate) async fn synchronize_async(dev: &WgpuDevice) -> crate::Result<()> {
    if let Some(last_submission_index) = maybe_await!(flush_gpu_command(dev, None))? {
        maybe_await!(wait_for_submission(dev, last_submission_index))?;
    }
    Ok(())
}

//when on wasm we wait until the submission_index_wasm has finished by waiting on the on_submitted_work_done callback of the commandBuffer
//otherwise we use the wgpu submission index to poll and wait synchron for the submission_index
pub(crate) struct WasmSubmissionIndex {
    #[cfg(target_arch = "wasm32")]
    submission_index_wasm: u64,
    #[cfg(not(target_arch = "wasm32"))]
    submission_index: wgpu::SubmissionIndex,
}

//when on wasm we wait until the submission_index_wasm has finished by waiting on the on_submitted_work_done callback of the commandBuffer
//otherwise we use the wgpu submission index to poll and wait synchron for the submission_index
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn wait_for_submission(
    dev: &WgpuDevice,
    index: WasmSubmissionIndex,
) -> crate::Result<()> {
    dev.device.poll(wgpu::wgt::PollType::Wait {
        submission_index: Some(index.submission_index),
        timeout: None,
    })?;
    Ok(())
}

#[cfg(target_arch = "wasm32")]
pub(crate) async fn wait_for_submission(
    dev: &WgpuDevice,
    index: WasmSubmissionIndex,
) -> crate::Result<()> {
    // Fast path: already completed
    if dev
        .submission_tracker
        .completed_id
        .load(std::sync::atomic::Ordering::Acquire)
        >= index.submission_index_wasm
    {
        return Ok(());
    }

    // Slow path: wait on channel
    loop {
        let completed = dev.submission_tracker.rx.recv_async().await.unwrap();
        if completed >= index.submission_index_wasm {
            return Ok(());
        }
    }
}

#[instrument(skip(dev, buffer))]
#[cfg(feature = "wgpu_debug")]
pub(crate) async fn read_from_buffer_async<T: bytemuck::Pod>(
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

fn copy_buffer(dev: &WgpuDevice, buffer: &wgpu::Buffer, dest_buffer: &wgpu::Buffer, size: u64) {
    let mut encoder = dev
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    encoder.copy_buffer_to_buffer(buffer, 0, dest_buffer, 0, size);
    dev.queue.submit(Some(encoder.finish()));
}

fn create_staging_buffer(dev: &WgpuDevice, size: u64) -> wgpu::Buffer {
    dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

#[instrument(skip(dev))]
pub(crate) async fn read_from_buffer_reference_async<T: bytemuck::Pod>(
    dev: &WgpuDevice,
    buffer_reference: BufferReferenceId,
) -> crate::Result<Vec<T>> {
    let staging_buffer;
    {
        let cache = dev.cache.lock().unwrap();
        if let Some(buffer) = cache.buffer_reference.get(&buffer_reference) {
            staging_buffer = create_staging_buffer(dev, buffer.size());
        } else {
            panic!("Unespected error at read_data from gpu. Tensor WgpuStorage did not Point to a wgpu Buffer Reference")
        }
    }

    {
        maybe_await!(flush_gpu_command(
            dev,
            Some((buffer_reference, &staging_buffer))
        ))?; //send all previous commands to the gpu
    }

    read_from_staging_buffer_async(dev, staging_buffer).await
}

#[instrument(skip(dev, staging_buffer))]
async fn read_from_staging_buffer_async<T: bytemuck::Pod>(
    dev: &WgpuDevice,
    staging_buffer: wgpu::Buffer,
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
    dev.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();

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
