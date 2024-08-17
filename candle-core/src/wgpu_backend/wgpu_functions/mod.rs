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

use std::{
    hash::{DefaultHasher, Hash, Hasher}, num::NonZeroU64, sync::MutexGuard
};

use tracing::{instrument, span, Level};
use super::{
    cache::{BindgroupInputBase, BindgroupReferenceFull, CachedBindgroupFull, CachedBindgroupInput, CachedBufferId, ModelCache},
    device::{
        BindGroupReference, DispatchedBindgroup, MlQueue, OpIsInplaceable, PipelineType, QueueBuffer
    },
    util::{FixedArray, ToU32},
};
use crate::wgpu_backend::util::ReferenceTrait;


pub use candle_wgpu_kernels::Pipelines as Pipelines;
pub use candle_wgpu_kernels::DType as DType;
pub use crate::wgpu::WgpuDevice as WgpuDevice;
pub use crate::wgpu::cache::BufferReferenceId as BufferReferenceId;

use crate::{wgpu_backend::cache::BindgroupReferenceInput, Layout, WebGpuError};
use std::borrow::Cow;


//use tracing_mutex::stdsync::MutexGuard;

use wgpu::{Queue, ShaderModule};

pub use binary::queue_binary_buffer_from_buffer;
pub use cmp::queue_cmp_buffer_from_buffer;
pub use conv2d::{queue_conv1d, queue_conv1d_transpose, queue_conv2d, queue_conv2d_transpose};
pub use convert::{
    queue_convert_f32_to_u32, queue_convert_f32_to_u8, queue_convert_u32_to_f32,
    queue_convert_u32_to_u8, queue_convert_u8_to_f32,
};
pub use copy::{queue_copy, queue_copy2d, queue_copy3d,queue_copy3d_padded, queue_copy_strided};
pub use gather::{queue_gather, queue_index_add_inplace, queue_scatter_add_inplace};
pub use index_select::queue_index_select;
pub use matmul::queue_matmul_buffer;
pub use pool2d::{queue_avg_pool2d, queue_max_pool2d};
pub use reduce::queue_reduce_from_buffer_op;
pub use rms_norm::queue_rms_norm;
pub use softmax::queue_softmax;
pub use unary::{queue_unary_from_buffer_op, queue_unary_inplace_op};
pub use upsample::{queue_upsample1d, queue_upsample2d};
pub use where_cond::queue_where_cond_u32;

pub const MAX_DISPATCH_SIZE: u32 = 65535;

///Helper Type MetaArray, for constructing the MetaBuffer
#[derive(Debug)]
pub struct MetaArray(pub Vec<u32>);

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ConstArray(pub FixedArray<(candle_wgpu_kernels::Constants, u32), 32>);

pub trait KernelParameterMeta{
    fn write_meta(&self, meta : &mut MetaArray);
}

pub trait KernelParameterConsts{
    fn write_consts(&self, _consts : &mut ConstArray){}
}
pub trait KernelParameter : KernelParameterMeta + KernelParameterConsts{
   
}


impl MetaArray {
    pub fn new(capacity: u32) -> Self {
        MetaArray(Vec::with_capacity(capacity as usize))
    }

    pub fn add<T : KernelParameterMeta>(&mut self, value : T){
        value.write_meta(self);
    }
}

impl<T : ToU32 + Copy> KernelParameterMeta for T{
    fn write_meta(&self, meta : &mut MetaArray) {
        meta.0.push((*self).to_u32());
    }
}

impl ConstArray {
    pub fn new() -> Self {
        ConstArray(FixedArray::new())
    }

    pub fn add<T : KernelParameterConsts>(&mut self, value : T){
        value.write_consts(self);
    }

    pub fn insert<T : ToU32>(&mut self, key : candle_wgpu_kernels::Constants, value : T){
        self.0.push((key, value.to_u32()));
    }
}


const WORKGROUP_SIZE: u32 = 64;

pub fn get_dtype(dtype : crate::DType) -> crate::Result<DType>{
    match dtype{
        crate::DType::U8 =>  Ok(DType::U8),
        crate::DType::U32 => Ok(DType::U32),
        crate::DType::F32 =>  Ok(DType::F32),
        _ => Err(crate::Error::WebGpu(WebGpuError::from(format!("Dtype {:?} not supported on wgpu", dtype)))),
    }
}

#[instrument]
pub fn get_shader(device: &wgpu::Device, shader: &'static str) -> ShaderModule {
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader)),
    });
    return cs_module;
}


fn enqueue_workgroups(
    command_queue: MutexGuard<QueueBuffer>,
    pipeline: PipelineType,
    bind_group: BindGroupReference,
    x: u32,
    y: u32,
    z: u32,
    workload_size : usize
) {
    enqueue_workgroups_extra(command_queue, pipeline, bind_group, x, y, z, workload_size, #[cfg(feature = "wgpu_debug")]None)
}

fn enqueue_workgroups_extra(
    mut command_queue: MutexGuard<QueueBuffer>,
    pipeline: PipelineType,
    bind_group: BindGroupReference,
    x: u32,
    y: u32,
    z: u32,
    workload_size : usize,
    #[cfg(feature = "wgpu_debug")] _debug: Option<String>,
) {

    //println!("queueing {:?}, bindgroup: {:?}", pipeline, bind_group);


    if y > MAX_DISPATCH_SIZE || z > MAX_DISPATCH_SIZE  || x > MAX_DISPATCH_SIZE {
        panic!("can not queue y or z higher than 65535 x:{x}, y:{y}, z:{z}, pipeline: {:?}", pipeline);
    }
    let q = MlQueue::Dispatch(super::device::MlQueueDispatch {
        x,
        y,
        z,
        pipeline: pipeline.clone(),
        pipeline_cached : None,
        bindgroup: DispatchedBindgroup::BindgroupReference(bind_group),
        meta: command_queue.current_meta,
        workload_size,
        #[cfg(feature = "wgpu_debug")]
        debug : _debug
    });
    command_queue.command_queue.push(q);
}

fn next_divisible_by_n<T : num_traits::Num + Clone>(value: T, n: T) -> T {
    if n.is_zero(){
        panic!("n must be a non-zero integer");
    }

    if (value.clone() % n.clone()).is_zero() {
        value
    } else {
        value.clone() + (n.clone() - value % n)
    }
}

fn get_meta(dev: &WgpuDevice) -> MutexGuard<QueueBuffer> {
    let mut command_queue = dev.command_queue.lock().expect("could not get meta command_queue lock");
    let meta_array_length = command_queue.get_meta().len() as i32;
    let meta_offset = next_divisible_by_n(
        meta_array_length,
        dev.device_limits.min_storage_buffer_offset_alignment as i32 / 4,
    );
    command_queue.current_meta = meta_offset as u32;
    command_queue
        .get_meta_mut()
        .extend(std::iter::repeat(0).take((meta_offset - meta_array_length) as usize));

    return command_queue;
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
        &query_set,
        0..length,
        &dev.debug.query_set_buffer,
        global_index as u64,
    );
    let global_index = global_index + (length * 8) as u32;

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

#[instrument]
fn get_command_buffer(
    dev: &WgpuDevice,
    meta_array: &[u32],
    command_queue: &[MlQueue],
    current_meta: usize,
    waiting_buffer : &Option<CachedBufferId>, //a buffer, we want to wait for, after all commands have been queued
    cache : &mut ModelCache
) -> wgpu::CommandBuffer {
    #[cfg(feature = "wgpu_debug")]
    let query_set = &dev.debug.query_set;

    #[cfg(feature = "wgpu_debug")]
    let global_index = dev.debug.counter.load(std::sync::atomic::Ordering::Relaxed);
    
    #[cfg(feature = "wgpu_debug")]
    let mut debug_index = 0;

    let span1 = span!(Level::INFO, "Write Metabuffer");
    let _enter1 = span1.enter();

    let data = bytemuck::cast_slice(&meta_array);
    if data.len() as u32 + 256 > dev.configuration.meta_buffer_size{
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

    for q in command_queue.iter(){
        match q {
            MlQueue::Dispatch(q) => {
                if let DispatchedBindgroup::CachedBindgroup(bindgroup) = &q.bindgroup {
                    if let Some(pipeline) = &q.pipeline_cached{
                        let qx = q.x;
                        let qy = q.y;
                        let qz = q.z;
                        let meta = q.meta - current_meta as u32;

                        #[cfg(feature = "wgpu_debug")]
                        cpass.write_timestamp(&query_set, debug_index);
                        let span1 = span!(Level::INFO, "Set Pipeline");
                        let _enter1 = span1.enter();
                        cpass.set_pipeline(&pipeline);
                        drop(_enter1);

                        if meta * 4 >= dev.configuration.meta_buffer_size - 256 {
                            panic!(
                                "meta is to big!: meta was {meta}, q.meta: {}/{current_meta}",
                                q.meta
                            );
                        }

                        let span1 = span!(Level::INFO, "Set Bindgroup");
                        let _enter1 = span1.enter();




                        let bindgroup = cache.bindgroups.get_bindgroup(&bindgroup).expect("bindgroup could not be found!");

                        let buffers = bindgroup.buffer();
                        let vd = buffers.get_dest();
                        match buffers.get_input(){
                            BindgroupInputBase::Bindgroup0 => {},
                            BindgroupInputBase::Bindgroup1(v1, _) => {
                                if v1 == vd{
                                    panic!("B1: output and input are equal");
                                }
                            },
                            BindgroupInputBase::Bindgroup2(v1, v2, _) => {
                                if v1 == vd{
                                    panic!("B2: output and input1 are equal");
                                }
                                if v2 == vd{
                                    panic!("B2: output and input2 are equal");
                                }
                                // if v1 == v2{
                                //     panic!("B2: input1 and input2 are equal");
                                // }
                            },
                            BindgroupInputBase::Bindgroup3(v1, v2, v3) => {

                                if v1 == vd{
                                    panic!("B3: output and input1 are equal");
                                }
                                // if v1 == v2{
                                //     panic!("B3: input1 and input2 are equal");
                                // }
                                // if v1 == v3{
                                //     panic!("B3: input1 and input3 are equal");
                                // }

                                if v2 == vd{
                                    panic!("B3: output and input2 are equal");
                                }
                                // if v2 == v3{
                                //     panic!("B3: input3 and input2 are equal");
                                // }

                                if v3 == vd{
                                    panic!("B3: input3 and output are equal");
                                }

                            },
                        }

                        cpass.set_bind_group(0, &bindgroup.bindgroup(), &[meta * 4]);
                        drop(_enter1);
                        
                        let span1 = span!(Level::INFO, "Dispatch Workgroups");
                        let _enter1 = span1.enter();
                        cpass.dispatch_workgroups(qx, qy, qz);
                        drop(_enter1);
                        
                        
                        #[cfg(feature = "wgpu_debug")]
                        {
                            cpass.write_timestamp(&query_set, debug_index + 1);
                            dev.debug.insert_info(global_index + debug_index * 8,(
                                    format!("Pipeline: {:?}, {}", q.pipeline.0, q.debug.to_owned().map_or("".to_string(), |s| s)),
                                    q.workload_size as u64,
                                    q.x,
                                    q.y,
                                    q.z,
                                ),
                            );
                            debug_index += 2;
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

    if let Some(waiting_buffer) = waiting_buffer  {
        let staging_buffer = &dev.staging_probe_buffer;

        if let Some(buffer) = cache.buffers.get_buffer(waiting_buffer){
            encoder.copy_buffer_to_buffer(&buffer.buffer(), 0, &staging_buffer, 0, 4);
        }
    }


    #[cfg(feature = "wgpu_debug")]
    end_debug_queue(
        dev,
        command_queue.len() as u32 * 2,
        global_index,
        &mut encoder,
        &query_set,
    );

    let span1 = span!(Level::INFO, "Encoder Finish");
    let _enter1 = span1.enter();
    let result = encoder.finish();
    drop(_enter1);
    return result;
}

#[instrument]
fn prepare(dev: &WgpuDevice, queue_buffer: &mut QueueBuffer, cache : &mut ModelCache){
//   //let mut most_needed_storage;
//    //let mut total_used_storage;
    
    let global_index = queue_buffer.global_command_index();

    let queue = &mut queue_buffer.command_queue;
    {
        let mut hasher = DefaultHasher::new();
        for q in queue.iter() {
            match q {
                MlQueue::Dispatch(q) => {
                    q.pipeline.hash(&mut hasher);
                }
            }
        }

        let current_hash = hasher.finish();
        //let mut cache = dev.cache.lock().expect("flush gpu_commadn could not lock cache");
        cache.mappings.set_current_buffer_mapping(current_hash);

//        //total_used_storage = cache.buffers.total_used(); //the total amount of memory acutally used
//        //most_needed_storage = total_used_storage;

        let deleted_entries = cache.buffer_reference.get_deletion_entries();

        for entry in deleted_entries.iter(){
            if let Some(buffer_reference) = cache.buffer_reference.get_mut(&entry){
                buffer_reference.set_referenced_by_candle_storage(false);
                buffer_reference.set_last_used(0); //if this buffer will not be used in this queue, we can delete the queue and free the used buffer. 
            }
        }

      
        for (index, q) in queue.iter().enumerate() {
            let command_index = global_index + index as u32;

            let mut check_buffer = |buffer_reference|{
                if let Some(buffer) = cache.buffer_reference.get_mut(buffer_reference){
                    if buffer.first_used() == 0{ //not alreaedy set:
                        buffer.set_first_used(command_index);
                    }
                    if buffer.last_used() < command_index{
                        buffer.set_last_used(command_index);
                    }
                }
            };
            match q{
                MlQueue::Dispatch(q) => {
                    match &q.bindgroup{
                        DispatchedBindgroup::BindgroupReference(bindgroup) => {
                            let dest = bindgroup.get_dest();
                            let input = bindgroup.get_input();
                            
                            check_buffer(dest);

                            match input {
                                BindgroupInputBase::Bindgroup0 => {},
                                BindgroupInputBase::Bindgroup1(v1, _) => {
                                    check_buffer(v1);
                                },
                                BindgroupInputBase::Bindgroup2(v1,v2, _) => {
                                    check_buffer(v1);check_buffer(v2);
                                },
                                BindgroupInputBase::Bindgroup3(v1, v2,v3) => {
                                    check_buffer(v1);check_buffer(v2);check_buffer(v3);
                                },
                            }
                        },
                        DispatchedBindgroup::CachedBindgroup(_) => panic!("not expected"),
                        DispatchedBindgroup::None => continue,
                    }

                }
            }
        }

        for entry in deleted_entries.iter(){
            if let Some(buffer_reference) = cache.buffer_reference.get_mut(&entry){
                if buffer_reference.last_used() == 0{ //This buffer was not used in the queue -> we can remove it!
                    //println!("buffer reference {:?} has last used == 0", entry);
                    let buffer_cached_id = buffer_reference.cached_buffer_id().clone();
                    if buffer_reference.cached_buffer_id().is_valid(){
                        cache.buffers.free_buffer(&buffer_cached_id);
                    }
                    cache.buffer_reference.delete(entry);
                }
            }
        }

        // for (index, q) in queue.iter().enumerate() {
        //     let command_index = global_index + index as u32;

        //     let mut check_buffer = |buffer_id: &BufferReferenceId| {
        //         let buffer = cache.buffer_reference.get(buffer_id).unwrap();
                
        //         if buffer.first_used() == command_index{
        //             total_used_storage += buffer.size();
        //         }

        //         if total_used_storage > most_needed_storage {
        //             most_needed_storage = total_used_storage;
        //         }

        //         if buffer.last_used() == command_index{
        //             total_used_storage -= buffer.size();
        //         }
        //     };
        //     match q {
        //         MlQueue::Dispatch(q) => match &q.bindgroup {
        //             DispatchedBindgroup::BindgroupReference(br) => {
        //                 check_buffer(br.get_dest());
        //                 match br.get_input() {
        //                     BindgroupInputBase::Bindgroup0 => {}
        //                     BindgroupInputBase::Bindgroup1(v1, _) => {
        //                         check_buffer(v1);
        //                     }
        //                     BindgroupInputBase::Bindgroup2(v1, v2, _) => {
        //                         check_buffer(v1);
        //                         check_buffer(v2);
        //                     }
        //                     BindgroupInputBase::Bindgroup3(v1, v2, v3) => {
        //                         check_buffer(v1);
        //                         check_buffer(v2);
        //                         check_buffer(v3);
        //                     }
        //                 }
        //             }
        //             DispatchedBindgroup::None => {continue;},
        //             DispatchedBindgroup::CachedBindgroup(_) => todo!(),
        //         },
        //     }
        // }

        //allow 25% margin more:
        // let most_needed_storage = (most_needed_storage  * 5) / 4;
        
      
        // if most_needed_storage >  cache.buffers.max_memory_allowed(){
        //     cache.buffers.set_max_memory_allowed(most_needed_storage);
        // }
        // else{
        //     let new_size = ((7 *  cache.buffers.max_memory_allowed()) / 8) + (most_needed_storage/8);
        //     cache.buffers.set_max_memory_allowed(new_size);
        // }
        cache.buffers.set_max_memory_allowed(dev.configuration.buffer_cached_max_allowed_size); //8gb TODO: change back

    }
}

#[instrument]
fn set_buffers(dev: &WgpuDevice, command_buffer: &mut QueueBuffer, index : &mut usize, current_meta: usize, last_meta : &mut usize, mut cache : &mut ModelCache) -> crate::Result<bool>{
    let global_index = command_buffer.global_command_index();
    let queue = &mut command_buffer.command_queue; 
    let mut cache_limit = false;
    let mut total_workload = 0u64; //we only allow a certain amount of workload per commandBuffer 
    let start_index = *index; 
    //let mut cache = dev.cache.lock().expect("flush gpu_commadn could not lock cache");

    for q in queue[*index..].iter_mut() {
        #[cfg(feature="wgpu_debug")]{
            let ele_size =  *index-start_index;
            if ele_size >= 4095{
                break;
            }
        }

        *index += 1;
        match q {
            MlQueue::Dispatch(q) => {

                let command_index = (*index - 1) as u32 + global_index;
              
                let ele_size =  *index-start_index;
                if (total_workload + q.workload_size as u64)  > dev.configuration.max_workload_size && ele_size > 1 {
                    *index -= 1;
                    break;
                }
                else{
                    total_workload += q.workload_size as u64;
                }
                
                let span1 = span!(Level::INFO, "SetBuffers_Analyse UnaryBuffer ");
                let _enter1 = span1.enter();
                let mut optimize_inplace = false;
                let mut optimize_copy_inplace = false;
                let mut vdest_ref_id = BufferReferenceId::new(0, 0);
                let mut v1_ref_id = BufferReferenceId::new(0, 0);
                
                let mut optmize_inplace = |vdest_id : &BufferReferenceId, v1_id : &BufferReferenceId| -> bool{
                    let vdest = cache.buffer_reference.get(vdest_id);
                    let v1 = cache.buffer_reference.get(v1_id);
                   
                    if let Some(vdest) = vdest{
                        if let Some(v1) = v1{

                            if !v1.cached_buffer_id().is_valid(){
                                panic!("while optimizing: input buffer {:?}({:?}) storage was not set in {command_index}", v1, v1_id)
                            }

                            //this buffer was last used in this pipeline 
                            if v1.last_used() == command_index{
                                if vdest.size() <= v1.size() {
                                    if !vdest.cached_buffer_id().is_valid() {
                                        vdest_ref_id = vdest_id.clone();
                                        v1_ref_id = v1_id.clone();
                                        optimize_inplace = true;
                                        return true;
                                    }
                                }
                            }
                        }
                    }
                    return false;
                };

                if let DispatchedBindgroup::BindgroupReference(bindgroup_reference) = &q.bindgroup
                {
                    if let Pipelines::Unary(dtype, candle_wgpu_kernels::unary::Functions::UnaryFromBufferContiguous) = &q.pipeline.0{
                        if q.pipeline.2.input1_inplaceable{
                            if let BindgroupReferenceInput::Bindgroup1(v1_id, _) = bindgroup_reference.get_input()
                            {
                                if optmize_inplace(bindgroup_reference.get_dest(), v1_id)
                                {
                                    //println!("optime unary inplace: reuse {:?} instead of copy to {:?}", v1_id, bindgroup_reference.get_dest());
                                    dev.unary_inplace_counter.inc();            
                                    q.pipeline.0 = Pipelines::Unary(dtype.clone(), candle_wgpu_kernels::unary::Functions::UnaryInplaceContiguous);
                                    q.bindgroup = DispatchedBindgroup::BindgroupReference(
                                            BindGroupReference::new(v1_id.clone(), BindgroupInputBase::Bindgroup0)
                                        );
                                }
                            } 
                        }
                    }
                    else if let Pipelines::Binary(dtype, candle_wgpu_kernels::binary::Functions::BinaryBufferFromBufferContiguousBoth) = &q.pipeline.0{
                        if let BindgroupReferenceInput::Bindgroup2(v1_id, v2_id,_) = bindgroup_reference.get_input()
                        {
                            if !cache.buffer_reference.get(v1_id).expect("buffer_reference v1 not found").cached_buffer_id().is_valid(){
                                panic!("input buffer v1 {:?}({:?}) has not input cache storage set {command_index}", cache.buffer_reference.get(v1_id).unwrap(), v1_id);
                            }
                            if !cache.buffer_reference.get(v2_id).expect("buffer_reference v2 not found").cached_buffer_id().is_valid(){
                                panic!("input buffer v2 {:?}({:?}) has not input cache storage set {command_index}", cache.buffer_reference.get(v2_id).unwrap(), v2_id);
                            }

                            if q.pipeline.2.input1_inplaceable{ 
                                if optmize_inplace(bindgroup_reference.get_dest(), v1_id) {
                                    
                                    //println!("optime binary inplace: reuse {:?} instead of copy to {:?}, additional: {:?}", v1_id, bindgroup_reference.get_dest(), v2_id);
                                    
                                    dev.binary_inplace_counter.inc();

                                    q.pipeline.0 = Pipelines::Binary(dtype.clone(), candle_wgpu_kernels::binary::Functions::BinaryBufferInplace1ContiguousBoth);
                                    q.bindgroup =
                                        DispatchedBindgroup::BindgroupReference(
                                            BindGroupReference::new(v1_id.clone(), BindgroupInputBase::Bindgroup1(v2_id.clone(), false)));
                                }
                            }
                            else if q.pipeline.2.input2_inplaceable{
                                if optmize_inplace(bindgroup_reference.get_dest(), v2_id) {
                                    //println!("optime binary inplace2: reuse {:?} instead of copy to {:?}, additional: {:?}", v2_id, bindgroup_reference.get_dest(), v1_id);
                                    
                                    dev.binary_inplace_counter.inc();
                                    q.pipeline.0 = Pipelines::Binary(dtype.clone(), candle_wgpu_kernels::binary::Functions::BinaryBufferInplace1ContiguousBoth);
                                    q.bindgroup =
                                        DispatchedBindgroup::BindgroupReference(
                                            BindGroupReference::new(v2_id.clone(), BindgroupInputBase::Bindgroup1(v1_id.clone(), false)));
                                }
                            }
                        }
                    }
                    else if let Pipelines::Copy(_, candle_wgpu_kernels::copy::Functions::Copy) = &q.pipeline.0{
                        if q.pipeline.2.input1_inplaceable{
                            if let BindgroupReferenceInput::Bindgroup1(v1_id, _) = bindgroup_reference.get_input()
                            {
                                let v1 = cache.buffer_reference.get(v1_id);
                                if let Some(v1) = v1{
                                    let vdest_id = bindgroup_reference.get_dest();
                                    
                                    let v1_cached_id = v1.cached_buffer_id().clone();
                                    let v1_size = v1.size();
                                    //this buffer was last used in this pipeline 
                                    if v1.last_used() == command_index{
                                        let vdest = cache.buffer_reference.get_mut(vdest_id);
                                        if let Some(vdest) = vdest{
                                            if vdest.size() <= v1_size {
                                                if !vdest.cached_buffer_id().is_valid() {
                                                    vdest.set_cached_buffer_id(v1_cached_id);
                  
                                                    dev.copy_inplace_counter.inc();
                                                    optimize_copy_inplace = true;
                                                }
                                            }
                                        }

                                        if optimize_copy_inplace{
                                            let v1 = cache.buffer_reference.get_mut(v1_id);
                                            if let Some(v1) = v1{
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



                fn check_for_removal(bindgroup_reference : &BindgroupReferenceFull, command_index: u32, cache : &mut ModelCache){  
                    let chec_buffer = |buffer_reference : &BufferReferenceId,cache : &mut ModelCache,command_index: u32|
                        if let Some(buffer) = cache.buffer_reference.get_mut(&buffer_reference){
                            if buffer.last_used() <= command_index{ //this buffer reference is not used after this:
                               
                                //println!("bindgroup {:?} used buffer {:?} at {command_index}(buffer last used = {})", bindgroup_reference,buffer_reference, buffer.last_used());

                                let cached_buffer_id = buffer.cached_buffer_id().clone();
                                cache.buffer_reference.delete(buffer_reference);
                                if cached_buffer_id.is_valid(){
                                    cache.buffers.free_buffer(&cached_buffer_id);
                                }
                            }
                        };
                    

                    let dest = bindgroup_reference.get_dest();
                    let input = bindgroup_reference.get_input();

                    if let Some(_) = cache.buffer_reference.get(dest){

                    }
                    else{
                        panic!("dest was not set!");
                    }

                    chec_buffer(dest, cache, command_index);
                    match input {
                        BindgroupInputBase::Bindgroup0 => {},
                        BindgroupInputBase::Bindgroup1(v1, _) => {chec_buffer(v1, cache, command_index);},
                        BindgroupInputBase::Bindgroup2(v1, v2, _) =>  {chec_buffer(v1, cache, command_index);chec_buffer(v2, cache, command_index);},
                        BindgroupInputBase::Bindgroup3(v1, v2, v3) =>  {chec_buffer(v1, cache, command_index);chec_buffer(v2, cache, command_index);chec_buffer(v3, cache, command_index);},
                    }
                        
                }
                


                if !optimize_copy_inplace {
                    let pl: &wgpu::PipelineLayout = match &q.bindgroup {
                        DispatchedBindgroup::BindgroupReference(bindgroup_reference) => {
                            match bindgroup_reference.get_input() {
                                BindgroupReferenceInput::Bindgroup0 => {
                                    &dev.bindgroup_layouts.pipeline_layout0
                                }
                                BindgroupReferenceInput::Bindgroup1( _,false) => {
                                    &dev.bindgroup_layouts.pipeline_layout1
                                }
                                BindgroupReferenceInput::Bindgroup1( _, true) => {
                                    &dev.bindgroup_layouts.pipeline_layout1_16
                                }
                                BindgroupReferenceInput::Bindgroup2( _, _, false) => {
                                    &dev.bindgroup_layouts.pipeline_layout2
                                }
                                BindgroupReferenceInput::Bindgroup2( _, _, true) => {
                                    &dev.bindgroup_layouts.pipeline_layout2_16
                                }
                                BindgroupReferenceInput::Bindgroup3( _, _, _) => {
                                    &dev.bindgroup_layouts.pipeline_layout3
                                }
                            }
                        }
                        _ => panic!("not expected"),
                    };



                    let consts = &command_buffer.id_to_const_array[q.pipeline.1];
                    let pipeline = dev.get_pipeline( &q.pipeline, pl, consts)?;
    
                    if let DispatchedBindgroup::BindgroupReference(bindgroup_reference) =
                        &q.bindgroup
                    {
                        let bindgroup = cache.get_bind_group(
                            dev,
                            bindgroup_reference,
                            q.pipeline.clone(),
                            command_index
                        );
    
                        
                         //needs to be deleayed, we want to set v1_storage to None, but to create a BindGroup, we need to have v1_storage set
                         if optimize_inplace{
                            let v1_cached_buffer_id;
                            if let Some(v1_ref) = cache.buffer_reference.get_mut(&v1_ref_id) {
                                v1_cached_buffer_id = v1_ref.cached_buffer_id().clone();
                                v1_ref.set_cached_buffer_id(CachedBufferId::new(0, 0));
                            }
                            else{
                                panic!("buffer reference not found: {:?}, cd={command_index}", v1_ref_id);
                            }
                            if let Some(vdest_ref) = cache.buffer_reference.get_mut(&vdest_ref_id) {
                                //println!("Optimize: set cache {:?} of {:?} to {:?}({:?}), cd={command_index}", v1_cached_buffer_id, v1_ref_id, vdest_ref, vdest_ref_id);
                                    
                                vdest_ref.set_cached_buffer_id(v1_cached_buffer_id);
                            }
                            else{
                                panic!("buffer reference not found: {:?}, cd={command_index}", vdest_ref_id);
                            }
                        }

                        check_for_removal(bindgroup_reference, command_index, &mut cache);
                        if cache.should_delete_unused(){ //we hit the max cache size
                            cache_limit = true;
                        }
            
                        //this may drop a bufferReference. The BufferReference needs to access cache, therefore cache was droped
                        q.bindgroup = DispatchedBindgroup::CachedBindgroup(bindgroup);
                        q.pipeline_cached = Some(pipeline);
                    }  
                }
                else{
                    if let DispatchedBindgroup::BindgroupReference(bindgroup_reference) = &q.bindgroup
                    {
                        check_for_removal(bindgroup_reference, command_index, &mut cache);
                    }
                    q.bindgroup = DispatchedBindgroup::None;
                }

                *last_meta = q.meta as usize;

               
                let meta_size = (*last_meta - current_meta) * 4 + 256 * 3;
                if meta_size > dev.configuration.meta_buffer_size as usize
                {
                    break;
                }
                if cache_limit{
                    break;
                }
                if total_workload > dev.configuration.max_workload_size{
                    break;
                }
            }
        }
    }
    let meta_size = (*last_meta - current_meta) * 4 + 256 * 3;
    let ele_size =  *index-start_index;
    log::trace!("queue {ele_size}, Meta: {meta_size}, workload: {total_workload}, cache_limit: {cache_limit}");

    return Ok(cache_limit);
}

fn finish_commands(command_buffer: &mut QueueBuffer, index : usize){
    let global_index = command_buffer.global_command_index();
    command_buffer.set_global_command_index(global_index + index as u32);
}

#[instrument]
pub(crate) fn flush_gpu_command(dev: &WgpuDevice, queue_buffer: &mut QueueBuffer) -> crate::Result<()> {
    if queue_buffer.command_queue.len() > 0 {
        let mut cache = dev.cache.lock().expect("flush gpu_commadn could not lock cache");
        prepare(dev, queue_buffer, &mut cache);
        {
            let mut start_index = 0;
            let mut index = 0;
            let mut current_meta: usize = 0;
            let mut last_meta: usize = 0;

            while index < queue_buffer.command_queue.len() {
                let should_reuse_unused = set_buffers(dev, queue_buffer, &mut index, current_meta, &mut last_meta, &mut cache)?;

                let last_meta_index = (last_meta + 256 / 4).min(queue_buffer.get_meta().len());
                
                //let mut cache = dev.cache.lock().expect("flush gpu_commadn could not lock cache");

                let cb = get_command_buffer(
                    dev,
                    &queue_buffer.get_meta()[current_meta..last_meta_index],
                    &queue_buffer.command_queue[start_index..index],
                    current_meta,
                    &None,
                    &mut cache
                );
                
                if should_reuse_unused{
                    cache.remove_unused();
                }

                #[cfg(not(target_arch = "wasm32"))]
                {
                    let span1 = span!(Level::INFO, "Device Poll");
                    let _enter1 = span1.enter();
                    dev.device.poll(wgpu::Maintain::wait()).panic_on_timeout();
                }

                //set last buffer, so we can wait for it to finish in the future
                match &queue_buffer.command_queue[index - 1]{
                    MlQueue::Dispatch(d) => {
                        match &d.bindgroup{
                            DispatchedBindgroup::CachedBindgroup(c) => {
                                if let Some(c) = cache.bindgroups.get_bindgroup(c){
                                    queue_buffer.last_buffer = Some(c.buffer().get_dest().clone())
                                }
                               
                            },
                            _ => {},
                        }

                    }
                }

                let span1 = span!(Level::INFO, "Submit");
                let _enter1 = span1.enter();
                dev.queue.submit(Some(cb));
                drop(_enter1); 
               
                start_index = index;
                current_meta = last_meta;
            }
            finish_commands(queue_buffer, index);
        }
        queue_buffer.clear();
        {
            //let mut cache = dev.cache.lock().expect("flush gpu_commadn could not lock cache");

            log::warn!("current memory {} / {}", cache.buffers.buffer_memory(), cache.buffers.max_memory_allowed());
            cache.mappings.finish();
            //cache.buffers.remove_unused();
            cache.remove_unused();
        }
    }
    Ok(())
}

#[instrument]
pub(crate) async fn flush_gpu_command_async(dev: &WgpuDevice, queue_buffer: &mut QueueBuffer) -> crate::Result<()> {
    if queue_buffer.command_queue.len() > 0 {
        log::warn!("flush_gpu_command_async");
        let mut cache = dev.cache.lock().expect("flush gpu_commadn could not lock cache");
        prepare(dev, queue_buffer, &mut cache);
        {
            let mut start_index = 0;
            let mut index = 0;
            let mut current_meta: usize = 0;
            let mut last_meta: usize = 0;

            while index < queue_buffer.command_queue.len() {
                let should_reuse_unused = set_buffers(dev, queue_buffer, &mut index, current_meta, &mut last_meta, &mut cache)?;

                let last_meta_index = (last_meta + 256 / 4).min(queue_buffer.get_meta().len());
              
                //let mut cache = dev.cache.lock().expect("flush gpu_commadn could not lock cache");
                let cb = get_command_buffer(
                    dev,
                    &queue_buffer.get_meta()[current_meta..last_meta_index],
                    &queue_buffer.command_queue[start_index..index],
                    current_meta,
                    &queue_buffer.last_buffer,
                    &mut cache
                );
              
                if should_reuse_unused{
                    cache.remove_unused();
                }

                // let span1 = span!(Level::INFO, "Device Poll");
                // let _enter1 = span1.enter();
                //dev.device.poll(wgpu::Maintain::wait()).panic_on_timeout();
                if !dev.device.poll(wgpu::Maintain::Poll).is_queue_empty(){
                    synchronize_device(&dev, &dev.queue).await?;
                }
                
                // if start_index > 0{
                //     //get buffer of prev group
                //     match(queue_buffer.command_queue[start_index-1]){
                //         MlQueue::Dispatch(d) => {
                //             match(d.bindgroup){
                //                 DispatchedBindgroup::CachedBindgroup(c) => {
                                    
                //                     read_data_from_gpu_async_buffer(dev, &c.buffers.get_dest().buffer);

                //                 },
                //                 _ => {},
                //             }
    
                //         }
                //     }
                // }

                let span1 = span!(Level::INFO, "Submit");
                let _enter1 = span1.enter();
                dev.queue.submit(Some(cb));
                drop(_enter1); 
               
                start_index = index;
                current_meta = last_meta;
            }
            finish_commands(queue_buffer, index);
        }

      
       
       
        queue_buffer.clear();
        {
            //let mut cache = dev.cache.lock().expect("flush gpu_commadn could not lock cache");
            log::warn!("current memory {} / {}", cache.buffers.buffer_memory(), cache.buffers.max_memory_allowed());
        
            cache.mappings.finish();
            //cache.buffers.remove_unused();
            cache.remove_unused();
        }
    }
    Ok(())
}



fn enqueue(
    command_queue: MutexGuard<QueueBuffer>,
    pipeline: PipelineType,
    bind_group: BindGroupReference,
    length: u32,
    workload_size : usize
) {
    return enqueue_extra(
        command_queue,
        pipeline,
        bind_group,
        length,
        workload_size,
        #[cfg(feature = "wgpu_debug")]
        None,
    );
}

fn enqueue_extra(
    command_queue: MutexGuard<QueueBuffer>,
    pipeline: PipelineType,
    bind_group: BindGroupReference,
    length: u32,
    workload_size : usize,
    #[cfg(feature = "wgpu_debug")] _debug: Option<String>,
) {
    return enqueue_workgroups_extra(
        command_queue,
        pipeline,
        bind_group,
        (length + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE,
        1,
        1,
        workload_size,
        #[cfg(feature = "wgpu_debug")]
        _debug,
    );
}

fn enqueue_big(
    command_queue: MutexGuard<QueueBuffer>,
    pipeline: PipelineType,
    bind_group: BindGroupReference,
    length: u32
) {
    return enqueue_big_extra(
        command_queue,
        pipeline,
        bind_group,
        length,
        #[cfg(feature = "wgpu_debug")]
        None,
    );
}

fn enqueue_big_extra(
    command_queue: MutexGuard<QueueBuffer>,
    pipeline: PipelineType,
    bind_group: BindGroupReference,
    length: u32,
    #[cfg(feature = "wgpu_debug")] _debug: Option<String>,
) {

    let id = (length + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    let x = id.min(65535);
    let y = (id + 65534) / 65535;

    return enqueue_workgroups_extra(
        command_queue,
        pipeline,
        bind_group,
        x,
        y,
        1,
        length as usize,
        #[cfg(feature = "wgpu_debug")]
        _debug,
    );
}

#[instrument]
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

#[instrument]
pub fn create_bindgroup(dev: &WgpuDevice, bindgroup: CachedBindgroupFull, cache : &ModelCache) -> wgpu::BindGroup {
    let buffer_meta = &dev.meta_buffer;

    let meta_binding = wgpu::BufferBinding {
        buffer: &buffer_meta,
        offset: 0,
        size: Some(NonZeroU64::new(256).unwrap()),
    };
    let meta_binding = wgpu::BindingResource::Buffer(meta_binding);

    let meta_entry = wgpu::BindGroupEntry {
        binding: 1,
        resource: meta_binding,
    };

    let bind_group_layout = match bindgroup.get_input() {
        CachedBindgroupInput::Bindgroup0 => &dev.bindgroup_layouts.bind_group_layout0,
        CachedBindgroupInput::Bindgroup1(_, false) => &dev.bindgroup_layouts.bind_group_layout1,
        CachedBindgroupInput::Bindgroup1(_, true) => &dev.bindgroup_layouts.bind_group_layout1_16,
        CachedBindgroupInput::Bindgroup2(_, _, false) => &dev.bindgroup_layouts.bind_group_layout2,
        CachedBindgroupInput::Bindgroup2(_, _, true) => &dev.bindgroup_layouts.bind_group_layout2_16,
        CachedBindgroupInput::Bindgroup3(_, _, _) => &dev.bindgroup_layouts.bind_group_layout3,
    };

    let buffer_dest = bindgroup.get_dest();

    let buffer_resource = cache.buffers.get_buffer(buffer_dest).expect("buffer_dest could not be found").buffer().as_entire_binding();

    match bindgroup.get_input() {
        CachedBindgroupInput::Bindgroup0 => {
            let entries = &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_resource,
                },
                meta_entry,
            ];
            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: entries,
            })
        }
        CachedBindgroupInput::Bindgroup1(buffer_input1, _) => {

            if cache.buffers.get_buffer(buffer_input1).is_none(){
                panic!("buffer_input_1 : {:?} could not be found(in {:?})", buffer_input1, bindgroup);
            }


            let entries = &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_resource,
                },
                meta_entry,
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cache.buffers.get_buffer(buffer_input1).expect("buffer_input1 could not be found").buffer().as_entire_binding(),
                },
            ];
            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: entries,
            })
        }
        CachedBindgroupInput::Bindgroup2(buffer_input1, buffer_input2, _) => {
            let entries = &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_resource,
                },
                meta_entry,
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cache.buffers.get_buffer(buffer_input1).expect("buffer_input1 could not be found").buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cache.buffers.get_buffer(buffer_input2).expect("buffer_input2 could not be found").buffer().as_entire_binding(),
                },
            ];
            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: entries,
            })
        }
        CachedBindgroupInput::Bindgroup3(
            buffer_input1,
            buffer_input2,
            buffer_input3,
        ) => {
            let entries = &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_resource,
                },
                meta_entry,
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cache.buffers.get_buffer(buffer_input1).expect("buffer_input1 could not be found").buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cache.buffers.get_buffer(buffer_input2).expect("buffer_input2 could not be found").buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: cache.buffers.get_buffer(buffer_input3).expect("buffer_input3 could not be found").buffer().as_entire_binding(),
                },
            ];
            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: entries,
            })
        }
    }
}

fn create_bind_group_input0(buffer_dest: BufferReferenceId) -> BindGroupReference {
    BindGroupReference::new(buffer_dest, BindgroupInputBase::Bindgroup0)
}

fn create_bind_group_input1(
    buffer_dest: BufferReferenceId,
    buffer_input1: BufferReferenceId,
) -> BindGroupReference {
    BindGroupReference::new(buffer_dest, BindgroupInputBase::Bindgroup1(buffer_input1, false))
}

fn create_bind_group_input2(
    buffer_dest: BufferReferenceId,
    buffer_input1: BufferReferenceId,
    buffer_input2: BufferReferenceId,
) -> BindGroupReference {
    BindGroupReference::new(buffer_dest, BindgroupInputBase::Bindgroup2(buffer_input1,buffer_input2, false))
}

fn create_bind_group_input2_16(
    buffer_dest: BufferReferenceId,
    buffer_input1: BufferReferenceId,
    buffer_input2: BufferReferenceId,
) -> BindGroupReference {
    BindGroupReference::new(buffer_dest, BindgroupInputBase::Bindgroup2(buffer_input1,buffer_input2, true))
}

fn create_bind_group_input3(
    buffer_dest: BufferReferenceId,
    buffer_input1: BufferReferenceId,
    buffer_input2: BufferReferenceId,
    buffer_input3: BufferReferenceId,
) -> BindGroupReference {
    BindGroupReference::new(buffer_dest, BindgroupInputBase::Bindgroup3(buffer_input1,buffer_input2, buffer_input3))
}

#[instrument]
pub fn synchronize(dev: &WgpuDevice) -> crate::Result<()> {
    let mut command_queue = dev.command_queue.lock().unwrap();
    if command_queue.command_queue.len() > 0{
        flush_gpu_command(dev, &mut command_queue)?;
        if let Some(buffer) = &command_queue.last_buffer{
            let cache = dev.cache.lock().unwrap();
            if let Some(buffer) =  cache.buffers.get_buffer(buffer){
                copy_to_staging_probe(dev, &buffer.buffer());
            }
        }
       
        return pollster::block_on(synchronize_device(&dev, &dev.queue));
    }
    Ok(())
}

#[instrument]
pub async fn synchronize_async(dev: &WgpuDevice) -> crate::Result<()> {
    let mut command_queue = dev.command_queue.lock().unwrap();
    if command_queue.command_queue.len() > 0{
        flush_gpu_command_async(dev, &mut command_queue).await?;
        if let Some(buffer) = &command_queue.last_buffer{
            let cache = dev.cache.lock().unwrap();
            if let Some(buffer) =  cache.buffers.get_buffer(buffer){
                copy_to_staging_probe(dev, &buffer.buffer());
            }
        }
        return synchronize_device(&dev, &dev.queue).await;
    }
    Ok(())
}


// #[instrument]
// async fn synchronize_device(dev: &Device, queue: &Queue) -> crate::Result<()> {
//     let (sender, receiver) = flume::bounded(1);
//     queue.on_submitted_work_done(move || sender.send(()).unwrap());

//     dev.poll(wgpu::Maintain::wait()).panic_on_timeout();
//     if let Ok(()) = receiver.recv_async().await {
//         return Ok(());
//     }
//     Ok(())
// }


#[instrument]
async fn synchronize_device(dev: &WgpuDevice, queue: &Queue) -> crate::Result<()> {
    wait_for_gpu_buffer_async(dev).await
}


#[instrument]
pub async fn read_data_from_gpu_async<T: bytemuck::Pod>(
    dev: &WgpuDevice,
    buffer: BufferReferenceId,
) -> crate::Result<Vec<T>> {
    let mut command_queue = dev.command_queue.lock().unwrap();
    flush_gpu_command_async(dev, &mut command_queue).await?; //send all previous commands to the gpu
    
    let cache = dev.cache.lock().unwrap();
    if let Some(buffer) =  cache.buffer_reference.get(&buffer){
        let buffer_storage = buffer.cached_buffer_id();
        if buffer_storage.is_valid(){
            if let Some(buffer) = cache.buffers.get_buffer(buffer_storage){
                Ok(read_data_from_gpu_async_buffer(dev, &buffer.buffer()).await)
            }else {
                panic!("Unespected error at read_data from gpu. Tensor WgpuStorage did not Point to a wgpu Buffer")
            }
        } else {
            panic!("Unespected error at read_data from gpu. Tensor WgpuStorage did not Point to a wgpu Buffer")
        }
    }
    else{
        panic!("Unespected error at read_data from gpu. Tensor WgpuStorage did not Point to a wgpu Buffer Reference")
    }
}


pub fn copy_to_staging_probe(dev: &WgpuDevice, buffer: &wgpu::Buffer){
    let mut encoder = dev
    .device
    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    let staging_buffer = &dev.staging_probe_buffer;

    encoder.copy_buffer_to_buffer(&buffer, 0, &staging_buffer, 0, 4);

    // Submits command encoder for processing
    dev.queue.submit(Some(encoder.finish()));
}

#[instrument]
//wait for the current staging buffer, 
//the buffer one wants to 
pub async fn wait_for_gpu_buffer_async(
    dev: &WgpuDevice,
) -> crate::Result<()> {
    let staging_buffer = &dev.staging_probe_buffer;

    // Note that we're not calling `.await` here.
    let buffer_slice = staging_buffer.slice(..);
    // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    dev.device.poll(wgpu::Maintain::wait()).panic_on_timeout();

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



#[instrument]
pub async fn read_data_from_gpu_async_buffer<T: bytemuck::Pod>(
    dev: &WgpuDevice,
    buffer: &wgpu::Buffer,
) -> Vec<T> {
    let dest_size = buffer.size();

    //TODO: use cached staging buffer!
    let staging_buffer = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: dest_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = dev
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    encoder.copy_buffer_to_buffer(&buffer, 0, &staging_buffer, 0, dest_size);

    // Submits command encoder for processing
    dev.queue.submit(Some(encoder.finish()));

    // Note that we're not calling `.await` here.
    let buffer_slice = staging_buffer.slice(..);
    // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).expect("error in read_data could not send flume"));

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    dev.device.poll(wgpu::Maintain::wait()).panic_on_timeout();

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
        result
    } else {
        panic!("failed to run compute on gpu!")
    }
}
