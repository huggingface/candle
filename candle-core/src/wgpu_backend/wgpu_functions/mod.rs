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

use std::io::Write;

use super::{
    cache::{BindGroupReferenceBase, BufferReference, CachedBindGroupReference},
    device::{BindGroupReference, MlQueue, PipelineType, QueueBuffer, META_BUFFER_SIZE},
};
use crate::DType;
use crate::{wgpu_backend::device::WgpuDevice, Error, Layout, WebGpuError};
use std::{
    borrow::Cow,
    sync::{Arc, MutexGuard},
};
use wgpu::ShaderModule;

pub use binary::queue_binary_buffer_from_buffer;
pub use cmp::queue_cmp_buffer_from_buffer;
pub use conv2d::{queue_conv1d, queue_conv1d_transpose, queue_conv2d, queue_conv2d_transpose};
pub use convert::{queue_convert_f32_to_u32, queue_convert_u32_to_f32, queue_convert_u8_to_f32};
pub use copy::{queue_copy, queue_copy2d, queue_copy_strided};
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

#[derive(Debug)]
pub(crate) struct MetaArray(Vec<u32>);

pub(crate) trait MetaArrayToU32 {
    fn to_u32(self) -> u32;
}

impl MetaArray {
    pub(crate) fn new(capacity: u32) -> Self {
        MetaArray(Vec::with_capacity(capacity as usize))
    }

    pub(crate) fn add_layout(&mut self, layout: &Layout) {
        let shape = layout.shape().dims();
        let stride = layout.stride();
        self.0.push(shape.len() as u32);
        self.0.push(layout.start_offset() as u32);

        if layout.is_contiguous() {
            self.0.push(layout.shape().elem_count() as u32);
        } else {
            self.0.push(0);
        }

        self.0.extend(shape.iter().map(|&x| x as u32));
        self.0.extend(stride.iter().map(|&x| x as u32));
    }

    pub(crate) fn add<T: MetaArrayToU32>(&mut self, value: T) {
        self.0.push(value.to_u32());
    }
}

fn get_size(layout: &Layout) -> u32 {
    return 3 + layout.dims().len() as u32 * 2;
}

impl MetaArrayToU32 for u32 {
    fn to_u32(self) -> u32 {
        return self;
    }
}

impl MetaArrayToU32 for f32 {
    fn to_u32(self) -> u32 {
        return f32::to_bits(self);
    }
}

impl MetaArrayToU32 for usize {
    fn to_u32(self) -> u32 {
        return self as u32;
    }
}

#[derive(Debug, Hash, std::cmp::Eq, std::cmp::PartialEq, Clone)]
pub enum Shader {
    Binary(DType),
    Cmp(DType),
    Conv2D(DType),
    Convert(DType),
    Copy(DType),
    IndexSelect(DType),
    Matmul(DType),
    Reduce(DType),
    RmsNorm(DType),
    Softmax(DType),
    Unary(DType),
    WhereCond(DType),
    Pool2d(DType),
    Upsample(DType),
    Gather(DType),
}

pub fn load_shader(shader: Shader) -> crate::Result<&'static str> {
    match shader {
        Shader::Binary(DType::F32) => Ok(include_str!(
            "binary/generated/shader.pwgsl_generated_f32.wgsl"
        )),
        Shader::Binary(DType::U32) => Ok(include_str!(
            "binary/generated/shader.pwgsl_generated_u32.wgsl"
        )),
        Shader::Cmp(DType::F32) => Ok(include_str!(
            "cmp/generated/shader.pwgsl_generated_f32.wgsl"
        )),
        Shader::Cmp(DType::U32) => Ok(include_str!(
            "cmp/generated/shader.pwgsl_generated_u32.wgsl"
        )),
        Shader::Conv2D(DType::F32) => Ok(include_str!(
            "conv2d/generated/shader.pwgsl_generated_f32.wgsl"
        )),
        Shader::Conv2D(DType::U32) => Ok(include_str!(
            "conv2d/generated/shader.pwgsl_generated_u32.wgsl"
        )),
        Shader::Convert(DType::F32) => Ok(include_str!(
            "convert/generated/shader.pwgsl_generated_f32.wgsl"
        )),
        Shader::Convert(DType::U32) => Ok(include_str!(
            "convert/generated/shader.pwgsl_generated_u32.wgsl"
        )),
        Shader::Convert(DType::U8) => Ok(include_str!(
            "convert/generated/shader.pwgsl_generated_u8.wgsl"
        )),
        Shader::Copy(DType::F32) => Ok(include_str!(
            "copy/generated/shader.pwgsl_generated_f32.wgsl"
        )),
        Shader::Copy(DType::U32) => Ok(include_str!(
            "copy/generated/shader.pwgsl_generated_u32.wgsl"
        )),
        Shader::IndexSelect(DType::F32) => Ok(include_str!(
            "index_select/generated/shader.pwgsl_generated_f32.wgsl"
        )),
        Shader::IndexSelect(DType::U32) => Ok(include_str!(
            "index_select/generated/shader.pwgsl_generated_u32.wgsl"
        )),
        Shader::Matmul(DType::F32) => Ok(include_str!(
            "matmul/generated/shader.pwgsl_generated_f32.wgsl"
        )),
        Shader::Matmul(DType::U32) => Ok(include_str!(
            "matmul/generated/shader.pwgsl_generated_u32.wgsl"
        )),
        Shader::Reduce(DType::F32) => Ok(include_str!(
            "reduce/generated/shader.pwgsl_generated_f32.wgsl"
        )),
        Shader::Reduce(DType::U32) => Ok(include_str!(
            "reduce/generated/shader.pwgsl_generated_u32.wgsl"
        )),
        Shader::RmsNorm(DType::F32) => Ok(include_str!(
            "rms_norm/generated/shader.pwgsl_generated_f32.wgsl"
        )),
        Shader::RmsNorm(DType::U32) => Ok(include_str!(
            "rms_norm/generated/shader.pwgsl_generated_u32.wgsl"
        )),
        Shader::Softmax(DType::F32) => Ok(include_str!(
            "softmax/generated/shader.pwgsl_generated_f32.wgsl"
        )),
        Shader::Softmax(DType::U32) => Ok(include_str!(
            "softmax/generated/shader.pwgsl_generated_u32.wgsl"
        )),
        Shader::Unary(DType::F32) => Ok(include_str!(
            "unary/generated/shader.pwgsl_generated_f32.wgsl"
        )),
        Shader::Unary(DType::U32) => Ok(include_str!(
            "unary/generated/shader.pwgsl_generated_u32.wgsl"
        )),
        Shader::WhereCond(DType::F32) => Ok(include_str!(
            "where_cond/generated/shader.pwgsl_generated_f32.wgsl"
        )),
        Shader::WhereCond(DType::U32) => Ok(include_str!(
            "where_cond/generated/shader.pwgsl_generated_u32.wgsl"
        )),
        Shader::Pool2d(DType::F32) => Ok(include_str!(
            "pool2d/generated/shader.pwgsl_generated_f32.wgsl"
        )),
        Shader::Pool2d(DType::U32) => Ok(include_str!(
            "pool2d/generated/shader.pwgsl_generated_u32.wgsl"
        )),
        Shader::Upsample(DType::F32) => Ok(include_str!(
            "upsample/generated/shader.pwgsl_generated_f32.wgsl"
        )),
        Shader::Upsample(DType::U32) => Ok(include_str!(
            "upsample/generated/shader.pwgsl_generated_u32.wgsl"
        )),
        Shader::Gather(DType::F32) => Ok(include_str!(
            "gather/generated/shader.pwgsl_generated_f32.wgsl"
        )),
        Shader::Gather(DType::U32) => Ok(include_str!(
            "gather/generated/shader.pwgsl_generated_u32.wgsl"
        )),

        _ => Err(crate::Error::WebGpu(WebGpuError::Message(format!(
            "Could not find Pipeline: {:?}",
            shader
        )))),
    }
}

const WORKGROUP_SIZE: u32 = 64;

pub fn get_shader(device: &wgpu::Device, shader: &'static str) -> ShaderModule {
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader)),
    });
    return cs_module;
}

/// Size is in Bytes!
pub fn create_buffer(dev: &WgpuDevice, size: usize) -> Arc<BufferReference> {
    //let mut cache = dev.cache.lock().unwrap();

    return BufferReference::new(dev, size as u64);

    // let create_new = || { Arc::new(dev.device.create_buffer(&wgpu::BufferDescriptor {
    //     label: None,
    //     size: size as u64,
    //     usage: wgpu::BufferUsages::STORAGE
    //         | wgpu::BufferUsages::COPY_SRC
    //         | wgpu::BufferUsages::COPY_DST,
    //     mapped_at_creation: false,
    // }))};

    // match cache.as_mut() {
    //     Some(cache) => {
    //         let buffer_index = cache.counter_buffer;
    //         let buffer;
    //         if cache.cached_buffer.len() > buffer_index as usize {
    //             buffer = cache.cached_buffer[buffer_index as usize].clone();

    //             if buffer.size() as usize != size{
    //                 panic!("Error using Cache, Size mismatch: cached: {}, creating: {}", buffer.size(), size)
    //             }
    //         }
    //         else{
    //             buffer = create_new();
    //             cache.cached_buffer.push(buffer.clone());
    //             create_new();
    //         };
    //         cache.counter_buffer += 1;
    //         return buffer;
    //     },
    //     None => create_new()
    // }
}

pub fn create_buffer_init<T: bytemuck::Pod>(dev: &WgpuDevice, data: &[T]) -> Arc<BufferReference> {
    return BufferReference::new_init(dev, bytemuck::cast_slice(data));
}

fn enqueue_workgroups(
    mut command_queue: MutexGuard<QueueBuffer>,
    pipeline: PipelineType,
    bind_group: BindGroupReference,
    x: u32,
    y: u32,
    z: u32,
    #[cfg(feature = "wgpu_debug")] _debug: super::device::QueueDebugInfo,
) {
    if x > 65535 || y > 65535 || z > 65535 {
        enqueue_workgroups_indirect(
            command_queue,
            pipeline,
            bind_group,
            x,
            y,
            z,
            #[cfg(feature = "wgpu_debug")]
            _debug,
        );
    } else {
        let q = MlQueue::Dispatch(super::device::MlQueueDispatch {
            x,
            y,
            z,
            pipeline: pipeline,
            bindgroup: bind_group,
            indirect_buffer: None,
            #[cfg(feature = "wgpu_debug")]
            debug: _debug,
        });
        command_queue.command_queue.push(q);
    }
}

fn enqueue_workgroups_indirect(
    mut command_queue: MutexGuard<QueueBuffer>,
    pipeline: PipelineType,
    bind_group: BindGroupReference,
    x: u32,
    y: u32,
    z: u32,
    #[cfg(feature = "wgpu_debug")] _debug: super::device::QueueDebugInfo,
) {
    let data = DispatchIndirectArgs { x, y, z };

    let indirect_array = &mut command_queue.indirect_array;
    let indirect_offset = indirect_array.len();
    indirect_array.push(data);

    let q = MlQueue::Dispatch(super::device::MlQueueDispatch {
        x,
        y,
        z,
        pipeline: pipeline,
        bindgroup: bind_group,
        indirect_buffer: Some(indirect_offset),
        #[cfg(feature = "wgpu_debug")]
        debug: _debug,
    });
    command_queue.command_queue.push(q);
}

fn next_divisible_by_n(value: i32, n: i32) -> i32 {
    if n == 0 {
        panic!("n must be a non-zero integer");
    }

    if value % n == 0 {
        value
    } else {
        value + (n - value % n)
    }
}

//size: size you want to add
fn get_meta(dev: &WgpuDevice, size: u32) -> (MutexGuard<QueueBuffer>, u32) {
    //println!("get_meta dev.command_queue lock_start");
    let mut command_queue = dev.command_queue.lock().unwrap();
    //println!("get_meta dev.command_queue lock_end");
    let meta_array_length = command_queue.meta_array.0.len() as i32;
    let meta_offset = next_divisible_by_n(
        meta_array_length,
        dev.device_limits.min_storage_buffer_offset_alignment as i32 / 4,
    );

    if meta_offset as u32 + size > META_BUFFER_SIZE / 4 {
        flush_gpu_command(dev, &mut command_queue);
        return (command_queue, 0);
    }

    command_queue
        .meta_array
        .0
        .extend(std::iter::repeat(0).take((meta_offset - meta_array_length) as usize));

    return (command_queue, meta_offset as u32);
}

/// Argument buffer layout for dispatch_indirect commands.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DispatchIndirectArgs {
    /// The number of work groups in X dimension.
    pub x: u32,
    /// The number of work groups in Y dimension.
    pub y: u32,
    /// The number of work groups in Z dimension.
    pub z: u32,
}

#[cfg(feature = "wgpu_debug")]
fn init_debug_queue(dev: &WgpuDevice, length: u32) -> (u32, wgpu::QuerySet) {
    let global_index = dev.debug.counter.load(std::sync::atomic::Ordering::Relaxed);
    let query_set = dev.device.create_query_set(&wgpu::QuerySetDescriptor {
        count: length as u32 * 2, // We need 2 queries: one for start and one for end
        ty: wgpu::QueryType::Timestamp,
        label: None,
    });
    return (global_index, query_set);
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

pub(crate) fn flush_gpu_command(dev: &WgpuDevice, queue_buffer: &mut QueueBuffer) {
    if queue_buffer.command_queue.len() > 0 {
        // let mut cache = dev.cache.lock().unwrap();
        // let cache = cache.as_mut();
        // match cache{
        //     Some(cache) => {
        //         let mut queue = cache.queue.lock().unwrap();

        //         let mut new_buffer = QueueBuffer::new();

        //         new_buffer.command_queue.append(&mut queue_buffer.command_queue);
        //         new_buffer.meta_array.0.append(&mut queue_buffer.meta_array.0);
        //         new_buffer.indirect_array.append(&mut queue_buffer.indirect_array);
        //         queue.push(new_buffer);
        //         return;
        //     },
        //     None => {},
        // };
        // {
        //     let mut file = std::fs::OpenOptions::new()
        //     .write(true)
        //     .append(true)
        //     .create(true)
        //     .open("debug-llama2c.txt")
        //     .unwrap();
        //     writeln!(file, "FLUSHING COMMANDS");
        // }
       
        #[cfg(feature = "wgpu_debug")]
        let (global_index, query_set) = init_debug_queue(dev, queue.len() as u32 * 2);

        #[cfg(feature = "wgpu_debug")]
        let mut debug_index = 0;
    
        dev.queue.write_buffer(
            &dev.meta_buffer,
            0,
             &bytemuck::cast_slice(&queue_buffer.meta_array.0[..]),
        );
        queue_buffer.meta_array.0.clear();

        dev.queue.write_buffer(
            &dev.indirect_buffer,
            0,
            &bytemuck::cast_slice(&queue_buffer.indirect_array[..]),
        );
        queue_buffer.indirect_array.clear();
        {
            let queue = &mut queue_buffer.command_queue;

            let mut wgpu_data = Vec::with_capacity(queue.len());
            for q in queue.drain(..) {
                {
                    //println!("flush_gpu_command cache lock_start");
                    let mut cache = dev.cache.lock().unwrap();
           
                    //println!("flush_gpu_command cache lock_end");
                    match q {
                        MlQueue::Dispatch(q) => {

                            let meta = q.bindgroup.get_meta();
                            let pl: &wgpu::PipelineLayout = match q.bindgroup{
                                BindGroupReferenceBase::Bindgroup0(_,_) => &dev.bindgroup_layouts.pipeline_layout0,
                                BindGroupReferenceBase::Bindgroup1(_,_, _) => &dev.bindgroup_layouts.pipeline_layout1,
                                BindGroupReferenceBase::Bindgroup2(_,_, _, _) => &dev.bindgroup_layouts.pipeline_layout2,
                                BindGroupReferenceBase::Bindgroup3(_,_, _, _, _) => &dev.bindgroup_layouts.pipeline_layout3,
                            };

                            let pipeline = dev.get_pipeline2(q.pipeline.0.clone(), q.pipeline.1.clone(),pl).unwrap();
                            let bindgroup = cache.get_bind_group(dev, &q.bindgroup, pipeline.clone(), q.pipeline);
                            wgpu_data.push((pipeline, bindgroup, q.x, q.y, q.z, q.indirect_buffer, meta));
                            drop(cache);
                        }
                    }
                }
            }


            let mut encoder = dev
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });

                for data in wgpu_data.iter() {
                    let (pipline, bindgroup, qx, qy, qz, qindirect_buffer, meta) = data;
                     #[cfg(feature = "wgpu_debug")]
                    cpass.write_timestamp(&query_set, debug_index);

                    cpass.set_pipeline(&pipline);

                
                    cpass.set_bind_group(0, &bindgroup.bindgroup, &[meta * 4]);

                    if let Some(indirect_buffer_index) = &qindirect_buffer {
                        cpass.dispatch_workgroups_indirect(
                            &dev.indirect_buffer,
                            (indirect_buffer_index
                                * std::mem::size_of::<DispatchIndirectArgs>())
                                as u64,
                        );
                    } else {
                        cpass.dispatch_workgroups(*qx, *qy, *qz);
                    }

                    #[cfg(feature = "wgpu_debug")]
                    {
                        cpass.write_timestamp(&query_set, debug_index + 1);
                        dev.debug.insert_info(
                            global_index + debug_index * 8,
                            (
                                q.debug.name.as_ref().unwrap().to_owned(),
                                q.debug.output_size,
                                q.x,
                                q.y,
                                q.z,
                            ),
                        );
                        debug_index += 2;
                    }
                }
            }
            #[cfg(feature = "wgpu_debug")]
            end_debug_queue(
                dev,
                queue.len() as u32 * 2,
                global_index,
                &mut encoder,
                &query_set,
            );

            dev.queue.submit(Some(encoder.finish()));
        }
        queue_buffer.command_queue.clear();
    }
}

// pub (crate) fn flush_gpu_command_old(dev: &WgpuDevice, meta_array : &mut MutexGuard<MetaArray>){
//     let mut queue = dev.command_queue.command_queue.lock().unwrap();
//     if queue.len() > 0{

//         #[cfg(feature = "wgpu_debug")]
//         let (global_index, query_set) =  init_debug_queue(dev, queue.len() as u32 * 2);

//         #[cfg(feature = "wgpu_debug")]
//         let mut debug_index = 0;

//         let meta_array_slice = &meta_array.0[..];
//         dev.queue.write_buffer(&dev.meta_buffer, 0, &bytemuck::cast_slice(meta_array_slice));
//         meta_array.0.clear();

//         let indirect_array = &mut dev.command_queue.indirect_array.lock().unwrap();
//         dev.queue.write_buffer(&dev.indirect_buffer, 0, &bytemuck::cast_slice(&indirect_array[..]));
//         indirect_array.clear();

//         let mut encoder = dev.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
//         {
//             let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
//                 label: None,
//                 timestamp_writes: None,
//             });

//             for q in queue.iter(){
//                 match q{
//                     MlQueue::Dispatch(q) => {

//                         #[cfg(feature = "wgpu_debug")]
//                         cpass.write_timestamp(&query_set, debug_index);

//                         cpass.set_pipeline(&q.pipeline);
//                         cpass.set_bind_group(0, &q.bind_group, &[]);
//                         if let Some(indirect_buffer_index) = &q.indirect_buffer{

//                             cpass.dispatch_workgroups_indirect(& dev.indirect_buffer, (indirect_buffer_index * std::mem::size_of::<DispatchIndirectArgs>()) as u64);
//                         }
//                         else{
//                             cpass.dispatch_workgroups(q.x, q.y, q.z);
//                         }

//                         #[cfg(feature = "wgpu_debug")]
//                         {
//                             cpass.write_timestamp(&query_set, debug_index + 1);
//                             dev.debug.insert_info(global_index + debug_index * 8,(q.debug.name.as_ref().unwrap().to_owned(), q.debug.output_size, q.x, q.y, q.z));
//                             debug_index+= 2;
//                         }
//                     }
//                 }
//             }
//         }
//         #[cfg(feature = "wgpu_debug")]
//         end_debug_queue(dev, queue.len() as u32 * 2, global_index, &mut encoder, &query_set);

//         dev.queue.submit(Some(encoder.finish()));
//         queue.clear();
//     }
// }

fn enqueue(
    command_queue: MutexGuard<QueueBuffer>,
    pipeline: PipelineType,
    bind_group: BindGroupReference,
    length: u32,
    #[cfg(feature = "wgpu_debug")] _debug: super::device::QueueDebugInfo,
) {
    return enqueue_workgroups(
        command_queue,
        pipeline,
        bind_group,
        (length + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE,
        1,
        1,
        #[cfg(feature = "wgpu_debug")]
        _debug,
    );
}

// fn bind_group_helper<F : Fn() -> wgpu::BindGroup>( dev: &WgpuDevice, compute_pipeline : Arc<wgpu::ComputePipeline>,
//      create_new : F) -> Arc<BindGroup>{
//     let mut cache = dev.cache.lock().unwrap();
//     match cache.as_mut() {
//         Some(cache) => {
//             let bindgroup_index = cache.counter_bindgroup;
//             let bindgroup;
//             if cache.cached_bindgroup.len() > bindgroup_index as usize {
//                 bindgroup = cache.cached_bindgroup[bindgroup_index as usize].clone();
//                 let cached_pipeline =  cache.cached_pipeline[bindgroup_index as usize].clone();

//                 if cached_pipeline.global_id() != compute_pipeline.global_id(){
//                     panic!("Trying to use a bindgroup of an different pipeline! Index:{bindgroup_index}");
//                 }

//             }
//             else{
//                 bindgroup = Arc::new(create_new());
//                 cache.cached_bindgroup.push(bindgroup.clone());
//                 cache.cached_pipeline.push(compute_pipeline);
//                 create_new();
//             };
//             cache.counter_bindgroup += 1;
//             return bindgroup;
//         },
//         None => Arc::new(create_new())
//     }
// }

fn create_bind_group_input0(
    dev: &WgpuDevice,
    pipeline: PipelineType,
    meta_offset: u32,
    buffer_dest: Arc<BufferReference>,
) -> BindGroupReference {
    BindGroupReference::Bindgroup0(meta_offset, buffer_dest)
}

fn create_bind_group_input1(
    dev: &WgpuDevice,
    pipeline: PipelineType,
    meta_offset: u32,
    buffer_dest: Arc<BufferReference>,
    buffer_input1: Arc<BufferReference>,
) -> BindGroupReference {
    BindGroupReference::Bindgroup1(meta_offset, buffer_dest, buffer_input1)
}

fn create_bind_group_input2(
    dev: &WgpuDevice,
    pipeline: PipelineType,
    meta_offset: u32,
    buffer_dest: Arc<BufferReference>,
    buffer_input1: Arc<BufferReference>,
    buffer_input2: Arc<BufferReference>,
) -> BindGroupReference {
    BindGroupReference::Bindgroup2(meta_offset, buffer_dest, buffer_input1, buffer_input2)
}

fn create_bind_group_input3(
    dev: &WgpuDevice,
    pipeline: PipelineType,
    meta_offset: u32,
    buffer_dest: Arc<BufferReference>,
    buffer_input1: Arc<BufferReference>,
    buffer_input2: Arc<BufferReference>,
    buffer_input3: Arc<BufferReference>,
) -> BindGroupReference {
    BindGroupReference::Bindgroup3(
        meta_offset,
        buffer_dest,
        buffer_input1,
        buffer_input2,
        buffer_input3,
    )
}

pub fn synchronize(dev: &WgpuDevice) -> crate::Result<()> {
    //println!("synchronize dev.command_queue lock_start");
    let mut command_queue = dev.command_queue.lock().unwrap();
    //println!("synchronize dev.command_queue lock_end");
    flush_gpu_command(dev, &mut command_queue);

    let (sender, receiver) = flume::bounded(1);
    dev.queue
        .on_submitted_work_done(move || sender.send(()).unwrap());

    dev.device.poll(wgpu::Maintain::wait()).panic_on_timeout();
    receiver
        .recv()
        .map_err(|e| Error::WebGpu(WebGpuError::from(format!("failed to synchronize {}", e))))?;
    Ok(())
}

pub async fn read_data_from_gpu_async<T: bytemuck::Pod>(
    dev: &WgpuDevice,
    buffer: Arc<BufferReference>,
) -> Vec<T> {
    //println!("read_data_from_gpu_async dev.command_queue lock_start");
    let mut command_queue = dev.command_queue.lock().unwrap();
    //println!("read_data_from_gpu_async dev.command_queue lock_end");
    flush_gpu_command(dev, &mut command_queue); //send all previous commands to the gpu
    let dest_size = buffer.size;

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

    //insert_debug_info_start(dev, &mut encoder);
    //println!("read_data_from_gpu_async BufferReference,storage lock_start");
    let buffer_storage = buffer.storage.lock().unwrap();
    //println!("read_data_from_gpu_async BufferReference,storage lock_end");
    if let Some(buffer) = buffer_storage.as_ref() {
        encoder.copy_buffer_to_buffer(&buffer.buffer, 0, &staging_buffer, 0, dest_size);
    } else {
        panic!("Unespected error at read_data from gpu. Tensor WgpuStorage did not Point to a wgpu Buffer")
    }

    //insert_debug_info_end(dev, &mut encoder, &format!("copy to cpu"));
    // Submits command encoder for processing
    dev.queue.submit(Some(encoder.finish()));

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
