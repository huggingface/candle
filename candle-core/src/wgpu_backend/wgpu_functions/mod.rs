
pub mod binary;
pub mod cmp;
pub mod conv2d;
pub mod convert;
pub mod copy;
pub mod index_select;
pub mod matmul;
pub mod reduce;
pub mod rms_norm;
pub mod unary;
pub mod where_cond;

use std::{borrow::Cow, sync::Arc};
use wgpu::{util::DeviceExt, BindGroup, Buffer, ComputePipeline, ShaderModule};
use crate::{wgpu_backend::device::WgpuDevice,Layout, WebGpuError};
use super::device::MlQueue;
use crate::DType;

pub use binary::{queue_binary_buffer_from_buffer, queue_binary_buffer_inplace};
pub use cmp::queue_cmp_buffer_from_buffer;
pub use conv2d::{queue_conv2d, queue_conv2d_transpose};
pub use convert::{queue_convert_f32_to_u32, queue_convert_u32_to_f32, queue_convert_u8_to_f32};
pub use copy::{queue_copy, queue_copy2d, queue_copy_strided};
pub use index_select::queue_index_select;
pub use matmul::queue_matmul_buffer;
pub use reduce::queue_reduce_from_buffer_op;
pub use rms_norm::queue_rms_norm;
pub use unary::{queue_unary_from_buffer_op,queue_unary_inplace_op};
pub use where_cond::queue_where_cond_u32;



#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct MatrixLayout {
    shape1: u32,
    shape2: u32,
    shape3: u32,
    shape4: u32,
    shape5: u32,
    stride1: u32,
    stride2: u32,
    stride3: u32,
    stride4: u32,
    stride5: u32,
    offset: u32,
    length: u32,
}

impl MatrixLayout {
    fn new(shape: &[usize; 5], stride: &[usize; 5], offset: u32, length: u32) -> Self {
        Self {
            shape1: shape[0] as u32,
            shape2: shape[1] as u32,
            shape3: shape[2] as u32,
            shape4: shape[3] as u32,
            shape5: shape[4] as u32,
            stride1: stride[0] as u32,
            stride2: stride[1] as u32,
            stride3: stride[2] as u32,
            stride4: stride[3] as u32,
            stride5: stride[4] as u32,
            offset,
            length,
        }
    }

    fn from_layout(layout: &Layout) -> Self {
        let shape = layout.shape().dims();
        let mut dims = [1; 5];
        //dims[..shape.len()].clone_from_slice(shape);
        dims[5 - shape.len()..].clone_from_slice(shape);

        let mut stride_arr = [1; 5];
        let stride = layout.stride();
        //stride_arr[..stride.len()].clone_from_slice(stride);
        stride_arr[5 - stride.len()..].clone_from_slice(stride);

        let offset = layout.start_offset();

        if layout.is_contiguous() {
            return Self::new(
                &dims,
                &stride_arr,
                offset as u32,
                layout.shape().elem_count() as u32,
            );
        } else {
            return Self::new(&dims, &stride_arr, offset as u32, 0);
        }
    }
}


#[derive(Debug, Hash, std::cmp::Eq, std::cmp::PartialEq, Clone)]
pub enum Shader{
    Binary(DType),
    Cmp(DType),
    Conv2D(DType),
    Convert(DType),
    Copy(DType),
    IndexSelect(DType),
    Matmul(DType),
    Reduce(DType),
    RmsNorm(DType),
    Unary(DType),
    WhereCond(DType)       
}

pub fn load_shader(shader : Shader) -> crate::Result<&'static str>{
    match shader {
        Shader::Binary(DType::F32) => Ok(include_str!("binary/generated/shader.pwgsl_generated_f32.wgsl")),
        Shader::Binary(DType::U32) => Ok(include_str!("binary/generated/shader.pwgsl_generated_u32.wgsl")),
        Shader::Cmp(DType::F32) => Ok(include_str!("cmp/generated/shader.pwgsl_generated_f32.wgsl")),
        Shader::Cmp(DType::U32) => Ok(include_str!("cmp/generated/shader.pwgsl_generated_u32.wgsl")),
        Shader::Conv2D(DType::F32) => Ok(include_str!("conv2d/generated/shader.pwgsl_generated_f32.wgsl")),
        Shader::Conv2D(DType::U32) => Ok(include_str!("conv2d/generated/shader.pwgsl_generated_u32.wgsl")),
        Shader::Convert(DType::F32) => Ok(include_str!("convert/generated/shader.pwgsl_generated_f32.wgsl")),
        Shader::Convert(DType::U32) => Ok(include_str!("convert/generated/shader.pwgsl_generated_u32.wgsl")),
        Shader::Convert(DType::U8) => Ok(include_str!("convert/generated/shader.pwgsl_generated_u8.wgsl")),
        Shader::Copy(DType::F32) => Ok(include_str!("copy/generated/shader.pwgsl_generated_f32.wgsl")),
        Shader::Copy(DType::U32) => Ok(include_str!("copy/generated/shader.pwgsl_generated_u32.wgsl")),
        Shader::IndexSelect(DType::F32) => Ok(include_str!("index_select/generated/shader.pwgsl_generated_f32.wgsl")),
        Shader::IndexSelect(DType::U32) => Ok(include_str!("index_select/generated/shader.pwgsl_generated_u32.wgsl")),
        Shader::Matmul(DType::F32) => Ok(include_str!("matmul/generated/shader.pwgsl_generated_f32.wgsl")),
        Shader::Matmul(DType::U32) => Ok(include_str!("matmul/generated/shader.pwgsl_generated_u32.wgsl")),
        Shader::Reduce(DType::F32) => Ok(include_str!("reduce/generated/shader.pwgsl_generated_f32.wgsl")),
        Shader::Reduce(DType::U32) => Ok(include_str!("reduce/generated/shader.pwgsl_generated_u32.wgsl")),
        Shader::RmsNorm(DType::F32) => Ok(include_str!("rms_norm/generated/shader.pwgsl_generated_f32.wgsl")),
        Shader::RmsNorm(DType::U32) => Ok(include_str!("rms_norm/generated/shader.pwgsl_generated_u32.wgsl")),
        Shader::Unary(DType::F32) => Ok(include_str!("unary/generated/shader.pwgsl_generated_f32.wgsl")),
        Shader::Unary(DType::U32) => Ok(include_str!("unary/generated/shader.pwgsl_generated_u32.wgsl")),
        Shader::WhereCond(DType::F32) => Ok(include_str!("where_cond/generated/shader.pwgsl_generated_f32.wgsl")),
        Shader::WhereCond(DType::U32) => Ok(include_str!("where_cond/generated/shader.pwgsl_generated_u32.wgsl")),
       
        _ => Err(crate::Error::WebGpu(WebGpuError::Message(format!("Could not find Pipeline: {:?}", shader))))
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
pub fn create_buffer(dev: &WgpuDevice, size: usize) -> Buffer {
    let buffer = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    buffer
}

pub fn create_uniform_buffer<T: bytemuck::Pod>(dev: &WgpuDevice, value: T, name: &str) -> Buffer {
    return dev
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(name),
            contents: bytemuck::cast_slice(&[value]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
}

pub fn create_buffer_init<T: bytemuck::Pod>(dev: &WgpuDevice, data: &[T]) -> Buffer {
    let buffer = dev
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

    buffer
}

fn enqueue_workgroups(
    dev: &WgpuDevice,
    pipeline: Arc<ComputePipeline>,
    bind_group: BindGroup,
    x: u32,
    y: u32,
    z: u32,
    _name: &str,
) {
    let q = MlQueue::Dispatch(
        super::device::MlQueueDispatch{ x, y, z, pipeline: pipeline, bind_group, indirect_buffer: None, #[cfg(feature = "wgpu_debug")] name: Some(_name.to_owned())});
    dev.command_queue.lock().unwrap().push(q);
}


fn enqueue_workgroups_indirect(
    dev: &WgpuDevice,
    pipeline: Arc<ComputePipeline>,
    bind_group: BindGroup,
    x: u32,
    y: u32,
    z: u32,
    _name: &str,
) {
    let data = wgpu::util::DispatchIndirectArgs { x, y, z };

    let workgroup_buffer = dev
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(data.as_bytes()),
            usage: wgpu::BufferUsages::INDIRECT,
        });
    
    let q = MlQueue::Dispatch(
        super::device::MlQueueDispatch{ 
            x,
            y, 
            z, 
            pipeline: pipeline, 
            bind_group, 
            indirect_buffer : Some(workgroup_buffer),
            #[cfg(feature = "wgpu_debug")]
            name: Some(_name.to_owned())
    });
    dev.command_queue.lock().unwrap().push(q);
}


fn flush_gpu_command(dev: &WgpuDevice){
    let mut queue = dev.command_queue.lock().unwrap();
    if queue.len() > 0{
        #[cfg(feature = "wgpu_debug")]
        let global_index =  dev.debug.counter.load(std::sync::atomic::Ordering::Relaxed);
        #[cfg(feature = "wgpu_debug")]
        let query_set = dev.device.create_query_set(&wgpu::QuerySetDescriptor {
            count: queue.len() as u32 * 2, // We need 2 queries: one for start and one for end
            ty: wgpu::QueryType::Timestamp,
            label: None,
            });
        
        

        let mut encoder = dev.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });

            #[cfg(feature = "wgpu_debug")]
            let mut debug_index = 0;
            
            for q in queue.iter(){
                match q{
                    MlQueue::Dispatch(q) => {
                        
                        #[cfg(feature = "wgpu_debug")]
                        cpass.write_timestamp(&query_set, debug_index);
                        
                        cpass.set_pipeline(&q.pipeline);
                        cpass.set_bind_group(0, &q.bind_group, &[]);
                        if let Some(indirect_buffer) = &q.indirect_buffer{
                            cpass.dispatch_workgroups_indirect(indirect_buffer, 0);
                        }
                        else{
                            cpass.dispatch_workgroups(q.x, q.y, q.z);
                        }

                        #[cfg(feature = "wgpu_debug")]
                        {
                            cpass.write_timestamp(&query_set, debug_index + 1);
                            dev.debug.insert_info(global_index + debug_index * 8, q.name.as_ref().unwrap().to_owned());
                            debug_index+= 2;
                        }
                    }
                }
            }
        }
        #[cfg(feature = "wgpu_debug")]
        {
            if global_index % 256 != 0{
                panic!("global_index was:{global_index}")
            }
            encoder.resolve_query_set(
                &query_set,
                0..queue.len() as u32,
                &dev.debug.query_set_buffer,
                global_index as u64,
            );
            let global_index = global_index + (queue.len() * 2 * 8) as u32;
    
            let remainder = global_index % 256;
            let global_index = if remainder == 0 {
                global_index
            } else {
                global_index + (256 - remainder)
            };
            dev.debug.counter.store(global_index, std::sync::atomic::Ordering::Relaxed);
        }
        
        dev.queue.submit(Some(encoder.finish()));
        queue.clear();
    }
}


fn enqueue(
    dev: &WgpuDevice,
    pipeline: Arc<ComputePipeline>,
    bind_group: BindGroup,
    length: u32,
    name: &str,
) {
    return enqueue_workgroups(
        dev,
        pipeline,
        bind_group,
        (length + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE,
        1,
        1,
        name,
    );
}

fn create_bind_group_input0<T: bytemuck::Pod>(
    dev: &WgpuDevice,
    pipeline: Arc<ComputePipeline>,
    meta: T,
    buffer_dest: &Buffer,
) -> BindGroup {
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let buffer_meta = create_uniform_buffer(dev, meta, "input0");
    dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_dest.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer_meta.as_entire_binding(),
            },
        ],
    })
}

fn create_bind_group_input1<T: bytemuck::Pod>(
    dev: &WgpuDevice,
    pipeline: Arc<ComputePipeline>,
    meta: T,
    buffer_dest: &Buffer,
    buffer_input1: &Buffer,
) -> BindGroup {
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let buffer_meta = create_uniform_buffer(dev, meta, "input1");
    dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_dest.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer_meta.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffer_input1.as_entire_binding(),
            },
        ],
    })
}

fn create_bind_group_input2<T: bytemuck::Pod>(
    dev: &WgpuDevice,
    pipeline: Arc<ComputePipeline>,
    meta: T,
    buffer_dest: &Buffer,
    buffer_input1: &Buffer,
    buffer_input2: &Buffer,
) -> BindGroup {
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let buffer_meta = create_uniform_buffer(dev, meta, "input2");
    dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_dest.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer_meta.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffer_input1.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buffer_input2.as_entire_binding(),
            },
        ],
    })
}


fn create_bind_group_input3<T: bytemuck::Pod>(
    dev: &WgpuDevice,
    pipeline: Arc<ComputePipeline>,
    meta: T,
    buffer_dest: &Buffer,
    buffer_input1: &Buffer,
    buffer_input2: &Buffer,
    buffer_input3: &Buffer,
) -> BindGroup {
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let buffer_meta = create_uniform_buffer(dev, meta, "input3");
    dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_dest.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer_meta.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffer_input1.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buffer_input2.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: buffer_input3.as_entire_binding(),
            },
        ],
    })
}


pub async fn read_data_from_gpu_async<T: bytemuck::Pod>(
    dev: &WgpuDevice,
    buffer: &Buffer,
) -> Vec<T> {
    flush_gpu_command(dev); //send all previous commands to the gpu 
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

    //insert_debug_info_start(dev, &mut encoder);

    encoder.copy_buffer_to_buffer(&buffer, 0, &staging_buffer, 0, dest_size);

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
