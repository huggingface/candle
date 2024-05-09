use std::borrow::Cow;

use wgpu::{util::DeviceExt, BindGroup, Buffer, ComputePipeline, ShaderModule};

use crate::wgpu_backend::device::WgpuDevice;

use super::device::Pipelines;

#[derive(Clone,Copy,bytemuck::Pod,bytemuck::Zeroable)]
#[repr(C)]
struct MetaUnary{
    operation : u32,
    length : u32
}

#[derive(Clone,Copy,bytemuck::Pod,bytemuck::Zeroable)]
#[repr(C)]
struct MetaBinaryScalar{
    operation : u32,
    length : u32,
    scalar : f32
}

//(M X N) * (N X K)
#[derive(Clone,Copy,bytemuck::Pod,bytemuck::Zeroable)]
#[repr(C)]
struct MetaInfoMatMul{
    m : u32, 
    n : u32, 
    k : u32
}

#[derive(Copy, Clone)]
#[allow(dead_code)]
pub enum UnaryOperation{
    SetZero = 0,
    SetOne = 1,
    IncOne = 2,
    DecOne= 3,
    Abs= 4,
    Acos= 5,
    Acosh= 6,
    Asin= 7,
    Asinh= 8,
    Atan= 9,
    Atanh= 10,
    Ceil= 11,
    Cos=12,
    Cosh=13,
    Deg=17,
    Exp=21,
    Floor=22,
    Fract=23,
    InverseSqrt= 24,
    Log= 25,
    Log2= 26,
    Rad= 27,
    Sign= 28,
    Sin= 29,
    Sinh= 31,
    Sqrt= 32,
    Tan= 33,
    Tanh= 34,
    Trunc= 35,
    BinaryStep= 36, 
    Sigmoid= 37,
    Relu= 38,
    Softplus= 39,
    LeakyRelu= 40,
    SiLu= 41,
    Gassian= 42,
    Identity= 43,
    Square= 44,
    Neg= 45,
    Inverse= 46,
}

#[derive(Copy, Clone)]
#[allow(dead_code)]
pub enum BinaryOperation{
    SetY = 0,
    Add = 1,
    Mult = 2,
    Minus= 3,
    Div= 4,
    Max= 5,
    Min= 6,
    Pow= 7,
}

const WORKGROUP_SIZE : u32 = 64;

pub fn get_shader(device: &wgpu::Device) -> ShaderModule {
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });
    return cs_module;
}



pub fn create_buffer(dev : &WgpuDevice, size : usize) -> Buffer{
    let buffer = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    buffer
}

pub fn create_uniform_buffer<T : bytemuck::Pod>(dev : &WgpuDevice, value : T) -> Buffer{
    return dev.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Buffer MetaUnary"),
        contents: bytemuck::cast_slice(&[value]),
        usage: wgpu::BufferUsages::UNIFORM});
}

pub fn create_buffer_init<T : bytemuck::Pod>(dev : &WgpuDevice, data : &[T]) -> Buffer {
    let buffer = dev.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Buffer A"),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    buffer
}

fn enqueue(dev : &WgpuDevice, pipeline : &ComputePipeline, bind_group: BindGroup, length : u32){
    let mut encoder = dev.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(length / WORKGROUP_SIZE, 1, 1);
    }

    dev.queue.submit(Some(encoder.finish()));
}



pub fn queue_unary_inplace_op(dev : &WgpuDevice, buffer : &Buffer, length : u32, op : UnaryOperation){
    let meta = MetaUnary{length , operation : op as u32};
    let buffer_meta = create_uniform_buffer(dev, meta);

    let pipeline = dev.get_pipeline(Pipelines::UnaryInplace);
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer_meta.as_entire_binding(),
            },
        ],
    });

    enqueue(dev, pipeline, bind_group, length);
}

pub fn queue_unary_from_buffer_op(dev : &WgpuDevice, buffer_dest : &Buffer,buffer_input : &Buffer, length : u32, op : UnaryOperation){
    let meta = MetaUnary{length , operation : op as u32};
    let buffer_meta = create_uniform_buffer(dev, meta);

    let pipeline = dev.get_pipeline(Pipelines::UnaryFromBuffer);

    // Instantiates the bind group
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                resource: buffer_input.as_entire_binding(),
            },
        ],
    });

    enqueue(dev, pipeline, bind_group, length);
}



#[allow(dead_code)]
pub fn queue_binary_scalar_inplace(dev : &WgpuDevice, buffer_dest : &Buffer, scalar : f32, length : u32, op : BinaryOperation){
    let meta = MetaBinaryScalar{length , operation : op as u32, scalar};
    let buffer_meta = create_uniform_buffer(dev, meta);

    let pipeline = dev.get_pipeline(Pipelines::BinaryScalarInplace);

    // Instantiates the bind group
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            }
        ],
    });

    enqueue(dev,  pipeline, bind_group, length);
}

#[allow(dead_code)]
pub fn queue_binary_scalar_from_buffer(dev : &WgpuDevice, buffer_dest : &Buffer, buffer_input : &Buffer, scalar:f32, length : u32, op : BinaryOperation){
    let meta = MetaBinaryScalar{length , operation : op as u32, scalar};
    let buffer_meta = create_uniform_buffer(dev, meta);

    let pipeline = dev.get_pipeline(Pipelines::BinaryScalarFromBuffer);

    // Instantiates the bind group
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                resource: buffer_input.as_entire_binding(),
            }
        ],
    });

    enqueue(dev,  pipeline, bind_group, length);
}

#[allow(dead_code)]
pub fn queue_binary_buffer_inplace(dev : &WgpuDevice, buffer_dest : &Buffer, buffer_input1 : &Buffer, length : u32, op : BinaryOperation){
    let meta = MetaUnary{length , operation : op as u32};
    let buffer_meta = create_uniform_buffer(dev, meta);

    let pipeline = dev.get_pipeline(Pipelines::BinaryBufferInplace);

    // Instantiates the bind group
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            }
        ],
    });

    enqueue(dev,  pipeline, bind_group, length);
}

#[allow(dead_code)]
pub fn queue_binary_buffer_from_buffer(dev : &WgpuDevice, buffer_dest : &Buffer, buffer_input1 : &Buffer, buffer_input2 : &Buffer, length : u32, op : BinaryOperation){
    let meta = MetaUnary{length , operation : op as u32};
    let buffer_meta = create_uniform_buffer(dev, meta);

    let pipeline = dev.get_pipeline(Pipelines::BinaryBufferFromBuffer);

    // Instantiates the bind group
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            }
        ],
    });

    enqueue(dev, pipeline, bind_group, length);
}



pub fn queue_matmul_buffer(dev : &WgpuDevice, buffer_dest : &Buffer, buffer_input1 : &Buffer, buffer_input2 : &Buffer, m : u32, n : u32, k : u32){
    let meta = MetaInfoMatMul{m,n,k};
    let buffer_meta = create_uniform_buffer(dev, meta);

    let pipeline = dev.get_pipeline(Pipelines::MatmulBuffer);

    // Instantiates the bind group
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            }
        ],
    });

    let mut encoder = dev.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(k / 8, m / 8, 1);
    }

    dev.queue.submit(Some(encoder.finish()));
}




pub async fn read_data_from_gpu_async(
    dev : &WgpuDevice,
    buffer: &Buffer) -> Vec<f32> {

    let dest_size = buffer.size();

    let staging_buffer = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: dest_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });


    let mut encoder = dev.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    encoder.copy_buffer_to_buffer(&buffer, 0, &staging_buffer, 0, dest_size);

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
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

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