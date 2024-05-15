use std::borrow::Cow;

use wgpu::{util::DeviceExt, BindGroup, Buffer, ComputePipeline, ShaderModule};

use crate::{wgpu_backend::device::WgpuDevice, Layout};

use super::device::Pipelines;

//const MAX_STRIDE_INFO : usize = 4;
#[derive(Clone,Copy,bytemuck::Pod,bytemuck::Zeroable)]
#[repr(C)]
struct MetaUnary{
    operation : u32,
    length : u32, //elements
    scalar1 : f32,
    scalar2 : f32
}

#[derive(Clone,Copy,bytemuck::Pod,bytemuck::Zeroable)]
#[repr(C)]
struct MetaBinaryScalar{
    operation : u32,
    length : u32, //elements
    scalar : f32
}

//(M X N) * (N X K)
#[derive(Clone,Copy,bytemuck::Pod,bytemuck::Zeroable)]
#[repr(C)]
struct MetaInfoMatMul{
    b : u32,   //Batches
    m : u32,  //elements
    n : u32,  //elements
    k : u32,   //elements
    input1 : MatrixLayout,
    input2 : MatrixLayout
}

#[derive(Clone,Copy,bytemuck::Pod,bytemuck::Zeroable)]
#[repr(C)]
struct MatrixLayout{
    shape1 : u32, 
    shape2 : u32, 
    shape3 : u32, 
    shape4 : u32, 
    shape5 : u32, 
    stride1 : u32, 
    stride2 : u32, 
    stride3 : u32, 
    stride4 : u32, 
    stride5 : u32, 
    offset : u32,
    p : u32, 
}

impl MatrixLayout {
    fn new(shape: &[usize;5], stride : &[usize;5], offset: u32) -> Self {
        Self { shape1 : shape[0] as u32, shape2 : shape[1] as u32, shape3: shape[2] as u32, shape4: shape[3] as u32, shape5: shape[4]as u32, stride1 : stride[0]as u32, stride2: stride[1]as u32, stride3: stride[2]as u32, stride4: stride[3]as u32, stride5: stride[4]as u32, offset, p : 0 }
    }

    fn from_layout(layout : &Layout) -> Self{
        let shape = layout.shape().dims();
        let mut dims =  [1; 5];
        dims[5-shape.len()..].clone_from_slice(shape);

        let mut stride_arr = [1; 5];
        let stride = layout.stride();
        stride_arr[5-stride.len()..].clone_from_slice(stride);

        let offset = layout.start_offset();

        return Self::new(&dims, &stride_arr, offset as u32)

    }
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
    RandNormal=47,
    RandUniform=48,
    Gelu=49,
    Round=50,
    Affine=51,
    Elu=52,
    AddScalar=101,
    MultScalar=102,
    MinusScalar=103,
    DivScalar=104,
    MaxScalar=105,
    MinScalar=106,
    PowScalar=107,
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

#[derive(Copy, Clone)]
#[allow(dead_code)]
pub enum ReduceOperations{
    Sum = 0,
    Min = 1,
    Max = 2,
    ArgMin = 3,
    ArgMax = 4,
}

#[derive(Copy, Clone)]
#[allow(dead_code)]
pub enum CmpOperation{
    Eq = 0,
    Ne = 1,
    Lt = 2,
    Le= 3,
    Gt= 4,
    Ge = 5,
}


const WORKGROUP_SIZE : u32 = 64;

pub fn get_shader(device: &wgpu::Device, shader : &'static str) -> ShaderModule {
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader)),
    });
    return cs_module;
}



/// Size is in Bytes!
pub fn create_buffer(dev : &WgpuDevice, size : usize) -> Buffer{
    let buffer = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
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
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
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
        cpass.dispatch_workgroups((length + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE, 1, 1);
    }

    dev.queue.submit(Some(encoder.finish()));
}



pub fn queue_unary_inplace_op(dev : &WgpuDevice, buffer : &Buffer, length : u32, op : UnaryOperation, scalar1 : f32, scalar2 : f32, dtype : crate::DType){
    let meta = MetaUnary{length , operation : op as u32, scalar1, scalar2};
    let buffer_meta = create_uniform_buffer(dev, meta);

    let pipeline = dev.get_pipeline(
        match dtype{
            crate::DType::U32 => Pipelines::UnaryInplaceU32,
            crate::DType::F32 => Pipelines::UnaryInplace,
            _=>  panic!("can not create wgpu array of type {:?}", dtype)
        });
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

pub fn queue_unary_from_buffer_op(dev : &WgpuDevice, buffer_dest : &Buffer,buffer_input : &Buffer, length : u32, op : UnaryOperation, scalar1 : f32, scalar2 : f32, dtype : crate::DType){
    let meta = MetaUnary{length , operation : op as u32,scalar1,scalar2};
    let buffer_meta = create_uniform_buffer(dev, meta);

    let pipeline = dev.get_pipeline(
        match dtype{
            crate::DType::U32 => Pipelines::UnaryFromBufferU32,
            crate::DType::F32 => Pipelines::UnaryFromBuffer,
            _=> panic!("can not create wgpu array of type {:?}", dtype)
        });
   
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
pub fn queue_binary_buffer_inplace(dev : &WgpuDevice, buffer_dest : &Buffer, buffer_input1 : &Buffer, length : u32, op : BinaryOperation, dtype : crate::DType){
    let meta = MetaUnary{length , operation : op as u32,scalar1:0.0, scalar2:0.0};
    let buffer_meta = create_uniform_buffer(dev, meta);

    let pipeline = dev.get_pipeline(
        match dtype{
            crate::DType::U32 => Pipelines::BinaryBufferInplaceU32,
            crate::DType::F32 => Pipelines::BinaryBufferInplace,
            _=> panic!("can not create wgpu array of type {:?}", dtype)
        });

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
pub fn queue_binary_buffer_from_buffer(dev : &WgpuDevice, buffer_dest : &Buffer, buffer_input1 : &Buffer, buffer_input2 : &Buffer, length : u32, op : BinaryOperation, dtype : crate::DType){
    let meta = MetaUnary{length , operation : op as u32,scalar1:0.0, scalar2:0.0};
    let buffer_meta = create_uniform_buffer(dev, meta);

    let pipeline = dev.get_pipeline(
        match dtype{
            crate::DType::U32 => Pipelines::BinaryBufferFromBufferU32,
            crate::DType::F32 => Pipelines::BinaryBufferFromBuffer,
            _=> panic!("can not create wgpu array of type {:?}", dtype)
        });
   
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



pub fn queue_matmul_buffer(dev : &WgpuDevice, buffer_dest : &Buffer, buffer_input1 : &Buffer, buffer_input2 : &Buffer,b: u32, m : u32, n : u32, k : u32, layout_input1 : &Layout, layout_input2 : &Layout, dtype : crate::DType){
    
    let input1_info = MatrixLayout::from_layout(layout_input1);
    let input2_info = MatrixLayout::from_layout(layout_input2);
    
    let meta = MetaInfoMatMul{b,m,n,k,input1:input1_info, input2:input2_info};
    let buffer_meta = create_uniform_buffer(dev, meta);

    let pipeline = dev.get_pipeline(
        match dtype{
            crate::DType::U32 => Pipelines::MatmulBufferU32,
            crate::DType::F32 => Pipelines::MatmulBuffer,
            _=> panic!("can not create wgpu array of type {:?}", dtype)
        });
    
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
        cpass.dispatch_workgroups((k + 7) / 8, (m + 7) / 8, b);
    }

    dev.queue.submit(Some(encoder.finish()));
}

pub fn queue_reduce_from_buffer_op(dev : &WgpuDevice, buffer_dest : &Buffer,buffer_input : &Buffer, length : u32, op : ReduceOperations, dtype : crate::DType){
    let meta = MetaUnary{length , operation : op as u32,scalar1 : 0.0, scalar2: 0.0};
    let buffer_meta = create_uniform_buffer(dev, meta);

    let pipeline = dev.get_pipeline(
        match dtype{
            crate::DType::U32 => Pipelines::ReduceFromBufferU32,
            crate::DType::F32 => Pipelines::ReduceFromBuffer,
            _=> panic!("can not create wgpu array of type {:?}", dtype)
        });
   
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

    let mut encoder = dev.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }

    dev.queue.submit(Some(encoder.finish()));
}

pub fn queue_copy(dev : &WgpuDevice, buffer_dest : &Buffer, buffer_input : &Buffer, destination_offset : usize, source_offset : usize, copy_size : usize){
    let mut encoder = dev.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(buffer_input, source_offset as u64 * 4, buffer_dest, destination_offset as u64 * 4, copy_size as u64  * 4);
    dev.queue.submit(Some(encoder.finish()));
}

#[allow(dead_code)]
pub fn queue_cmp_buffer_from_buffer(dev : &WgpuDevice, buffer_dest : &Buffer, buffer_input1 : &Buffer, buffer_input2 : &Buffer, length : u32, op : CmpOperation, dtype : crate::DType){
    let meta = MetaUnary{length , operation : op as u32, scalar1:0.0, scalar2:0.0};
    let buffer_meta = create_uniform_buffer(dev, meta);

    let pipeline = dev.get_pipeline(
        match dtype{
            crate::DType::U32 => Pipelines::CmpFromBufferU32,
            crate::DType::F32 => Pipelines::CmpFromBuffer,
            _=> panic!("can not create wgpu array of type {:?}", dtype)
        });

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



pub async fn read_data_from_gpu_async<T: bytemuck::Pod>(
    dev : &WgpuDevice,
    buffer: &Buffer) -> Vec<T> {

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