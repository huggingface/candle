use wgpu::Buffer;

use crate::{wgpu::device::Pipelines, WgpuDevice};

use super::{create_bind_group_input1, enqueue_workgroups, MyArray};



// #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
// #[repr(C)]
// struct MetaSoftmaxContiguous{
//     workgroup_count : u32,
//     workgroup_size : u32,
//     length : u32, //Length of Reduction(e.g count of elements to sum per output),

//     input1_offset : u32,
// }


pub fn queue_softmax(
    dev: &WgpuDevice,
    buffer_dest: &Buffer,
    buffer_input1: &Buffer,
    dtype: crate::DType,
    input1_offset : u32,
    reduction_length : u32,
    dest_size: u32
) -> crate::Result<()> {
    let workgroup_count = u32::min(64, (reduction_length / 10 + 1) as u32);
    let workgroup_size = reduction_length as u32 / workgroup_count + 1;
    
    let mut meta = MyArray::new(4);
    meta.add(workgroup_count);
    meta.add(workgroup_size);
    meta.add(reduction_length);
    meta.add(input1_offset);

    // let meta = MetaSoftmaxContiguous {
    //     workgroup_count,
    //     workgroup_size,
    //     length: reduction_length as u32,
    //     input1_offset,
    // };

    let pipeline = dev.get_pipeline(super::Shader::Softmax(dtype), Pipelines::Softmax)?;

    let bind_group = create_bind_group_input1(dev, pipeline.clone(), &meta.0, buffer_dest, buffer_input1);
    enqueue_workgroups(
        dev,
        pipeline,
        bind_group,
        1,
        dest_size,
        1,
        #[cfg(feature = "wgpu_debug")] &format!("softmax, dtype:{:?}", dtype),
    );
    return Ok(());
}
