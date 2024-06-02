use wgpu::Buffer;

use crate::{wgpu::device::Pipelines, WgpuDevice};

use super::{create_bind_group_input2, enqueue_workgroups, MyArray};



// #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
// #[repr(C)]
// struct MetaInfoRmsNormContiguous{
//     workgroup_count : u32,
//     workgroup_size : u32,
//     length : u32, //Length of Reduction(e.g count of elements to sum per output),

//     input1_offset : u32,
//     input2_offset : u32,

//     eps : f32
// }

pub fn queue_rms_norm(
    dev: &WgpuDevice,
    buffer_dest: &Buffer,
    buffer_input1: &Buffer,
    buffer_alpha: &Buffer,
    dtype: crate::DType,
    input1_offset : u32,
    alpha_offset : u32,
    reduction_length : u32,
    dest_size: u32,
    eps : f32
) -> crate::Result<()> {
    let workgroup_count = u32::min(64, (reduction_length / 10 + 1) as u32);
    let workgroup_size = reduction_length as u32 / workgroup_count + 1;
    
    let mut meta = MyArray::new(5);
    meta.add(workgroup_count);
    meta.add(workgroup_size);
    meta.add(reduction_length);
    meta.add(input1_offset);
    meta.add(alpha_offset);
    meta.add(eps);

    // let meta = MetaInfoRmsNormContiguous {
    //     workgroup_count,
    //     workgroup_size,
    //     length: reduction_length as u32,
    //     input1_offset,
    //     input2_offset: alpha_offset,
    //     eps,
    // };

    let pipeline = dev.get_pipeline(super::Shader::RmsNorm(dtype), Pipelines::RmsNorm)?;

    let bind_group = create_bind_group_input2(dev, pipeline.clone(), &meta.0, buffer_dest, buffer_input1, buffer_alpha);
    enqueue_workgroups(
        dev,
        pipeline,
        bind_group,
        1,
        dest_size,
        1,
        #[cfg(feature = "wgpu_debug")] &format!("rms_norm, dtype:{:?}", dtype),
    );
    return Ok(());
}
