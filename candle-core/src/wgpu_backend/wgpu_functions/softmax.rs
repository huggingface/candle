use candle_wgpu_kernels::softmax::Functions;

use super::*;

pub fn queue_softmax(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input1: BufferReferenceId,
    dtype: crate::DType,
    input1_offset : u32,
    reduction_length : u32,
    dest_size: u32
) -> crate::Result<()> {
    let workgroup_count = u32::min(64, (reduction_length / 10 + 1) as u32);
    let workgroup_size = reduction_length as u32 / workgroup_count + 1;
    let workgroup_count = (reduction_length + (workgroup_size - 1)) / workgroup_size; 
    
    let const_vec = vec![
        workgroup_count,
        input1_offset];

    let mut meta = get_meta(&dev);
    //meta.add(workgroup_count);
    meta.add(workgroup_size);
    meta.add(reduction_length);
    //meta.add(input1_offset);
    meta.add(dest_size);
    
    let pipeline = get_pipeline_const(Pipelines::Softmax(get_dtype(dtype)?, Functions::Softmax), const_vec);

    let bind_group: crate::wgpu_backend::cache::BindGroupReferenceBase<BufferReferenceId> = create_bind_group_input1( buffer_dest, buffer_input1);
    let id: u32 = dest_size;
    enqueue_workgroups(
        meta,
        pipeline,
        bind_group,
        1,
        (id).min(65535),
        (id + 65534) / 65535,
        (reduction_length * dest_size) as usize
    );
    return Ok(());
}
