use candle_wgpu_kernels::binary::Functions;

use super::*;

#[derive(Copy, Clone, Debug)]
#[allow(dead_code)]
pub enum BinaryOperation {
    SetY = 0,
    Add = 1,
    Mult = 2,
    Minus = 3,
    Div = 4,
    Max = 5,
    Min = 6,
    Pow = 7,
}
pub fn queue_binary_buffer_from_buffer(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input1: BufferReferenceId,
    buffer_input2: BufferReferenceId,
    op: BinaryOperation,
    dtype: crate::DType,
    lay1: &crate::Layout,
    lay2: &crate::Layout,
) -> crate::Result<()> {
    let mut meta = get_meta(&dev);
    let pipeline = 
    if lay1.is_contiguous() && lay2.is_contiguous() {
        let const_vec = vec![
            op as usize,
            lay1.start_offset(),
            lay2.start_offset()];

        meta.add(lay1.shape().elem_count()); //input1_length
        
        let inplaceable = OpIsInplaceable{ input1_inplaceable: lay1.start_offset() == 0, input2_inplaceable: lay2.start_offset() == 0 };
        meta.get_pipeline_const_inplace(Pipelines::Binary(get_dtype(dtype)?, Functions::BinaryBufferFromBufferContiguousBoth) , const_vec, inplaceable)
    } else {
        let const_vec = vec![
            op as usize, lay1.shape().dims().len() as usize];
        meta.add_layout1(&lay1);
        meta.add_layout2(&lay2);

        meta.get_pipeline_const(Pipelines::Binary(get_dtype(dtype)?, Functions::BinaryBufferFromBuffer) , const_vec)
    };

    let bind_group = create_bind_group_input2(
        buffer_dest,
        buffer_input1,
        buffer_input2,
    );

    enqueue_big_extra(
        meta,
        pipeline,
        bind_group,
        lay1.shape().elem_count() as u32,
        #[cfg(feature="wgpu_debug")]
        Some(format!("OP: {:?}", op))

    );
    return Ok(());

}
