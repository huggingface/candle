use crate::utils::{BufferOffset, EncoderProvider};
use crate::{get_tile_size, linear_split};
use crate::{
    set_params, Buffer, ComputeCommandEncoder, ConstantValues, Device, Kernels, MetalKernelError,
    Source, Value,
};
use objc2_metal::MTLResourceUsage;

#[allow(clippy::too_many_arguments)]
pub fn call_where_cond(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    dtype_size: usize,
    shape: &[usize],
    cond: BufferOffset,
    cond_stride: &[usize],
    cond_is_contiguous: bool,
    left: BufferOffset,
    left_stride: &[usize],
    left_is_contiguous: bool,
    right: BufferOffset,
    right_stride: &[usize],
    right_is_contiguous: bool,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let constants = Some(ConstantValues::new(vec![
        (0, Value::Bool(cond_is_contiguous)),
        (1, Value::Bool(left_is_contiguous)),
        (2, Value::Bool(right_is_contiguous)),
    ]));
    let pipeline =
        kernels.load_pipeline_with_constants(device, Source::Ternary, name, constants)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let size: usize = shape.iter().product();
    let rank = shape.len();

    set_params!(
        encoder,
        (
            size,
            rank,
            shape,
            cond_stride,
            left_stride,
            right_stride,
            &cond,
            &left,
            &right,
            output
        )
    );

    let tile_size = get_tile_size(dtype_size);
    let tiles = size.div_ceil(tile_size);
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, tiles);

    encoder.use_resource(cond.buffer, MTLResourceUsage::Read);
    encoder.use_resource(left.buffer, MTLResourceUsage::Read);
    encoder.use_resource(right.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}
