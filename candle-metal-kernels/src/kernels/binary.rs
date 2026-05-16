use crate::kernels::macros::ops;
use crate::utils::{BufferOffset, EncoderProvider};
use crate::{get_tile_size, linear_split};
use crate::{
    set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError, Output, Source,
};

ops!(badd, bsub, bmul, bdiv, bminimum, bmaximum, eq, ne, le, lt, ge, gt);

#[allow(clippy::too_many_arguments)]
pub fn call_binary_contiguous<S: ToString>(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: S,
    dtype_size: usize,
    length: usize,
    left: BufferOffset,
    right: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Binary, kernel_name.to_string())?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (length, &left, &right, Output::new(output)));

    let tile_size = get_tile_size(dtype_size);
    let tiles = length.div_ceil(tile_size);
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, tiles);

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_binary_strided<S: ToString>(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: S,
    dtype_size: usize,
    shape: &[usize],
    left_input: BufferOffset,
    left_strides: &[usize],
    right_input: BufferOffset,
    right_strides: &[usize],
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Binary, kernel_name.to_string())?;

    let num_dims: usize = shape.len();
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    let length: usize = shape.iter().product();
    let tile_size = get_tile_size(dtype_size);
    let tiles = length.div_ceil(tile_size);
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, tiles);

    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            length,
            num_dims,
            shape,
            left_strides,
            right_strides,
            &left_input,
            &right_input,
            Output::new(output)
        )
    );
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}
