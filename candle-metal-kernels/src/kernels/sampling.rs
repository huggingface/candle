use crate::utils::{get_tile_size, BufferOffset, EncoderProvider};
use crate::{
    linear_split, set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError,
    Source,
};
use objc2_metal::MTLResourceUsage;

pub fn call_repeat_penalty(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    vocab_size: usize,
    input: BufferOffset,
    output: &Buffer,
    context: &Buffer,
    context_size: usize,
    penalty: f32,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Sampling, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            &input,
            output,
            context,
            vocab_size as u32,
            context_size as u32,
            penalty
        )
    );
    // f32 only for now
    let tile_size = get_tile_size(size_of::<f32>());
    let tiles = vocab_size.div_ceil(tile_size);
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, tiles);
    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.use_resource(context, MTLResourceUsage::Read);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}
