use crate::linear_split;
use crate::{
    set_params, Buffer, ComputeCommandEncoder, Device, EncoderParam, EncoderProvider, Kernels,
    MetalKernelError, Source,
};
use objc2_metal::MTLResourceUsage;

pub fn call_const_fill(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    length: usize,
    output: &Buffer,
    v: impl EncoderParam,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Fill, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(encoder, (output, v, length));
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}
