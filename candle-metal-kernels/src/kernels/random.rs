use crate::linear_split;
use crate::utils::EncoderProvider;
use crate::{
    set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError, Output, Source,
};

#[allow(clippy::too_many_arguments)]
pub fn call_random_uniform(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    min: f32,
    max: f32,
    length: usize,
    seed: &Buffer,
    buffer: &Buffer,
) -> Result<(), MetalKernelError> {
    if min >= max {
        return Err(MetalKernelError::LoadLibraryError(
            "min must be less than max".to_string(),
        ));
    }
    let pipeline = kernels.load_pipeline(device, Source::Random, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();

    let odd = (length % 2 != 0) as usize;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length / 2 + odd);

    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (length, min, max, Output::new(seed), Output::new(buffer))
    );

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_random_normal(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    mean: f32,
    stddev: f32,
    length: usize,
    seed: &Buffer,
    buffer: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Random, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();

    let odd = (length % 2 != 0) as usize;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length / 2 + odd);

    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (length, mean, stddev, Output::new(seed), Output::new(buffer))
    );

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}
