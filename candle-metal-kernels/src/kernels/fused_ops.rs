use crate::metal::{Buffer, ComputeCommandEncoder, Device};
use crate::utils::EncoderProvider;
use crate::{MetalKernelError, Source};
use objc2_foundation::NSUInteger;
use objc2_metal::{MTLResourceUsage, MTLSize};

pub fn call_fused_gelu(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &crate::Kernels,
    input: &Buffer,
    output: &Buffer,
    elem_count: usize,
    kernel_name: &str,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::FusedOps, kernel_name.to_string())?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_bytes(0, &elem_count);
    encoder.set_buffer(1, Some(input), 0);
    encoder.set_buffer(2, Some(output), 0);

    let width = elem_count as NSUInteger;
    let grid_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    }; // Standard threadgroup size

    encoder.use_resource(input, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_threads(grid_size, thread_group_size);
    Ok(())
}

pub fn call_fused_layernorm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &crate::Kernels,
    input: &Buffer,
    output: &Buffer,
    weight: &Buffer,
    bias: &Buffer,
    eps: f32,
    norm_size: usize,
    batch_size: usize,
    kernel_name: &str,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::FusedOps, kernel_name.to_string())?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(output), 0);
    encoder.set_buffer(2, Some(weight), 0);
    encoder.set_buffer(3, Some(bias), 0);
    encoder.set_bytes(4, &eps);
    let norm_size = norm_size as u32;
    encoder.set_bytes(5, &norm_size);
    let batch_size = batch_size as u32;
    encoder.set_bytes(6, &batch_size);

    // Dispatch: One threadgroup per row
    let grid_size = MTLSize {
        width: batch_size as NSUInteger,
        height: 1,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    }; // Fixed by kernel template

    encoder.use_resource(input, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.use_resource(weight, MTLResourceUsage::Read);
    encoder.use_resource(bias, MTLResourceUsage::Read);
    encoder.dispatch_thread_groups(grid_size, thread_group_size);

    Ok(())
}

pub fn call_gen_rel_pos_mask(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &crate::Kernels,
    term_h: &Buffer,
    term_w: &Buffer,
    output: &Buffer,
    h: usize,
    w: usize,
    elem_count: usize,
    kernel_name: &str,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::FusedOps, kernel_name.to_string())?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(0, Some(term_h), 0);
    encoder.set_buffer(1, Some(term_w), 0);
    encoder.set_buffer(2, Some(output), 0);
    let h_u32 = h as u32;
    encoder.set_bytes(3, &h_u32);
    let w_u32 = w as u32;
    encoder.set_bytes(4, &w_u32);
    encoder.set_bytes(5, &elem_count);

    let width = elem_count as NSUInteger;
    let grid_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };

    encoder.use_resource(term_h, MTLResourceUsage::Read);
    encoder.use_resource(term_w, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_threads(grid_size, thread_group_size);

    Ok(())
}
