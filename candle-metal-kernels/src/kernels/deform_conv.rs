//! Deformable Convolution 2D Metal kernels
//!
//! This module provides Metal implementations of deformable convolution operations
//! as described in "Deformable Convolutional Networks" (Dai et al., 2017).
//!
//! Ported from mps-deform-conv: https://github.com/mpsops/mps-deform-conv

use crate::utils::{EncoderProvider, BufferOffset};
use crate::{set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError, Source};
use crate::linear_split;
use objc2_metal::MTLResourceUsage;

/// Configuration for deformable im2col operation
#[derive(Debug, Clone)]
pub struct DeformConv2dConfig {
    pub height: usize,
    pub width: usize,
    pub weight_h: usize,
    pub weight_w: usize,
    pub pad_h: usize,
    pub pad_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub dilation_h: usize,
    pub dilation_w: usize,
    pub batch_sz: usize,
    pub n_in_channels: usize,
    pub n_offset_grps: usize,
    pub out_h: usize,
    pub out_w: usize,
    pub use_mask: bool,
}

impl DeformConv2dConfig {
    pub fn new(
        height: usize,
        width: usize,
        weight_h: usize,
        weight_w: usize,
        pad: (usize, usize),
        stride: (usize, usize),
        dilation: (usize, usize),
        batch_sz: usize,
        n_in_channels: usize,
        n_offset_grps: usize,
        use_mask: bool,
    ) -> Self {
        let out_h = (height + 2 * pad.0 - dilation.0 * (weight_h - 1) - 1) / stride.0 + 1;
        let out_w = (width + 2 * pad.1 - dilation.1 * (weight_w - 1) - 1) / stride.1 + 1;

        Self {
            height,
            width,
            weight_h,
            weight_w,
            pad_h: pad.0,
            pad_w: pad.1,
            stride_h: stride.0,
            stride_w: stride.1,
            dilation_h: dilation.0,
            dilation_w: dilation.1,
            batch_sz,
            n_in_channels,
            n_offset_grps,
            out_h,
            out_w,
            use_mask,
        }
    }
}

/// Forward pass: deformable im2col
///
/// Converts input image to columns using deformable sampling positions.
/// The output can then be used with a standard matrix multiplication for convolution.
#[allow(clippy::too_many_arguments)]
pub fn call_deformable_im2col(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    cfg: &DeformConv2dConfig,
    input: BufferOffset,
    offset: BufferOffset,
    mask: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::DeformConv, name)?;

    let n_elements = cfg.n_in_channels * cfg.out_h * cfg.out_w * cfg.batch_sz;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, n_elements);

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let use_mask_int: i32 = if cfg.use_mask { 1 } else { 0 };

    set_params!(
        encoder,
        (
            &input,
            &offset,
            &mask,
            output,
            cfg.height as i32,
            cfg.width as i32,
            cfg.weight_h as i32,
            cfg.weight_w as i32,
            cfg.pad_h as i32,
            cfg.pad_w as i32,
            cfg.stride_h as i32,
            cfg.stride_w as i32,
            cfg.dilation_h as i32,
            cfg.dilation_w as i32,
            cfg.batch_sz as i32,
            cfg.n_in_channels as i32,
            cfg.n_offset_grps as i32,
            cfg.out_h as i32,
            cfg.out_w as i32,
            use_mask_int
        )
    );

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(offset.buffer, MTLResourceUsage::Read);
    encoder.use_resource(mask.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}

/// Backward pass: gradient for input (col2im)
///
/// Computes the gradient of the input tensor from the column gradient.
#[allow(clippy::too_many_arguments)]
pub fn call_deformable_col2im(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    cfg: &DeformConv2dConfig,
    col: BufferOffset,
    offset: BufferOffset,
    mask: BufferOffset,
    grad_im: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::DeformConv, name)?;

    let n_elements = cfg.n_in_channels * cfg.weight_h * cfg.weight_w * cfg.out_h * cfg.out_w * cfg.batch_sz;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, n_elements);

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let use_mask_int: i32 = if cfg.use_mask { 1 } else { 0 };

    set_params!(
        encoder,
        (
            &col,
            &offset,
            &mask,
            grad_im,
            cfg.n_in_channels as i32,
            cfg.height as i32,
            cfg.width as i32,
            cfg.weight_h as i32,
            cfg.weight_w as i32,
            cfg.pad_h as i32,
            cfg.pad_w as i32,
            cfg.stride_h as i32,
            cfg.stride_w as i32,
            cfg.dilation_h as i32,
            cfg.dilation_w as i32,
            cfg.batch_sz as i32,
            cfg.n_offset_grps as i32,
            cfg.out_h as i32,
            cfg.out_w as i32,
            use_mask_int
        )
    );

    encoder.use_resource(col.buffer, MTLResourceUsage::Read);
    encoder.use_resource(offset.buffer, MTLResourceUsage::Read);
    encoder.use_resource(mask.buffer, MTLResourceUsage::Read);
    encoder.use_resource(grad_im, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}

/// Backward pass: gradient for offsets and mask
///
/// Computes gradients for the offset and mask tensors.
#[allow(clippy::too_many_arguments)]
pub fn call_deformable_col2im_coord(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    cfg: &DeformConv2dConfig,
    col: BufferOffset,
    im: BufferOffset,
    offset: BufferOffset,
    mask: BufferOffset,
    grad_offset: &Buffer,
    grad_mask: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::DeformConv, name)?;

    let offset_channels = 2 * cfg.weight_h * cfg.weight_w * cfg.n_offset_grps;
    let n_elements = cfg.out_h * cfg.out_w * offset_channels * cfg.batch_sz;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, n_elements);

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let use_mask_int: i32 = if cfg.use_mask { 1 } else { 0 };

    set_params!(
        encoder,
        (
            &col,
            &im,
            &offset,
            &mask,
            grad_offset,
            grad_mask,
            cfg.n_in_channels as i32,
            cfg.height as i32,
            cfg.width as i32,
            cfg.weight_h as i32,
            cfg.weight_w as i32,
            cfg.pad_h as i32,
            cfg.pad_w as i32,
            cfg.stride_h as i32,
            cfg.stride_w as i32,
            cfg.dilation_h as i32,
            cfg.dilation_w as i32,
            cfg.batch_sz as i32,
            cfg.n_offset_grps as i32,
            cfg.out_h as i32,
            cfg.out_w as i32,
            use_mask_int
        )
    );

    encoder.use_resource(col.buffer, MTLResourceUsage::Read);
    encoder.use_resource(im.buffer, MTLResourceUsage::Read);
    encoder.use_resource(offset.buffer, MTLResourceUsage::Read);
    encoder.use_resource(mask.buffer, MTLResourceUsage::Read);
    encoder.use_resource(grad_offset, MTLResourceUsage::Write);
    encoder.use_resource(grad_mask, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}
