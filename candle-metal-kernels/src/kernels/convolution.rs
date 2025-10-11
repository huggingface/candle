use crate::linear_split;
use crate::utils::{BufferOffset, EncoderProvider};
use crate::{set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError, Source};
use objc2_metal::MTLResourceUsage;

#[allow(clippy::too_many_arguments)]
pub fn call_im2col1d_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    (k_size, stride, padding, dilation): (usize, usize, usize, usize),
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;
    let l_out = (shape[2] + 2 * padding - dilation * (k_size - 1) - 1) / stride + 1;
    let dst_el = shape[0] * l_out * shape[1] * k_size;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (dst_el, l_out, k_size, stride, padding, dilation, shape, strides, &input, output)
    );
    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_col2im1d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    k_size: usize,
    stride: usize,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;
    let l_in = shape[1];
    let c_out = shape[2];
    let l_out = (l_in - 1) * stride + k_size;
    let dst_el = shape[0] * c_out * l_out;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (dst_el, l_out, l_in, c_out, k_size, stride, &input, output)
    );
    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_im2col_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    (h_k, w_k, stride, padding, dilation): (usize, usize, usize, usize, usize),
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;

    let h = shape[2];
    let w = shape[3];
    let h_out = (h + 2 * padding - dilation * (h_k - 1) - 1) / stride + 1;
    let w_out = (w + 2 * padding - dilation * (w_k - 1) - 1) / stride + 1;

    let dst_el = shape[0] * h_out * w_out * shape[1] * h_k * w_k;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            dst_el, h_out, w_out, h_k, w_k, stride, padding, dilation, shape, strides, &input,
            output
        )
    );
    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_upsample_nearest_2d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    out_w: usize,
    out_h: usize,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;
    let dst_el = out_w * out_h * shape[0] * shape[1];
    let scale_w = shape[2] as f32 / out_w as f32;
    let scale_h = shape[3] as f32 / out_h as f32;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (out_w, out_h, scale_w, scale_h, shape, strides, &input, output)
    );
    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_pool2d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    out_w: usize,
    out_h: usize,
    w_k: usize,
    h_k: usize,
    w_stride: usize,
    h_stride: usize,
    input: &Buffer,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let dst_el = out_w * out_h * shape[0] * shape[1];
    let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (w_k, h_k, w_stride, h_stride, shape, strides, input, output)
    );
    encoder.use_resource(input, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_conv_transpose1d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    dilation: usize,
    stride: usize,
    padding: usize,
    out_padding: usize,
    c_out: usize,
    l_out: usize,
    b_size: usize,
    src_shape: &[usize],
    src_strides: &[usize],
    kernel_shape: &[usize],
    kernel_strides: &[usize],
    input: &Buffer,
    input_offset: usize,
    kernel: &Buffer,
    kernel_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let dst_el = c_out * l_out * b_size;
    let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            l_out,
            stride,
            padding,
            out_padding,
            dilation,
            src_shape,
            src_strides,
            kernel_shape,
            kernel_strides,
            (input, input_offset),
            (kernel, kernel_offset),
            output
        )
    );
    encoder.use_resource(input, MTLResourceUsage::Read);
    encoder.use_resource(kernel, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

pub struct CallConvTranspose2dCfg<'a> {
    pub dilation: usize,
    pub stride: usize,
    pub padding: usize,
    pub output_padding: usize,
    pub c_out: usize,
    pub out_w: usize,
    pub out_h: usize,
    pub b_size: usize,
    pub input_dims: &'a [usize],
    pub input_stride: &'a [usize],
    pub kernel_dims: &'a [usize],
    pub kernel_stride: &'a [usize],
    pub input_offset: usize,
    pub kernel_offset: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn call_conv_transpose2d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    cfg: CallConvTranspose2dCfg,
    input: &Buffer,
    kernel: &Buffer,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let dst_el = cfg.c_out * cfg.out_w * cfg.out_h * cfg.b_size;
    let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            cfg.out_w,
            cfg.out_h,
            cfg.stride,
            cfg.padding,
            cfg.output_padding,
            cfg.dilation,
            cfg.input_dims,
            cfg.input_stride,
            cfg.kernel_dims,
            cfg.kernel_stride,
            (input, cfg.input_offset),
            (kernel, cfg.kernel_offset),
            output
        )
    );
    encoder.use_resource(input, MTLResourceUsage::Read);
    encoder.use_resource(kernel, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}
