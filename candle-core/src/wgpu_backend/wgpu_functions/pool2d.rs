use candle_wgpu_kernels::pool2d::Functions;

use super::*;
use crate::WgpuDevice;

pub fn queue_max_pool2d(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input1: BufferReferenceId,
    layout: &crate::Layout,
    dtype: crate::DType,
    kernel_size: (usize, usize),
    stride: (usize, usize),
) -> crate::Result<()> {
    let (b, c, h, w) = layout.shape().dims4()?;
    let h_out = (h - kernel_size.1) / stride.1 + 1;
    let w_out = (w - kernel_size.0) / stride.0 + 1;

    let input_stride = layout.stride();

    let mut queue = dev.get_queue();

    queue.add(b);
    queue.add(c);
    queue.add(kernel_size.1);
    queue.add(kernel_size.0);
    queue.add(w); //size_in_x
    queue.add(h); //size_in_y
    queue.add(w_out * h_out * c); //Stride_batch_out
    queue.add(w_out * h_out); //stride_c_out
    queue.add(w_out); //stride_y_out
    queue.add(h_out); //size_y_out

    queue.add(input_stride[0]); //stride_batch_input
    queue.add(input_stride[1]); //stride_c_in
    queue.add(input_stride[2]); //stride_y_in
    queue.add(input_stride[3]); //stride_x_in
    queue.add(stride.1);
    queue.add(stride.0);
    queue.add(layout.start_offset());

    let pipeline = queue.get_pipeline(Pipelines::Pool2d(get_dtype(dtype)?, Functions::MaxPool2d));

    let bind_group = dev.create_bind_group_input1(buffer_dest, buffer_input1, dtype.into());
    queue.enqueue_workgroups(
        pipeline,
        bind_group,
        (w_out as u32 + 7) / 8,
        (h_out as u32 + 7) / 8,
        c as u32,
        h_out * w_out * b * c,
    );
    Ok(())
}

pub fn queue_avg_pool2d(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input1: BufferReferenceId,
    layout: &crate::Layout,
    dtype: crate::DType,
    kernel_size: (usize, usize),
    stride: (usize, usize),
) -> crate::Result<()> {
    let (b, c, h, w) = layout.shape().dims4()?;
    let h_out = (h - kernel_size.1) / stride.1 + 1;
    let w_out = (w - kernel_size.0) / stride.0 + 1;

    let input_stride = layout.stride();

    let mut queue = dev.get_queue();

    queue.add(b);
    queue.add(c);
    queue.add(kernel_size.1);
    queue.add(kernel_size.0);
    queue.add(w); //size_in_x
    queue.add(h); //size_in_y
    queue.add(w_out * h_out * c); //Stride_batch_out
    queue.add(w_out * h_out); //stride_c_out
    queue.add(w_out); //stride_y_out
    queue.add(h_out); //size_y_out

    queue.add(input_stride[0]); //stride_batch_input
    queue.add(input_stride[1]); //stride_c_in
    queue.add(input_stride[2]); //stride_y_in
    queue.add(input_stride[3]); //stride_x_in
    queue.add(stride.1);
    queue.add(stride.0);
    queue.add(layout.start_offset());

    let pipeline = queue.get_pipeline(Pipelines::Pool2d(get_dtype(dtype)?, Functions::AvgPool2d));

    let bind_group = dev.create_bind_group_input1(buffer_dest, buffer_input1, dtype.into());
    queue.enqueue_workgroups(
        pipeline,
        bind_group,
        (w_out as u32 + 7) / 8,
        (h_out as u32 + 7) / 8,
        c as u32,
        w_out * h_out * c * b,
    );
    Ok(())
}
