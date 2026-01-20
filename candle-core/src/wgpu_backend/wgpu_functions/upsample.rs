use super::*;
use candle_wgpu_kernels::upsample::Functions;

pub fn queue_upsample1d(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input1: BufferReferenceId,
    layout: &crate::Layout,
    dtype: crate::DType,
    target_size: usize,
) -> crate::Result<()> {
    let (b, c, l) = layout.shape().dims3()?;

    let strides = layout.stride();

    let mut queue = dev.get_queue();

    queue.add(target_size);
    queue.add(b);
    queue.add(c);
    queue.add(l);
    queue.add(layout.start_offset());

    queue.add(strides[0]);
    queue.add(strides[1]);
    queue.add(strides[2]);

    queue.add(c * target_size);
    queue.add(target_size);

    let pipeline = queue.get_pipeline(Pipelines::Upsample(
        dev.get_dtype(dtype)?,
        Functions::Upsample1d,
    ));

    let bind_group = dev.create_bind_group_input1(buffer_dest, buffer_input1, dtype.into());
    queue.enqueue_workgroups(
        pipeline,
        bind_group,
        (target_size as u32 + 63) / 63,
        c as u32,
        b as u32,
        target_size * b * c,
    );
    Ok(())
}

pub fn queue_upsample2d(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input1: BufferReferenceId,
    layout: &crate::Layout,
    dtype: crate::DType,
    target_size: (usize, usize),
) -> crate::Result<()> {
    let (b, c, h, w) = layout.shape().dims4()?;

    let strides = layout.stride();

    let mut queue = dev.get_queue();

    queue.add(target_size.0);
    queue.add(target_size.1);
    queue.add(b);
    queue.add(c);
    queue.add(h);
    queue.add(w);
    queue.add(layout.start_offset());

    queue.add(strides[0]);
    queue.add(strides[1]);
    queue.add(strides[2]);
    queue.add(strides[3]);

    queue.add(c * target_size.0 * target_size.1);
    queue.add(target_size.0 * target_size.1);
    queue.add(target_size.1);

    let pipeline = queue.get_pipeline(Pipelines::Upsample(
        dev.get_dtype(dtype)?,
        Functions::Upsample2d,
    ));

    let bind_group = dev.create_bind_group_input1(buffer_dest, buffer_input1, dtype.into());
    queue.enqueue_workgroups(
        pipeline,
        bind_group,
        (target_size.1 as u32).div_ceil(8),
        (target_size.0 as u32).div_ceil(8),
        c as u32,
        b * c * target_size.0 * target_size.1,
    );
    Ok(())
}


#[allow(clippy::too_many_arguments)]
pub fn queue_upsample_bilinear2d(
    dev: &WgpuDevice,
    buffer_src: (BufferReferenceId, u32),
    dtype: crate::DType,
    buffer_dest: BufferReferenceId,
    n: u32,
    c: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
    align_corners: bool,
    scale_h: Option<f64>,
    scale_w: Option<f64>,
) -> crate::Result<()> {
    let (buffer_src, src_offset) = buffer_src;

    let workgroup_size_x: u32 = 8;
    let workgroup_size_y: u32 = 8;

    let num_invocations_x = n * c;
    let num_invocations_y = out_h * out_w;

    fn ceil_div(a: u32, b: u32) -> u32 {
        if a == 0 { 0 } else { a.div_ceil(b) }
    }

    let workgroup_count_x = ceil_div(num_invocations_x, workgroup_size_x);
    let workgroup_count_y = ceil_div(num_invocations_y, workgroup_size_y);

    let mut queue = dev.get_queue();

    let use_scale = scale_h.is_some() || scale_w.is_some();

    let sh = if use_scale {
        // PyTorch internal scale
        in_h as f32 / out_h as f32
    } else {
        0.0
    };

    let sw = if use_scale {
        in_w as f32 / out_w as f32
    } else {
        0.0
    };

    // op_meta
    queue.add(n);
    queue.add(c);
    queue.add(in_h);
    queue.add(in_w);
    queue.add(out_h);
    queue.add(out_w);
    queue.add(src_offset);
    queue.add(if align_corners { 1u32 } else { 0u32 });
    queue.add(if use_scale { 1u32 } else { 0u32 });
    queue.add(sh.to_bits());
    queue.add(sw.to_bits());

    let pipeline = queue.get_pipeline(Pipelines::Upsample(
        dev.get_dtype(dtype)?,
        candle_wgpu_kernels::upsample::Functions::UpsampleBilinear2d,
    ));

    let bind_group = dev.create_bind_group_input1(
        buffer_dest,
        buffer_src,
        dtype.into(),
    );

    queue.enqueue_workgroups(
        pipeline,
        bind_group,
        workgroup_count_x,
        workgroup_count_y,
        1,
        (n * c * out_h * out_w) as usize,
    );

    Ok(())
}
