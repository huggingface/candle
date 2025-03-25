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
        (target_size.1 as u32 + 7) / 8,
        (target_size.0 as u32 + 7) / 8,
        c as u32,
        b * c * target_size.0 * target_size.1,
    );
    Ok(())
}
