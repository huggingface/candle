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

    let mut meta = get_meta(dev);

    meta.add(target_size);
    meta.add(b);
    meta.add(c);
    meta.add(l);
    meta.add(layout.start_offset());

    meta.add(strides[0]);
    meta.add(strides[1]);
    meta.add(strides[2]);

    meta.add(c * target_size);
    meta.add(target_size);

    let pipeline = meta.get_pipeline(Pipelines::Upsample(
        get_dtype(dtype)?,
        Functions::Upsample1d,
    ));

    let bind_group = create_bind_group_input1(buffer_dest, buffer_input1, dtype.into());
    enqueue_workgroups(
        meta,
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

    let mut meta = get_meta(dev);

    meta.add(target_size.0);
    meta.add(target_size.1);
    meta.add(b);
    meta.add(c);
    meta.add(h);
    meta.add(w);
    meta.add(layout.start_offset());

    meta.add(strides[0]);
    meta.add(strides[1]);
    meta.add(strides[2]);
    meta.add(strides[3]);

    meta.add(c * target_size.0 * target_size.1);
    meta.add(target_size.0 * target_size.1);
    meta.add(target_size.1);

    let pipeline = meta.get_pipeline(Pipelines::Upsample(
        get_dtype(dtype)?,
        Functions::Upsample2d,
    ));

    let bind_group = create_bind_group_input1(buffer_dest, buffer_input1, dtype.into());
    enqueue_workgroups(
        meta,
        pipeline,
        bind_group,
        (target_size.1 as u32 + 7) / 8,
        (target_size.0 as u32 + 7) / 8,
        c as u32,
        b * c * target_size.0 * target_size.1,
    );
    Ok(())
}
