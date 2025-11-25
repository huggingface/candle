use candle_wgpu_kernels::copy::Functions;

use super::*;

pub fn queue_copy_strided(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input: BufferReferenceId,
    dtype: crate::DType,
    input_layout: &crate::Layout,
    dst_offset: u32,
) -> crate::Result<()> {
    if input_layout.shape().elem_count() > 0 {
        let result = input_layout
            .shape()
            .dims()
            .iter()
            .zip(input_layout.stride())
            .filter(|(dim, _)| **dim > 1)
            .map(|(dim, stride)| (*dim, *stride))
            .collect::<Vec<(usize, usize)>>();
        let (shape, stride): (Vec<usize>, Vec<usize>) = result.into_iter().unzip();
        if shape.len() == 3 {
            //try copy 3d
            if dst_offset == 0 {
                let layout: Layout = Layout::new(
                    crate::Shape::from_dims(&shape),
                    stride,
                    input_layout.start_offset(),
                );
                return queue_copy3d(
                    dev,
                    buffer_dest,
                    buffer_input,
                    dtype,
                    &layout,
                    (shape[0] as u32, shape[1] as u32, shape[2] as u32),
                    &Layout::contiguous(shape),
                );
            }
        }

        let mut queue = dev.get_queue();
        queue.add(dst_offset);
        queue.add_layout1(input_layout);

        if input_layout.shape().elem_count() > 65535 * 64 {
            queue.add_const(candle_wgpu_kernels::Constants::UseZ, true);
        }

        let pipeline =
            queue.get_pipeline(Pipelines::Copy(dev.get_dtype(dtype)?, Functions::CopyStrided));

        let bind_group = dev.create_bind_group_input1(buffer_dest, buffer_input, dtype.into());
        queue.enqueue_64_big_extra(
            pipeline,
            bind_group,
            input_layout.shape().elem_count() as u32,
            #[cfg(feature = "wgpu_debug")]
            Some(format!(
                "shape: {:?}, stride: {:?}",
                input_layout.shape(),
                input_layout.stride()
            )),
        );
    }
    Ok(())
}

pub fn queue_copy(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input: BufferReferenceId,
    destination_offset: usize,
    source_offset: usize,
    copy_size: usize,
    dtype: crate::DType,
) -> crate::Result<()> {
    if copy_size > 0 {
        let const_vec = vec![
            (source_offset == 0) as u32,
            (destination_offset == 0) as u32,
        ];

        let mut queue = dev.get_queue();

        let inplaceble = OpIsInplaceable {
            input1_inplaceable: destination_offset == source_offset,
            input2_inplaceable: false,
        };

        let use_vec4 = copy_size.is_multiple_of(4)
            && source_offset.is_multiple_of(4)
            && destination_offset.is_multiple_of(4)
            && dtype.size_in_bytes() == 4;

        if use_vec4 {
            queue.add(copy_size / 4);
            queue.add(destination_offset / 4);
            queue.add(source_offset / 4);
            if copy_size / 4 > 65535 * 64 {
                queue.add_const(candle_wgpu_kernels::Constants::UseZ, true);
            }

            let pipeline = queue.get_pipeline_const_inplace(
                Pipelines::Copy(DType::U32, Functions::Copy4),
                const_vec,
                inplaceble,
            );
            let bind_group = dev.create_bind_group_input1(
                buffer_dest,
                buffer_input,
                BindgroupAlignment::Aligned16,
            );
            queue.enqueue_64_big(pipeline, bind_group, (copy_size / 4) as u32);
        } else {
            queue.add(copy_size);
            queue.add(destination_offset);
            queue.add(source_offset);
            if copy_size > 65535 * 64 {
                queue.add_const(candle_wgpu_kernels::Constants::UseZ, true);
            }
            let pipeline = queue.get_pipeline_const_inplace(
                Pipelines::Copy(DType::U32, Functions::Copy),
                const_vec,
                inplaceble,
            );

            let bind_group = dev.create_bind_group_input1(buffer_dest, buffer_input, dtype.into());
            queue.enqueue_64_big(pipeline, bind_group, copy_size as u32);
        }
    }
    Ok(())
}

pub fn queue_copy2d(
    dev: &WgpuDevice,
    dest: (BufferReferenceId, u32, u32),
    input: (BufferReferenceId, u32, u32),
    dtype: crate::DType,
    d1: u32,
    d2: u32,
) -> crate::Result<()> {
    let (buffer_input, input_stride1, input_offset) = input;
    let (buffer_dest, dest_stride1, dest_offset) = dest;

    if d1 == 1 || (input_stride1 == d2 && input_stride1 == dest_stride1) {
        return queue_copy(
            dev,
            buffer_dest,
            buffer_input,
            dest_offset as usize,
            input_offset as usize,
            (d2 * d1) as usize,
            dtype,
        );
    }
    let const_vec = vec![input_offset == 0, dest_offset == 0];

    let mut queue = dev.get_queue();
    queue.add(d1);
    queue.add(d2);
    queue.add(input_stride1);
    queue.add(dest_stride1);
    if dest_offset != 0 || input_offset != 0 {
        queue.add(dest_offset);
    }
    if input_offset != 0 {
        queue.add(input_offset);
    }

    let bind_group = dev.create_bind_group_input1(buffer_dest, buffer_input, dtype.into());

    let x = d1.div_ceil(16);
    let y = d2.div_ceil(16);
    
    if y > crate::wgpu_backend::queue_buffer::MAX_DISPATCH_SIZE {
        queue.add_const(candle_wgpu_kernels::Constants::UseZ, true);

        let pipeline = queue.get_pipeline_const(
            Pipelines::Copy(dev.get_dtype(dtype)?, Functions::Copy2dTranspose),
            const_vec,
        );
        queue.enqueue_workgroups(
            pipeline,
            bind_group,
            y.min(65535),
            x,
            y.div_ceil(65535),
            (d1 * d2) as usize,
        );
    } else {
        if x > 65535 {
            queue.add_const(candle_wgpu_kernels::Constants::UseZ, true);
        }
        let pipeline = queue.get_pipeline_const(
            Pipelines::Copy(dev.get_dtype(dtype)?, Functions::Copy2d),
            const_vec,
        );
        queue.enqueue_workgroups(
            pipeline,
            bind_group,
            x.min(65535),
            y,
            x.div_ceil(65535),
            (d1 * d2) as usize,
        );
    }
    Ok(())
}

pub fn queue_copy3d(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input: BufferReferenceId,
    dtype: crate::DType,
    input_layout: &crate::Layout,
    input_shape: (u32, u32, u32), //b, m, k
    dest_layout: &crate::Layout,
) -> crate::Result<()> {
    let mut input1_stride = input_layout.stride().iter().rev();

    let input1_stride_1 = *input1_stride.next().unwrap_or(&1); //k
    let input1_stride_2 = *input1_stride.next().unwrap_or(&1); //m
    let input1_stride_3 = *input1_stride.next().unwrap_or(&1); //b

    let mut dest_stride = dest_layout.stride().iter().rev();
    let dest_stride_1 = *dest_stride.next().unwrap_or(&1);
    let dest_stride_2 = *dest_stride.next().unwrap_or(&1);
    let dest_stride_3 = *dest_stride.next().unwrap_or(&1);

    let const_vec = vec![
        input_layout.start_offset() == 0,
        (dest_stride_1 != 1),
        (dest_stride_2 != 1),
        (dest_stride_3 != 1),
        (input1_stride_1 != 1),
        (input1_stride_2 != 1),
        (input1_stride_3 != 1),
    ];

    let mut queue = dev.get_queue();
    queue.add(input_shape.2);
    queue.add(input_shape.1);
    queue.add(dest_stride_1);
    queue.add(dest_stride_2);
    queue.add(dest_stride_3);
    queue.add(input1_stride_1);
    queue.add(input1_stride_2);
    queue.add(input1_stride_3);
    if input_layout.start_offset() != 0 {
        queue.add(input_layout.start_offset());
    }

    let bind_group = dev.create_bind_group_input1(buffer_dest, buffer_input, dtype.into());

    let pipeline = queue.get_pipeline_const(
        Pipelines::Copy(dev.get_dtype(dtype)?, Functions::Copy3d),
        const_vec,
    );
    queue.enqueue_workgroups(
        pipeline,
        bind_group,
        input_shape.2.div_ceil(16_u32),
        input_shape.1.div_ceil(16_u32),
        input_shape.0,
        input_layout.shape().elem_count(),
    );
    Ok(())
}

pub fn queue_copy3d_padded(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    input: WgpuTensor,
    dtype: crate::DType,
    input_shape: (u32, u32, u32), //b, m, k
    dest_layout: &crate::Layout,
    _debug_info: Option<String>,
) -> crate::Result<()> {
    let mut input1_stride = input.layout().stride().iter().rev();

    let input1_stride_1 = *input1_stride.next().unwrap_or(&1); //k
    let input1_stride_2 = *input1_stride.next().unwrap_or(&1); //m
    let input1_stride_3 = *input1_stride.next().unwrap_or(&1); //b

    let mut dest_stride = dest_layout.stride().iter().rev();
    let dest_stride_1 = *dest_stride.next().unwrap_or(&1);
    let dest_stride_2 = *dest_stride.next().unwrap_or(&1);
    let dest_stride_3 = *dest_stride.next().unwrap_or(&1);

    let dest_shape = dest_layout.shape().dims3()?;

    let const_vec = vec![
        input.layout().start_offset() == 0,
        dest_stride_1 != 1,
        dest_stride_2 != 1,
        dest_stride_3 != 1,
        input1_stride_1 != 1,
        input1_stride_2 != 1,
        input1_stride_3 != 1,
    ];

    let mut queue = dev.get_queue();
    queue.add(input_shape.2);
    queue.add(input_shape.1);
    queue.add(dest_stride_1);
    queue.add(dest_stride_2);
    queue.add(dest_stride_3);
    queue.add(input1_stride_1);
    queue.add(input1_stride_2);
    queue.add(input1_stride_3);
    queue.add(dest_shape.2);
    queue.add(dest_shape.1);
    if input.layout().start_offset() != 0 {
        queue.add(input.layout().start_offset());
    }

    let bind_group = dev.create_bind_group_input1(buffer_dest, input.buffer(), dtype.into());
    let pipeline = if input_shape.0 == 1 {
        Functions::Copy3dPaddedNobatch
    } else {
        Functions::Copy3dPadded
    };
    let pipeline =
        queue.get_pipeline_const(Pipelines::Copy(dev.get_dtype(dtype)?, pipeline), const_vec);
    queue.enqueue_workgroups_extra(
        pipeline,
        bind_group,
        dest_shape.2.div_ceil(16) as u32,
        dest_shape.1.div_ceil(16) as u32,
        input_shape.0,
        input.layout().shape().elem_count(),
        #[cfg(feature = "wgpu_debug")]
        _debug_info,
    );
    Ok(())
}

pub fn queue_transpose3d(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input: BufferReferenceId,
    dtype: crate::DType,
    input_shape: (u32, u32, u32), //b, width, height
    start_offset: usize,
    batch_stride: usize,
) -> crate::Result<()> {
    let (batch, width, height) = input_shape;
    let mut queue = dev.get_queue();
    queue.add(width);
    queue.add(height);
    queue.add(start_offset);
    queue.add(batch_stride);

    let const_vec = vec![batch > 1, start_offset == 0];

    let bind_group = dev.create_bind_group_input1(buffer_dest, buffer_input, dtype.into());
    let pipeline = Functions::TransposeBatched;

    let pipeline =
        queue.get_pipeline_const(Pipelines::Copy(dev.get_dtype(dtype)?, pipeline), const_vec);

    queue.enqueue_workgroups(
        pipeline,
        bind_group,
        width.div_ceil(32),
        height.div_ceil(32),
        batch,
        (width * height * batch) as usize,
    );
    Ok(())
}

pub fn queue_copy4d_padded(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input: BufferReferenceId,
    dtype: crate::DType,
    input_layout: &crate::Layout,
    padding: usize,
    dest_layout: &crate::Layout,
) -> crate::Result<()> {
    let input1_stride = input_layout.stride();
    let dest_stride = dest_layout.stride();
    let input_shape = input_layout.shape().dims4()?;
    let dest_shape = dest_layout.shape().dims4()?;

    let const_vec = vec![
        (input_layout.start_offset() == 0) as usize,
        (dest_stride[3] != 1) as usize, //x (d1)
        (dest_stride[2] != 1) as usize, //y (d2)
        (dest_stride[1] != 1) as usize, //cin
        (dest_stride[0] != 1) as usize, //b
        (input1_stride[3] != 1) as usize,
        (input1_stride[2] != 1) as usize,
        (input1_stride[1] != 1) as usize,
        (input1_stride[0] != 1) as usize,
        input_shape.1, //channels
    ];

    let mut queue = dev.get_queue();
    queue.add(input_shape.3 + padding);
    queue.add(input_shape.2 + padding);
    queue.add(padding);
    queue.add(padding);

    queue.add(dest_stride[3]);
    queue.add(dest_stride[2]);
    queue.add(dest_stride[1]);
    queue.add(dest_stride[0]);
    queue.add(input1_stride[3]);
    queue.add(input1_stride[2]);
    queue.add(input1_stride[1]);
    queue.add(input1_stride[0]);
    queue.add(dest_shape.3);
    queue.add(dest_shape.2);

    if input_layout.start_offset() != 0 {
        queue.add(input_layout.start_offset());
    }

    let bind_group = dev.create_bind_group_input1(buffer_dest, buffer_input, dtype.into());

    let pipeline = Functions::Copy4dPadded;

    let pipeline =
        queue.get_pipeline_const(Pipelines::Copy(dev.get_dtype(dtype)?, pipeline), const_vec);
    queue.enqueue_workgroups(
        pipeline,
        bind_group,
        dest_shape.3.div_ceil(16) as u32,
        dest_shape.2.div_ceil(16) as u32,
        (input_shape.0 * input_shape.1) as u32,
        input_layout.shape().elem_count(),
    );
    Ok(())
}
