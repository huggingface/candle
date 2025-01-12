use candle_wgpu_kernels::conv1d::Functions as Functions1d;
use candle_wgpu_kernels::conv2d::Functions;
use copy::queue_copy4d_padded;
use matmul::SGEMMParams;

use crate::Shape;

use super::*;

pub fn queue_conv2d(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    input: WgpuTensor,
    kernel: WgpuTensor,
    dtype: crate::DType,
    params: &crate::conv::ParamsConv2D,
) -> crate::Result<()> {
    //if input stride_x is not 1, performance can be extremly bad! -> copy strided
    let input_stride = input.layout().stride();
    let kernel_stride = kernel.layout().stride();

    //check if we might use a matrix multiplication instead of convolution:
    if params.k_h == 1
        && params.k_w == 1
        && input_stride[2] == input_stride[3] * params.i_w
        && params.padding == 0
        && params.dilation == 1
        && params.stride == 1
    {
        let m = params.c_out;
        let k = params.c_in;
        let n = params.i_h * params.i_w;

        let new_kernel_layout: Layout = Layout::new(
            Shape::from_dims(&[params.b_size, m, k]),
            vec![0, kernel_stride[0], kernel_stride[1]],
            kernel.layout().start_offset(),
        ); //batch kernel stride is 0, so we will reuse the same kernel for multiple batches
        let new_input_layout: Layout = Layout::new(
            Shape::from_dims(&[params.b_size, k, n]),
            vec![input_stride[0], input_stride[1], input_stride[3]],
            input.layout().start_offset(),
        );

        queue_matmul_buffer(
            dev,
            buffer_dest,
            WgpuTensor::new(&new_kernel_layout, kernel.buffer()),
            WgpuTensor::new(&new_input_layout, input.buffer()),
            SGEMMParams::new(params.b_size, m, k, n),
            dtype,
        )?;

        return Ok(());
    }

    //kernel is contiguous in k_h, k_w, c_in -> we might use im2col:
    //this is way faster, but also needs way more memory:
    if kernel_stride[2] == params.k_w && kernel_stride[1] == params.k_h * params.k_w {
        let mem_needed = 4
            * params.c_in
            * params.k_h
            * params.k_w
            * params.b_size
            * params.out_h()
            * params.out_w();
        //for small c_in, k_h, k_w,  matmul k will be small (e.g. 9)
        //for small c_out            matmul m will be small (e.g. 1)
        //in this case only a relativ slowly naive matmul impl will be used.
        //it may be faster to just use the conv2d shader directly instead of using im2col as this conversion will not result in a fast matrix multipliation.

        let m = params.c_out;
        let k = params.c_in * params.k_h * params.k_w;
        if (k >= 64 || m >= 16)
            && mem_needed < dev.device_limits.max_storage_buffer_binding_size as usize
        {
            return queue_conv2d_matmul(dev, buffer_dest, input, kernel, dtype, params);
        }
    }

    let mut use_padded = false;

    const MAY_PAD_INPUT: bool = false;

    let is_continues_in_c_in = input_stride[1] == 1;

    let (input_buffer, input_layout) = if MAY_PAD_INPUT && params.padding > 0 {
        use_padded = true;
        let current_shape = input.layout().shape().dims4()?;
        let padded_shape = (
            current_shape.0,
            current_shape.1,
            current_shape.2 + params.padding * 2,
            current_shape.3 + params.padding * 2,
        );
        let new_layout = Layout::contiguous_with_offset(Shape::from(padded_shape), 0);

        let mut cache = dev.cache.lock().unwrap();
        let tmp_buffer = cache.create_buffer_reference(
            new_layout.shape().elem_count() * dtype.size_in_bytes(),
            false,
        );
        drop(cache);
        queue_copy4d_padded(
            dev,
            tmp_buffer,
            input.buffer(),
            dtype,
            input.layout(),
            params.padding,
            &new_layout,
        )?;

        (tmp_buffer, new_layout)
    } else {
        //the performance is bad if the input is not contiguous
        if input_stride[3] != 1 && (params.c_out > 32) && (params.i_h >= 64 && params.i_w >= 64) {
            let mut cache = dev.cache.lock().unwrap();
            let tmp_buffer = cache.create_buffer_reference(
                input.layout().shape().elem_count() * dtype.size_in_bytes(),
                false,
            );

            queue_copy_strided(dev, tmp_buffer, input.buffer(), dtype, input.layout(), 0)?;
            (tmp_buffer, Layout::contiguous(input.layout().shape()))
        } else {
            (input.buffer(), input.layout().clone())
        }
    };
    let padding = if use_padded { 0 } else { params.padding };

    let input_stride = input_layout.stride();
    let kernel_stride = kernel.layout().stride();

    let mut meta = get_queue(dev);

    let const_vec = vec![
        kernel_stride[3], //kernel_x_stride
        input_stride[3],  //stride_x_in
        params.dilation,
        params.k_w,
        params.k_h,
        params.b_size,
        params.c_in,
        params.i_w,
        params.i_h,
    ];

    meta.add(input_layout.start_offset());
    meta.add(kernel_stride[2]); //kernel_y_stride
    meta.add(kernel_stride[1]); //kernel_c_stride
    meta.add(kernel_stride[0]); //kernel_b_stride
    meta.add(kernel.layout().start_offset());
    meta.add(params.i_w); //size_in_x
    meta.add(params.i_h); //size_in_y
    meta.add(params.out_w() * params.out_h() * params.c_out); //Stride_batch_out
    meta.add(params.out_w() * params.out_h()); //stride_c_out
    meta.add(params.out_w()); //stride_y_out
    meta.add(params.out_h()); //size_y_out

    meta.add(input_stride[0]); //stride_batch_input
    meta.add(input_stride[1]); //stride_c_in
    meta.add(input_stride[2]); //stride_y_in
    meta.add(padding);
    meta.add(params.stride);
    meta.add(params.c_out);

    let pipeline_function = if is_continues_in_c_in && params.c_in >= 64 {
        if padding == 0 {
            Functions::Conv2dLongchannelNopadding
        } else {
            Functions::Conv2dLongchannel
        }
    } else if params.k_h == 1 && params.k_w == 1 {
        if padding == 0 {
            Functions::Conv2dKernelSize1Nopadding
        } else {
            Functions::Conv2dKernelSize1
        }
    } else if padding == 0 {
        Functions::Conv2dNopadding
    } else {
        Functions::Conv2d
    };

    let pipeline = meta.get_pipeline_const(
        Pipelines::Conv2d(get_dtype(dtype)?, pipeline_function),
        const_vec,
    );

    let bind_group =
        create_bind_group_input2(buffer_dest, input_buffer, kernel.buffer(), dtype.into());

    enqueue_workgroups_extra(
        meta,
        pipeline,
        bind_group,
        (params.out_w() as u32 + 15) / 16,
        (params.out_h() as u32 + 15) / 16,
        (params.c_out * params.b_size) as u32,
        params.out_w()
            * params.out_h()
            * params.c_out
            * params.b_size
            * kernel.layout().shape().elem_count(),
        #[cfg(feature = "wgpu_debug")]
        Some(format!(
            "{:?}, input1: ({:?}, {:?}), kernel: ({:?}, {:?})",
            params,
            input_layout.shape(),
            input_layout.stride(),
            kernel.layout().shape(),
            kernel.layout().stride()
        )),
    );

    Ok(())
}

//calculated conv2d(uses im2col and matmul)
//+ fast(matmul)
//-im2col creates much more memory
pub fn queue_conv2d_matmul(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    input: WgpuTensor,
    kernel: WgpuTensor,
    dtype: crate::DType,
    params: &crate::conv::ParamsConv2D,
) -> crate::Result<()> {
    //1. im2col
    // Calculate output dimensions
    let o_h = params.out_h();
    let o_w = params.out_w();

    // Get strides from the layouts
    let src_stride = input.layout().stride();
    let kernel_stride = kernel.layout().stride();

    if kernel_stride[2] != params.k_w || kernel_stride[1] != params.k_h * params.k_w {
        panic!("kernel is not contiguous in c_in, k_h, k_w")
    }

    let dst_numel = params.k_h * params.k_w * params.b_size * params.c_in * o_h * o_w;

    let const_vec = vec![
        params.padding,
        params.stride,
        params.dilation,
        params.k_h,
        params.k_w,
        (input.layout().start_offset() == 0) as usize,
    ];

    let mut meta = get_queue(dev);
    meta.add(dst_numel); // op_conv2d_dst_numel
    meta.add(o_h); // op_conv2d_h_out
    meta.add(o_w); // op_conv2d_w_out
    meta.add(params.c_in); // op_conv2d_c_in
    meta.add(params.i_h); // op_conv2d_h_in
    meta.add(params.i_w); // op_conv2d_w_in
    meta.add(src_stride[0] as u32); // op_conv2d_src_s0 (batch stride)
    meta.add(src_stride[1] as u32); // op_conv2d_src_s1 (channel stride)
    meta.add(src_stride[2] as u32); // op_conv2d_src_s2 (height stride)
    meta.add(src_stride[3] as u32); // op_conv2d_src_s3 (width stride)
    meta.add(input.layout().start_offset()); // op_conv2d_src_s3 (width stride)

    // Dispatch the convolution kernel
    let workgroup_size = 256; // Assumed workgroup size, adjust based on hardware
    let num_workgroups = dst_numel.div_ceil(workgroup_size);

    let b = params.b_size;
    let n = o_h * o_w;
    let m: usize = params.c_out;
    let k = params.c_in * params.k_h * params.k_w;
    let im2col_layout = Layout::new(Shape::from_dims(&[b, k, n]), vec![k * n, n, 1], 0);

    let im2col_buffer;
    let pipeline = meta.get_pipeline_const(
        Pipelines::Conv2d(get_dtype(dtype)?, Functions::Im2col),
        const_vec,
    );
    {
        let mut cache = dev.cache.lock().unwrap();

        im2col_buffer = cache.create_buffer_reference(n * k * b * dtype.size_in_bytes(), false);

        let bind_group = create_bind_group_input1(im2col_buffer, input.buffer(), dtype.into());

        let x = num_workgroups.min(65535);
        let y = (num_workgroups + 65534) / 65535;

        enqueue_workgroups_extra(
            meta,
            pipeline,
            bind_group,
            x as u32,
            y as u32,
            1,
            dst_numel,
            #[cfg(feature = "wgpu_debug")]
            Some(format!(
                "{:?}, input1: ({:?}, {:?}), kernel: ({:?}, {:?})",
                params,
                input.layout().shape(),
                input.layout().stride(),
                kernel.layout().shape(),
                kernel.layout().stride(),
            )),
        );
    }

    let flattened_kernel_layout = Layout::new(
        Shape::from_dims(&[1, params.c_out, params.k_h * params.k_w * params.c_in]),
        vec![0, kernel_stride[0], kernel_stride[3]],
        kernel.layout().start_offset(),
    );
    queue_matmul_buffer(
        dev,
        buffer_dest, // The final output buffer
        WgpuTensor::new(&flattened_kernel_layout, kernel.buffer()),
        WgpuTensor::new(&im2col_layout, im2col_buffer),
        SGEMMParams::new(params.b_size, m, k, n),
        dtype,
    )?;

    Ok(())
}

pub fn queue_conv2d_transpose(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    input: WgpuTensor,
    kernel: WgpuTensor,
    dtype: crate::DType,
    params: &crate::conv::ParamsConvTranspose2D,
) -> crate::Result<()> {
    let input_stride = input.layout().stride();
    let kernel_stride = kernel.layout().stride();

    let mut meta = get_queue(dev);

    let const_vec = vec![
        kernel_stride[3], //kernel_x_stride
        input_stride[3],  //stride_x_in
        params.dilation,
        params.k_w,
        params.k_h,
        params.b_size,
        params.c_in,
        params.i_w,
        params.i_h,
    ];

    meta.add(input.layout().start_offset());
    meta.add(kernel_stride[2]); //kernel_y_stride
    meta.add(kernel_stride[0]); //kernel_c_stride
    meta.add(kernel_stride[1]); //kernel_b_stride
    meta.add(kernel.layout().start_offset());
    meta.add(params.i_w); //size_in_x
    meta.add(params.i_h); //size_in_y
    meta.add(params.out_w() * params.out_h() * params.c_out); //Stride_batch_out
    meta.add(params.out_w() * params.out_h()); //stride_c_out
    meta.add(params.out_w()); //stride_y_out
    meta.add(params.out_h()); //size_y_out

    meta.add(input_stride[0]); //stride_batch_input
    meta.add(input_stride[1]); //stride_c_in
    meta.add(input_stride[2]); //stride_y_in

    meta.add(params.padding);
    meta.add(params.stride);

    let pipeline = meta.get_pipeline_const(
        Pipelines::Conv2d(get_dtype(dtype)?, Functions::Conv2dTranspose),
        const_vec,
    );
    let bind_group =
        create_bind_group_input2(buffer_dest, input.buffer(), kernel.buffer(), dtype.into());
    enqueue_workgroups(
        meta,
        pipeline,
        bind_group,
        ((params.out_w() - params.output_padding) as u32 + 15) / 16,
        ((params.out_h() - params.output_padding) as u32 + 15) / 16,
        params.c_out as u32,
        params.out_w()
            * params.out_h()
            * params.c_out
            * params.b_size
            * kernel.layout().shape().elem_count(),
    );
    Ok(())
}

pub fn queue_conv1d(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    input: WgpuTensor,
    kernel: WgpuTensor,
    dtype: crate::DType,
    params: &crate::conv::ParamsConv1D,
) -> crate::Result<()> {
    let input_stride = input.layout().stride();
    let kernel_stride = kernel.layout().stride();

    let const_vec = vec![
        kernel_stride[2], //kernel_x_stride
        input_stride[2],  //stride_x_in
        params.padding,
        params.stride,
        params.dilation,
        input.layout().start_offset(),
        params.k_size,
        params.b_size,
        params.c_in,
    ];
    let mut meta = get_queue(dev);

    meta.add(kernel_stride[1]); //kernel_c_stride
    meta.add(kernel_stride[0]); //kernel_b_stride
    meta.add(kernel.layout().start_offset());
    meta.add(params.l_in); //size_in_x
    meta.add(params.l_out() * params.c_out); //Stride_batch_out
    meta.add(params.l_out()); //stride_c_out
    meta.add(params.l_out()); //size_y_out

    meta.add(input_stride[0]); //stride_batch_input
    meta.add(input_stride[1]); //stride_c_in

    let pipeline = meta.get_pipeline_const(
        Pipelines::Conv1d(get_dtype(dtype)?, Functions1d::Conv1d),
        const_vec,
    );

    let bind_group =
        create_bind_group_input2(buffer_dest, input.buffer(), kernel.buffer(), dtype.into());
    enqueue_workgroups(
        meta,
        pipeline,
        bind_group,
        (params.l_out() as u32 + 63) / 64,
        params.c_out as u32,
        1,
        params.l_out() * params.c_out * params.b_size * kernel.layout().shape().elem_count(),
    );
    Ok(())
}

pub fn queue_conv1d_transpose(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    input: WgpuTensor,
    kernel: WgpuTensor,
    dtype: crate::DType,
    params: &crate::conv::ParamsConvTranspose1D,
) -> crate::Result<()> {
    let input_stride = input.layout().stride();
    let kernel_stride = kernel.layout().stride();

    let const_vec = vec![
        kernel_stride[2], //kernel_x_stride
        input_stride[2],  //stride_x_in
        params.padding,
        params.stride,
        params.dilation,
        input.layout().start_offset(),
        params.k_size,
        params.b_size,
        params.c_in,
    ];
    let mut meta = get_queue(dev);
    meta.add(kernel_stride[0]); //kernel_c_stride
    meta.add(kernel_stride[1]); //kernel_b_stride
    meta.add(kernel.layout().start_offset());
    meta.add(params.l_in); //size_in_x
    meta.add(params.l_out() * params.c_out); //Stride_batch_out
    meta.add(params.l_out()); //stride_c_out
    meta.add(params.l_out()); //size_y_out

    meta.add(input_stride[0]); //stride_batch_input
    meta.add(input_stride[1]); //stride_c_in

    let pipeline = meta.get_pipeline_const(
        Pipelines::Conv1d(get_dtype(dtype)?, Functions1d::Conv1dTranspose),
        const_vec,
    );
    let bind_group =
        create_bind_group_input2(buffer_dest, input.buffer(), kernel.buffer(), dtype.into());
    enqueue_workgroups(
        meta,
        pipeline,
        bind_group,
        ((params.l_out() - params.output_padding) as u32 + 63) / 64,
        params.c_out as u32,
        1u32,
        params.l_out() * params.c_out * params.b_size * kernel.layout().shape().elem_count(),
    );
    Ok(())
}
