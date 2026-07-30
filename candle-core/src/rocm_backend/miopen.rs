use rocm_rs::miopen::tensor::{DataType, TensorDescriptor};
use rocm_rs::miopen::{ConvolutionDescriptor, Handle};

pub fn miopen_dtype<T: Copy>() -> DataType {
    if std::any::type_name::<T>().contains("f32") {
        DataType::MiopenFloat
    } else if std::any::type_name::<T>().contains("f64") {
        DataType::MiopenDouble
    } else if std::any::type_name::<T>().contains("f16") {
        DataType::MiopenHalf
    } else if std::any::type_name::<T>().contains("bf16") {
        DataType::MiopenBFloat16
    } else {
        panic!(
            "Unsupported dtype for MIOpen: {}",
            std::any::type_name::<T>()
        )
    }
}

pub fn conv2d_forward<T: Copy>(
    handle: &Handle,
    x_ptr: *mut std::ffi::c_void,
    w_ptr: *mut std::ffi::c_void,
    y_ptr: *mut std::ffi::c_void,
    b: usize,
    c_in: usize,
    c_out: usize,
    i_h: usize,
    i_w: usize,
    k_h: usize,
    k_w: usize,
    out_h: usize,
    out_w: usize,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
    dilation_h: usize,
    dilation_w: usize,
) -> crate::Result<()> {
    let x_desc = TensorDescriptor::new_4d(
        miopen_dtype::<T>(),
        b as i32,
        c_in as i32,
        i_h as i32,
        i_w as i32,
    )
    .map_err(|e| crate::Error::Msg(format!("MIOpen x_desc creation failed: {}", e)))?;

    let w_desc = TensorDescriptor::new_4d(
        miopen_dtype::<T>(),
        c_out as i32,
        c_in as i32,
        k_h as i32,
        k_w as i32,
    )
    .map_err(|e| crate::Error::Msg(format!("MIOpen w_desc creation failed: {}", e)))?;

    let y_desc = TensorDescriptor::new_4d(
        miopen_dtype::<T>(),
        b as i32,
        c_out as i32,
        out_h as i32,
        out_w as i32,
    )
    .map_err(|e| crate::Error::Msg(format!("MIOpen y_desc creation failed: {}", e)))?;

    let mut conv_desc = ConvolutionDescriptor::new()
        .map_err(|e| crate::Error::Msg(format!("MIOpen conv_desc creation failed: {}", e)))?;
    conv_desc
        .init_2d(
            0, // miopenConvolution
            pad_h as i32,
            pad_w as i32,
            stride_h as i32,
            stride_w as i32,
            dilation_h as i32,
            dilation_w as i32,
        )
        .map_err(|e| crate::Error::Msg(format!("MIOpen conv_desc init_2d failed: {}", e)))?;

    let workspace_size = rocm_rs::miopen::convolution::get_convolution_forward_workspace_size(
        handle, &w_desc, &x_desc, &conv_desc, &y_desc,
    )
    .map_err(|e| crate::Error::Msg(format!("MIOpen workspace size query failed: {}", e)))?;

    let alpha: [u8; 4] = 1.0f32.to_le_bytes();
    let beta: [u8; 4] = 0.0f32.to_le_bytes();

    let workspace = if workspace_size > 0 {
        Some(
            rocm_rs::hip::DeviceMemory::<u8>::new(workspace_size).map_err(|e| {
                crate::Error::Msg(format!("MIOpen workspace allocation failed: {}", e))
            })?,
        )
    } else {
        None
    };
    let workspace_ptr = workspace
        .as_ref()
        .map(|w| w.as_ptr() as *mut std::ffi::c_void)
        .unwrap_or(std::ptr::null_mut());

    unsafe {
        let (_, perf_results) = rocm_rs::miopen::convolution::find_convolution_forward_algorithm(
            handle,
            &x_desc,
            x_ptr,
            &w_desc,
            w_ptr,
            &conv_desc,
            &y_desc,
            y_ptr,
            1,
            workspace_ptr,
            workspace_size,
            false,
        )
        .map_err(|e| {
            crate::Error::Msg(format!(
                "MIOpen find_convolution_forward_algorithm failed: {}",
                e
            ))
        })?;

        let algo = perf_results
            .first()
            .map(|p| p.__bindgen_anon_1.fwd_algo)
            .unwrap_or(4);

        rocm_rs::miopen::convolution::convolution_forward(
            handle,
            &alpha,
            &x_desc,
            x_ptr,
            &w_desc,
            w_ptr,
            &conv_desc,
            algo,
            &beta,
            &y_desc,
            y_ptr,
            workspace_ptr,
            workspace_size,
        )
        .map_err(|e| crate::Error::Msg(format!("MIOpen convolution_forward failed: {}", e)))?;
    }

    Ok(())
}

pub fn conv_transpose1d_forward<T: Copy>(
    handle: &Handle,
    x_ptr: *mut std::ffi::c_void,
    w_ptr: *mut std::ffi::c_void,
    y_ptr: *mut std::ffi::c_void,
    b: usize,
    c_in: usize,
    c_out: usize,
    l_in: usize,
    k_size: usize,
    l_out: usize,
    padding: usize,
    output_padding: usize,
    stride: usize,
    dilation: usize,
) -> crate::Result<()> {
    let x_desc =
        TensorDescriptor::new_4d(miopen_dtype::<T>(), b as i32, c_in as i32, 1, l_in as i32)
            .map_err(|e| crate::Error::Msg(format!("MIOpen x_desc creation failed: {}", e)))?;

    let w_desc = TensorDescriptor::new_4d(
        miopen_dtype::<T>(),
        c_in as i32,
        c_out as i32,
        1,
        k_size as i32,
    )
    .map_err(|e| crate::Error::Msg(format!("MIOpen w_desc creation failed: {}", e)))?;

    let y_desc =
        TensorDescriptor::new_4d(miopen_dtype::<T>(), b as i32, c_out as i32, 1, l_out as i32)
            .map_err(|e| crate::Error::Msg(format!("MIOpen y_desc creation failed: {}", e)))?;

    let mut conv_desc = ConvolutionDescriptor::new()
        .map_err(|e| crate::Error::Msg(format!("MIOpen conv_desc creation failed: {}", e)))?;
    conv_desc
        .init_2d(0, padding as i32, 0, stride as i32, 1, dilation as i32, 1)
        .map_err(|e| crate::Error::Msg(format!("MIOpen conv_desc init_2d failed: {}", e)))?;
    conv_desc
        .set_transpose_conv_output_padding(output_padding as i32, 0)
        .map_err(|e| {
            crate::Error::Msg(format!(
                "MIOpen set_transpose_conv_output_padding failed: {}",
                e
            ))
        })?;

    let workspace_size =
        rocm_rs::miopen::convolution::get_convolution_backward_data_workspace_size(
            handle, &x_desc, &w_desc, &conv_desc, &y_desc,
        )
        .map_err(|e| {
            crate::Error::Msg(format!(
                "MIOpen backward data workspace size query failed: {}",
                e
            ))
        })?;

    let alpha: [u8; 4] = 1.0f32.to_le_bytes();
    let beta: [u8; 4] = 0.0f32.to_le_bytes();

    let workspace = if workspace_size > 0 {
        Some(
            rocm_rs::hip::DeviceMemory::<u8>::new(workspace_size).map_err(|e| {
                crate::Error::Msg(format!("MIOpen workspace allocation failed: {}", e))
            })?,
        )
    } else {
        None
    };
    let workspace_ptr = workspace
        .as_ref()
        .map(|w| w.as_ptr() as *mut std::ffi::c_void)
        .unwrap_or(std::ptr::null_mut());

    unsafe {
        let (_, perf_results) =
            rocm_rs::miopen::convolution::find_convolution_backward_data_algorithm(
                handle,
                &x_desc,
                x_ptr,
                &w_desc,
                w_ptr,
                &conv_desc,
                &y_desc,
                y_ptr,
                1,
                workspace_ptr,
                workspace_size,
                false,
            )
            .map_err(|e| {
                crate::Error::Msg(format!(
                    "MIOpen find_convolution_backward_data_algorithm failed: {}",
                    e
                ))
            })?;

        let algo = perf_results
            .first()
            .map(|p| p.__bindgen_anon_1.bwd_data_algo)
            .unwrap_or(0);

        rocm_rs::miopen::convolution::convolution_backward_data(
            handle,
            &alpha,
            &x_desc,
            x_ptr,
            &w_desc,
            w_ptr,
            &conv_desc,
            algo,
            &beta,
            &y_desc,
            y_ptr,
            workspace_ptr,
            workspace_size,
        )
        .map_err(|e| {
            crate::Error::Msg(format!(
                "MIOpen convolution_backward_data (transpose conv) failed: {}",
                e
            ))
        })?;
    }

    Ok(())
}
