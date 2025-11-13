//! MIOpen integration for ROCm backend
//!
//! This module provides convolution and pooling operations using AMD's MIOpen library.
//! Matches cuda_backend/cudnn.rs pattern.

use crate::backend::BackendStorage;
use crate::rocm_backend::{RocmDevice, RocmError, RocmStorageSlice as S};
use crate::{Layout, Result};
use half::f16;
use rocm_rs::miopen::{ConvolutionDescriptor, Handle, PoolingDescriptor, TensorDescriptor};
use std::os::raw::c_void;

/// Helper function for 2D pooling operations (average and max)
pub(crate) fn pool2d(
    storage: &crate::rocm_backend::RocmStorage,
    layout: &Layout,
    k: (usize, usize),
    stride: (usize, usize),
    mode: rocm_rs::miopen::ffi::miopenPoolingMode_t,
) -> Result<crate::rocm_backend::RocmStorage> {
    // Get input dimensions (assuming NCHW format)
    let shape = layout.shape();
    if shape.rank() != 4 {
        return Err(RocmError::InternalError("pool2d requires 4D tensor (NCHW)").into());
    }

    let (n, c, h, w) = (shape.dims()[0], shape.dims()[1], shape.dims()[2], shape.dims()[3]);

    // Calculate output dimensions
    let out_h = (h - k.1) / stride.1 + 1;
    let out_w = (w - k.0) / stride.0 + 1;
    let out_size = n * c * out_h * out_w;

    // Create MIOpen handle
    let handle = Handle::new().map_err(|e| {
        RocmError::InternalError(&format!("MIOpen handle creation failed: {:?}", e))
    })?;

    // Create pooling descriptor
    let mut pool_desc = PoolingDescriptor::new().map_err(|e| {
        RocmError::InternalError(&format!("Pooling descriptor creation failed: {:?}", e))
    })?;
    pool_desc
        .set_2d(mode, k.1 as i32, k.0 as i32, 0, 0, stride.1 as i32, stride.0 as i32)
        .map_err(|e| {
            RocmError::InternalError(&format!("Pooling descriptor set failed: {:?}", e))
        })?;

    // Create input tensor descriptor
    let mut input_desc = TensorDescriptor::new().map_err(|e| {
        RocmError::InternalError(&format!("Tensor descriptor creation failed: {:?}", e))
    })?;
    let data_type = match storage.dtype() {
        crate::DType::F32 => rocm_rs::miopen::ffi::miopenDataType_t_miopenFloat,
        crate::DType::F16 => rocm_rs::miopen::ffi::miopenDataType_t_miopenHalf,
        _ => return Err(RocmError::InternalError("pool2d only supports f32 and f16").into()),
    };
    input_desc.set_4d(data_type, n as i32, c as i32, h as i32, w as i32).map_err(|e| {
        RocmError::InternalError(&format!("Input tensor descriptor set failed: {:?}", e))
    })?;

    // Create output tensor descriptor
    let mut output_desc = TensorDescriptor::new().map_err(|e| {
        RocmError::InternalError(&format!("Output tensor descriptor creation failed: {:?}", e))
    })?;
    output_desc.set_4d(data_type, n as i32, c as i32, out_h as i32, out_w as i32).map_err(|e| {
        RocmError::InternalError(&format!("Output tensor descriptor set failed: {:?}", e))
    })?;

    // Get workspace size
    let workspace_size = pool_desc
        .get_workspace_size(&output_desc)
        .map_err(|e| RocmError::InternalError(&format!("Workspace size query failed: {:?}", e)))?;

    // Allocate workspace if needed
    let workspace = if workspace_size > 0 {
        unsafe { storage.device().hip_device().alloc::<u8>(workspace_size)? }
    } else {
        unsafe { storage.device().hip_device().alloc::<u8>(0)? }
    };

    // Perform pooling based on dtype
    let slice = match &storage.slice {
        S::F32(input) => {
            let input_slice = &input.slice(layout.start_offset()..);
            let mut output = unsafe { storage.device().hip_device().alloc::<f32>(out_size)? };

            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;

            unsafe {
                pool_desc
                    .forward(
                        &handle,
                        &alpha.to_ne_bytes(),
                        &input_desc,
                        input_slice.as_ptr() as *const c_void,
                        &beta.to_ne_bytes(),
                        &output_desc,
                        output.as_mut_ptr() as *mut c_void,
                        false, // do_backward
                        workspace.as_mut_ptr() as *mut c_void,
                        workspace_size,
                    )
                    .map_err(|e| {
                        RocmError::InternalError(&format!("Pooling forward failed: {:?}", e))
                    })?;
            }

            S::F32(output)
        }
        S::F16(input) => {
            let input_slice = &input.slice(layout.start_offset()..);
            let mut output = unsafe { storage.device().hip_device().alloc::<f16>(out_size)? };

            let alpha = f16::ONE;
            let beta = f16::ZERO;

            unsafe {
                pool_desc
                    .forward(
                        &handle,
                        &alpha.to_ne_bytes(),
                        &input_desc,
                        input_slice.as_ptr() as *const c_void,
                        &beta.to_ne_bytes(),
                        &output_desc,
                        output.as_mut_ptr() as *mut c_void,
                        false,
                        workspace.as_mut_ptr() as *mut c_void,
                        workspace_size,
                    )
                    .map_err(|e| {
                        RocmError::InternalError(&format!("Pooling forward failed: {:?}", e))
                    })?;
            }

            S::F16(output)
        }
        _ => return Err(RocmError::InternalError("pool2d only supports f32 and f16").into()),
    };

    Ok(crate::rocm_backend::RocmStorage { slice, device: storage.device().clone() })
}

/// 2D convolution using MIOpen
/// Matches cuda_backend/cudnn.rs::launch_conv2d pattern
pub(crate) fn conv2d(
    storage: &crate::rocm_backend::RocmStorage,
    inp_l: &Layout,
    kernel: &crate::rocm_backend::RocmStorage,
    kernel_l: &Layout,
    params: &crate::conv::ParamsConv2D,
) -> Result<crate::rocm_backend::RocmStorage> {
    let device = storage.device().clone();

    // Calculate output dimensions
    let (out_w, out_h) = (params.out_w(), params.out_h());
    let dst_el = params.c_out * out_w * out_h * params.b_size;

    // Get input shape (NCHW format)
    let inp_shape = inp_l.shape();
    if inp_shape.rank() != 4 {
        return Err(RocmError::InternalError("conv2d requires 4D input tensor (NCHW)").into());
    }
    let (n, c_in, h_in, w_in) =
        (inp_shape.dims()[0], inp_shape.dims()[1], inp_shape.dims()[2], inp_shape.dims()[3]);

    // Get kernel shape (NCHW format: out_channels, in_channels, kh, kw)
    let kernel_shape = kernel_l.shape();
    if kernel_shape.rank() != 4 {
        return Err(RocmError::InternalError("conv2d requires 4D kernel tensor").into());
    }

    let slice = match (&storage.slice, &kernel.slice) {
        (S::F32(inp), S::F32(k)) => {
            let inp_slice = &inp.slice(inp_l.start_offset()..);
            let k_slice = &k.slice(kernel_l.start_offset()..);
            let mut out = unsafe { device.hip_device().alloc::<f32>(dst_el)? };

            // Create MIOpen handle
            let handle = Handle::new().map_err(|e| {
                RocmError::InternalError(&format!("MIOpen handle creation failed: {:?}", e))
            })?;

            // Create input tensor descriptor
            let mut input_desc = TensorDescriptor::new().map_err(|e| {
                RocmError::InternalError(&format!(
                    "Input tensor descriptor creation failed: {:?}",
                    e
                ))
            })?;
            input_desc
                .set_4d(
                    rocm_rs::miopen::ffi::miopenDataType_t_miopenFloat,
                    n as i32,
                    c_in as i32,
                    h_in as i32,
                    w_in as i32,
                )
                .map_err(|e| {
                    RocmError::InternalError(&format!(
                        "Input tensor descriptor set failed: {:?}",
                        e
                    ))
                })?;

            // Create filter descriptor
            let mut filter_desc = TensorDescriptor::new().map_err(|e| {
                RocmError::InternalError(&format!("Filter descriptor creation failed: {:?}", e))
            })?;
            filter_desc
                .set_4d(
                    rocm_rs::miopen::ffi::miopenDataType_t_miopenFloat,
                    params.c_out as i32,
                    c_in as i32,
                    params.k_h as i32,
                    params.k_w as i32,
                )
                .map_err(|e| {
                    RocmError::InternalError(&format!("Filter descriptor set failed: {:?}", e))
                })?;

            // Create convolution descriptor
            let mut conv_desc = ConvolutionDescriptor::new().map_err(|e| {
                RocmError::InternalError(&format!(
                    "Convolution descriptor creation failed: {:?}",
                    e
                ))
            })?;
            conv_desc
                .init_2d(
                    rocm_rs::miopen::ffi::miopenConvolutionMode_t_miopenConvolution,
                    params.padding as i32,
                    params.padding as i32,
                    params.stride as i32,
                    params.stride as i32,
                    params.dilation as i32,
                    params.dilation as i32,
                )
                .map_err(|e| {
                    RocmError::InternalError(&format!(
                        "Convolution descriptor init failed: {:?}",
                        e
                    ))
                })?;

            // Create output tensor descriptor
            let mut output_desc = TensorDescriptor::new().map_err(|e| {
                RocmError::InternalError(&format!(
                    "Output tensor descriptor creation failed: {:?}",
                    e
                ))
            })?;
            output_desc
                .set_4d(
                    rocm_rs::miopen::ffi::miopenDataType_t_miopenFloat,
                    n as i32,
                    params.c_out as i32,
                    out_h as i32,
                    out_w as i32,
                )
                .map_err(|e| {
                    RocmError::InternalError(&format!(
                        "Output tensor descriptor set failed: {:?}",
                        e
                    ))
                })?;

            // Get workspace size
            let workspace_size =
                rocm_rs::miopen::convolution::get_convolution_forward_workspace_size(
                    &handle,
                    &filter_desc,
                    &input_desc,
                    &conv_desc,
                    &output_desc,
                )
                .map_err(|e| {
                    RocmError::InternalError(&format!("Workspace size query failed: {:?}", e))
                })?;

            // Allocate workspace
            let workspace = if workspace_size > 0 {
                unsafe { device.hip_device().alloc::<u8>(workspace_size)? }
            } else {
                unsafe { device.hip_device().alloc::<u8>(0)? }
            };

            // Find best algorithm
            let (_, perf_results) = unsafe {
                rocm_rs::miopen::convolution::find_convolution_forward_algorithm(
                    &handle,
                    &input_desc,
                    inp_slice.as_ptr() as *const c_void,
                    &filter_desc,
                    k_slice.as_ptr() as *const c_void,
                    &conv_desc,
                    &output_desc,
                    out.as_mut_ptr() as *mut c_void,
                    1, // request_algo_count
                    workspace.as_mut_ptr() as *mut c_void,
                    workspace_size,
                    false, // exhaustive_search
                )
                .map_err(|e| {
                    RocmError::InternalError(&format!("Algorithm search failed: {:?}", e))
                })?
            };

            let algo = if !perf_results.is_empty() {
                perf_results[0].fwd_algo
            } else {
                return Err(RocmError::InternalError("No convolution algorithm found").into());
            };

            // Run convolution
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;

            unsafe {
                rocm_rs::miopen::convolution::convolution_forward(
                    &handle,
                    &alpha.to_ne_bytes(),
                    &input_desc,
                    inp_slice.as_ptr() as *const c_void,
                    &filter_desc,
                    k_slice.as_ptr() as *const c_void,
                    &conv_desc,
                    algo,
                    &beta.to_ne_bytes(),
                    &output_desc,
                    out.as_mut_ptr() as *mut c_void,
                    workspace.as_mut_ptr() as *mut c_void,
                    workspace_size,
                )
                .map_err(|e| {
                    RocmError::InternalError(&format!("Convolution forward failed: {:?}", e))
                })?;
            }

            S::F32(out)
        }
        (S::F16(inp), S::F16(k)) => {
            let inp_slice = &inp.slice(inp_l.start_offset()..);
            let k_slice = &k.slice(kernel_l.start_offset()..);
            let mut out = unsafe { device.hip_device().alloc::<f16>(dst_el)? };

            let handle = Handle::new().map_err(|e| {
                RocmError::InternalError(&format!("MIOpen handle creation failed: {:?}", e))
            })?;

            let mut input_desc = TensorDescriptor::new().map_err(|e| {
                RocmError::InternalError(&format!(
                    "Input tensor descriptor creation failed: {:?}",
                    e
                ))
            })?;
            input_desc
                .set_4d(
                    rocm_rs::miopen::ffi::miopenDataType_t_miopenHalf,
                    n as i32,
                    c_in as i32,
                    h_in as i32,
                    w_in as i32,
                )
                .map_err(|e| {
                    RocmError::InternalError(&format!(
                        "Input tensor descriptor set failed: {:?}",
                        e
                    ))
                })?;

            let mut filter_desc = TensorDescriptor::new().map_err(|e| {
                RocmError::InternalError(&format!("Filter descriptor creation failed: {:?}", e))
            })?;
            filter_desc
                .set_4d(
                    rocm_rs::miopen::ffi::miopenDataType_t_miopenHalf,
                    params.c_out as i32,
                    c_in as i32,
                    params.k_h as i32,
                    params.k_w as i32,
                )
                .map_err(|e| {
                    RocmError::InternalError(&format!("Filter descriptor set failed: {:?}", e))
                })?;

            let mut conv_desc = ConvolutionDescriptor::new().map_err(|e| {
                RocmError::InternalError(&format!(
                    "Convolution descriptor creation failed: {:?}",
                    e
                ))
            })?;
            conv_desc
                .init_2d(
                    rocm_rs::miopen::ffi::miopenConvolutionMode_t_miopenConvolution,
                    params.padding as i32,
                    params.padding as i32,
                    params.stride as i32,
                    params.stride as i32,
                    params.dilation as i32,
                    params.dilation as i32,
                )
                .map_err(|e| {
                    RocmError::InternalError(&format!(
                        "Convolution descriptor init failed: {:?}",
                        e
                    ))
                })?;

            let mut output_desc = TensorDescriptor::new().map_err(|e| {
                RocmError::InternalError(&format!(
                    "Output tensor descriptor creation failed: {:?}",
                    e
                ))
            })?;
            output_desc
                .set_4d(
                    rocm_rs::miopen::ffi::miopenDataType_t_miopenHalf,
                    n as i32,
                    params.c_out as i32,
                    out_h as i32,
                    out_w as i32,
                )
                .map_err(|e| {
                    RocmError::InternalError(&format!(
                        "Output tensor descriptor set failed: {:?}",
                        e
                    ))
                })?;

            let workspace_size =
                rocm_rs::miopen::convolution::get_convolution_forward_workspace_size(
                    &handle,
                    &filter_desc,
                    &input_desc,
                    &conv_desc,
                    &output_desc,
                )
                .map_err(|e| {
                    RocmError::InternalError(&format!("Workspace size query failed: {:?}", e))
                })?;

            let workspace = if workspace_size > 0 {
                unsafe { device.hip_device().alloc::<u8>(workspace_size)? }
            } else {
                unsafe { device.hip_device().alloc::<u8>(0)? }
            };

            let (_, perf_results) = unsafe {
                rocm_rs::miopen::convolution::find_convolution_forward_algorithm(
                    &handle,
                    &input_desc,
                    inp_slice.as_ptr() as *const c_void,
                    &filter_desc,
                    k_slice.as_ptr() as *const c_void,
                    &conv_desc,
                    &output_desc,
                    out.as_mut_ptr() as *mut c_void,
                    1,
                    workspace.as_mut_ptr() as *mut c_void,
                    workspace_size,
                    false,
                )
                .map_err(|e| {
                    RocmError::InternalError(&format!("Algorithm search failed: {:?}", e))
                })?
            };

            let algo = if !perf_results.is_empty() {
                perf_results[0].fwd_algo
            } else {
                return Err(RocmError::InternalError("No convolution algorithm found").into());
            };

            let alpha = f16::ONE;
            let beta = f16::ZERO;

            unsafe {
                rocm_rs::miopen::convolution::convolution_forward(
                    &handle,
                    &alpha.to_ne_bytes(),
                    &input_desc,
                    inp_slice.as_ptr() as *const c_void,
                    &filter_desc,
                    k_slice.as_ptr() as *const c_void,
                    &conv_desc,
                    algo,
                    &beta.to_ne_bytes(),
                    &output_desc,
                    out.as_mut_ptr() as *mut c_void,
                    workspace.as_mut_ptr() as *mut c_void,
                    workspace_size,
                )
                .map_err(|e| {
                    RocmError::InternalError(&format!("Convolution forward failed: {:?}", e))
                })?;
            }

            S::F16(out)
        }
        _ => return Err(RocmError::InternalError("conv2d only supports f32 and f16").into()),
    };

    Ok(crate::rocm_backend::RocmStorage { slice, device })
}
