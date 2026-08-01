//! `BackendStorage::to_dtype`, i.e. the `cast.cu` launchers.
//!
//! Split out of `mod.rs`, which is far over the workspace 400-line cap.

use float8::F8E4M3;
use half::{bf16, f16};

use super::launch::{launch_config_layout, launch_kernel};
use super::params::dims_and_strides;
use super::{kernels, rocm_error, RocmStorage, RocmStorageSlice};
use crate::{DType, Layout, Result};

/// How `cast.cu` spells `dtype` in a kernel name.
///
/// Only F8E4M3 differs from [`DType::as_str`], which yields `f8e4m3` where the
/// kernels are `cast_f8_e4m3_f32`, `cast_bf16_f8_e4m3` and so on.
fn cast_dtype_str(dtype: DType) -> &'static str {
    match dtype {
        DType::F8E4M3 => "f8_e4m3",
        other => other.as_str(),
    }
}

macro_rules! cast_launch {
    ($dev:expr, $grid:expr, $block:expr, $el:expr, $dims_len:expr, $ds_ptr:expr, $src_ptr:expr, $src_dtype:expr, $rust_type:ty, $variant:ident) => {{
        cast_launch!(
            $dev,
            $grid,
            $block,
            $el,
            $dims_len,
            $ds_ptr,
            $src_ptr,
            $src_dtype,
            $rust_type,
            $variant,
            stringify!($rust_type)
        )
    }};
    ($dev:expr, $grid:expr, $block:expr, $el:expr, $dims_len:expr, $ds_ptr:expr, $src_ptr:expr, $src_dtype:expr, $rust_type:ty, $variant:ident, $dst_name:expr) => {{
        let out = $dev.alloc::<$rust_type>($el)?;
        let out_ptr = out.as_ptr();
        let func_name = format!("cast_{}_{}", cast_dtype_str($src_dtype), $dst_name);
        unsafe {
            launch_kernel(
                &$dev,
                &kernels::CAST,
                &func_name,
                $grid,
                $block,
                &mut [
                    &$el as *const usize as *mut std::ffi::c_void,
                    &$dims_len as *const usize as *mut std::ffi::c_void,
                    (&$ds_ptr) as *const *const usize as *mut std::ffi::c_void,
                    (&$src_ptr) as *const *const std::ffi::c_void as *mut std::ffi::c_void,
                    (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                ],
            )?;
        }
        RocmStorageSlice::$variant(out)
    }};
}

pub(super) fn to_dtype(
    storage: &RocmStorage,
    layout: &Layout,
    dtype: DType,
) -> Result<RocmStorage> {
    let shape = layout.shape();
    let dims = shape.dims();
    let el = shape.elem_count();
    let dev = storage.device.clone();

    let ds = dims_and_strides(&dev, layout)?;
    let start_o = layout.start_offset();
    // SAFETY: `start_offset` is an in-bounds element index of `layout`.
    let src_ptr = unsafe { storage.slice.offset_ptr(start_o) };

    let (grid, block) = launch_config_layout(&dev, el, ds.is_null());
    let ds_ptr: *const usize = ds.as_ptr();

    let src_dtype = storage.slice.dtype();
    // `cast.cu` instantiates fp8 against these and nothing else, in either
    // direction (`CAST_OP_FP8` / `CAST_OP_FP8_INTO`, plus the identity pair).
    // Checked once here so that both directions report the same way instead of
    // one failing as a missing kernel at launch.
    if src_dtype == DType::F8E4M3 || dtype == DType::F8E4M3 {
        let partner = if src_dtype == DType::F8E4M3 {
            dtype
        } else {
            src_dtype
        };
        if !matches!(
            partner,
            DType::F32
                | DType::F64
                | DType::F16
                | DType::BF16
                | DType::U8
                | DType::I32
                | DType::F8E4M3
        ) {
            return Err(rocm_error(format!(
                "to_dtype {src_dtype:?} -> {dtype:?} is not supported on ROCm: \
                 candle-kernels/src/cast.cu declares no such CAST_OP"
            )));
        }
    }
    let slice = match dtype {
        DType::U8 => cast_launch!(
            dev,
            grid,
            block,
            el,
            dims.len(),
            ds_ptr,
            src_ptr,
            src_dtype,
            u8,
            U8
        ),
        DType::U32 => {
            cast_launch!(
                dev,
                grid,
                block,
                el,
                dims.len(),
                ds_ptr,
                src_ptr,
                src_dtype,
                u32,
                U32
            )
        }
        DType::I64 => {
            cast_launch!(
                dev,
                grid,
                block,
                el,
                dims.len(),
                ds_ptr,
                src_ptr,
                src_dtype,
                i64,
                I64
            )
        }
        DType::BF16 => {
            cast_launch!(
                dev,
                grid,
                block,
                el,
                dims.len(),
                ds_ptr,
                src_ptr,
                src_dtype,
                bf16,
                BF16
            )
        }
        DType::F16 => {
            cast_launch!(
                dev,
                grid,
                block,
                el,
                dims.len(),
                ds_ptr,
                src_ptr,
                src_dtype,
                f16,
                F16
            )
        }
        DType::F32 => {
            cast_launch!(
                dev,
                grid,
                block,
                el,
                dims.len(),
                ds_ptr,
                src_ptr,
                src_dtype,
                f32,
                F32
            )
        }
        DType::F64 => {
            cast_launch!(
                dev,
                grid,
                block,
                el,
                dims.len(),
                ds_ptr,
                src_ptr,
                src_dtype,
                f64,
                F64
            )
        }
        // `cast.cu` ships exactly one i32 conversion in each direction, and both
        // have an fp8 partner; there is no `CAST_OP` with an int16_t source or
        // destination at all.
        DType::I32 if src_dtype == DType::F8E4M3 => {
            cast_launch!(
                dev,
                grid,
                block,
                el,
                dims.len(),
                ds_ptr,
                src_ptr,
                src_dtype,
                i32,
                I32
            )
        }
        DType::I16 | DType::I32 => {
            return Err(rocm_error(format!(
                "to_dtype {src_dtype:?} -> {dtype:?} is not supported on ROCm: \
                 candle-kernels/src/cast.cu declares no such CAST_OP"
            )))
        }
        DType::F8E4M3 => {
            cast_launch!(
                dev,
                grid,
                block,
                el,
                dims.len(),
                ds_ptr,
                src_ptr,
                src_dtype,
                F8E4M3,
                F8E4M3,
                "f8_e4m3"
            )
        }
        DType::F4 | DType::F6E2M3 | DType::F6E3M2 | DType::F8E8M0 => {
            return Err(rocm_error(format!(
                "{dtype:?} dtype is not supported for to_dtype on ROCm"
            )))
        }
    };

    Ok(RocmStorage { slice, device: dev })
}
