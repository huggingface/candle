//! `BackendStorage::to_dtype`, i.e. the `cast.cu` launchers.
//!
//! Split out of `mod.rs`, which is far over the workspace 400-line cap.

use half::{bf16, f16};

use super::launch::{launch_config_layout, launch_kernel};
use super::params::dims_and_strides;
use super::{kernels, rocm_error, RocmStorage, RocmStorageSlice};
use crate::{DType, Layout, Result};

macro_rules! cast_launch {
    ($dev:expr, $grid:expr, $block:expr, $el:expr, $dims_len:expr, $ds_ptr:expr, $src_ptr:expr, $src_dtype:expr, $rust_type:ty, $variant:ident) => {{
        let out = $dev.alloc::<$rust_type>($el)?;
        let out_ptr = out.as_ptr();
        let func_name = format!("cast_{}_{}", $src_dtype.as_str(), stringify!($rust_type));
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
        // Still real: the shared `cast.cu` declares no `CAST_OP` with an
        // int16_t/int32_t source or destination (the only i32 entries are
        // the fp8 pair below), so there is no kernel to launch.
        DType::I16 | DType::I32 => {
            return Err(rocm_error(
                "i16/i32 dtypes are not supported for to_dtype on ROCm",
            ))
        }
        // `cast.cu` does ship the f8e4m3 casts, but they are named
        // `cast_*_f8_e4m3` while `DType::as_str` yields `f8e4m3` and
        // `cast_launch!` derives the destination suffix from the Rust type
        // (F8E4M3 is stored as `u8` here). Wiring that up is a separate
        // change; the other four dtypes have no kernels at all.
        DType::F8E4M3 | DType::F4 | DType::F6E2M3 | DType::F6E3M2 | DType::F8E8M0 => {
            return Err(rocm_error(format!(
                "{dtype:?} dtype is not supported for to_dtype on ROCm"
            )))
        }
    };

    Ok(RocmStorage { slice, device: dev })
}
