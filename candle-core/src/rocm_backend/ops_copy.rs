//! The three `BackendStorage` methods that move or fill bytes in place:
//! `copy_strided_src`, `copy2d` and `const_set`.
//!
//! Split out of `mod.rs`, which is far over the workspace 400-line cap.

use half::{bf16, f16};
use rocm_rs::hip::bindings;

use super::launch::{launch_config_layout, launch_kernel};
use super::params::dims_and_strides;
use super::{kernels, RocmStorage, RocmStorageSlice};
use crate::backend::BackendStorage;
use crate::{Layout, Result};

pub(super) fn copy_strided_src(
    src: &RocmStorage,
    dst: &mut RocmStorage,
    dst_offset: usize,
    src_l: &Layout,
) -> Result<()> {
    let src_shape = src_l.shape();
    let dims = src_shape.dims();
    let el_count = src_shape.elem_count();
    if el_count == 0 {
        return Ok(());
    }
    if src.dtype() != dst.dtype() {
        crate::bail!("dtype mismatch in copy_strided_src");
    }
    src.device.bind()?;

    // Callers over-request: `Tensor::cat` and the autograd accumulators size
    // the copy from the *source shape*, which can run past the end of either
    // allocation once start offsets are applied. cuda_backend clamps for the
    // same reason (`slice_src_and_dst`); without it this memcpy walks off the
    // end of a device buffer.
    let dst_avail = dst.slice.count().saturating_sub(dst_offset);

    if src_l.is_contiguous() {
        let src_avail = src.slice.count().saturating_sub(src_l.start_offset());
        let to_copy = el_count.min(src_avail).min(dst_avail);
        if to_copy == 0 {
            return Ok(());
        }
        let el_size = src.slice.elem_size();
        // SAFETY: both offsets are in-bounds element indices, and `to_copy` was
        // clamped to what remains in the shorter of the two allocations.
        let src_ptr = unsafe { src.slice.offset_ptr(src_l.start_offset()) };
        let dst_ptr = unsafe { dst.slice.offset_mut_ptr(dst_offset) };
        // Stream-ordered: the copy is sequenced against the kernels that
        // produced `src` and consume `dst`, so the host never has to wait.
        let result = unsafe {
            bindings::hipMemcpyAsync(
                dst_ptr,
                src_ptr,
                to_copy * el_size,
                bindings::hipMemcpyKind_hipMemcpyDeviceToDevice,
                src.device.stream().as_raw(),
            )
        };
        if result != bindings::hipError_t_hipSuccess {
            crate::bail!("hipMemcpyAsync failed with error {}", result);
        }
        return Ok(());
    }

    // Same destination clamp. The source side is deliberately *not* clamped
    // here: a broadcast layout carries stride 0 and legitimately reads far
    // fewer elements than it writes.
    let el_count = el_count.min(dst_avail);
    if el_count == 0 {
        return Ok(());
    }
    // This branch is only reached for a non-contiguous source.
    let (grid, block) = launch_config_layout(&src.device, el_count, false);
    let ds = dims_and_strides(&src.device, src_l)?;

    macro_rules! copy_strided {
        ($variant:ident, $suffix:expr) => {{
            let (src_mem, dst_mem) = match (&src.slice, &mut dst.slice) {
                (RocmStorageSlice::$variant(s), RocmStorageSlice::$variant(d)) => (s, d),
                _ => crate::bail!("dtype mismatch in copy_strided_src"),
            };
            let func_name = format!("ucopy_{}", $suffix);
            let (src_ptr, dst_ptr) = unsafe {
                (
                    src_mem.ptr_at(src_l.start_offset()),
                    dst_mem.ptr_at(dst_offset),
                )
            };
            let ds_ptr: *const usize = ds.as_ptr();
            unsafe {
                launch_kernel(
                    &src.device,
                    &kernels::UNARY,
                    &func_name,
                    grid,
                    block,
                    &mut [
                        &el_count as *const usize as *mut std::ffi::c_void,
                        &dims.len() as *const usize as *mut std::ffi::c_void,
                        (&ds_ptr) as *const *const usize as *mut std::ffi::c_void,
                        (&src_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                        (&dst_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }};
    }

    match &src.slice {
        RocmStorageSlice::U8(_) => copy_strided!(U8, "u8"),
        RocmStorageSlice::U32(_) => copy_strided!(U32, "u32"),
        RocmStorageSlice::I16(_) => copy_strided!(I16, "i16"),
        RocmStorageSlice::I32(_) => copy_strided!(I32, "i32"),
        RocmStorageSlice::I64(_) => copy_strided!(I64, "i64"),
        RocmStorageSlice::BF16(_) => copy_strided!(BF16, "bf16"),
        RocmStorageSlice::F16(_) => copy_strided!(F16, "f16"),
        RocmStorageSlice::F32(_) => copy_strided!(F32, "f32"),
        RocmStorageSlice::F64(_) => copy_strided!(F64, "f64"),
        // `unary.cu` gates `ucopy_f8_e4m3` on `__CUDA_ARCH__ >= 890` while the
        // ROCm module is compiled at 800, so that symbol is not in the
        // binary. F8E4M3 is exactly one byte and its payload is already held
        // as `u8`, so `ucopy_u8` moves the identical bytes — this is the same
        // reasoning `try_clone` uses for its raw buffer copy.
        RocmStorageSlice::F8E4M3(_) => copy_strided!(F8E4M3, "u8"),
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn copy2d(
    src: &RocmStorage,
    dst: &mut RocmStorage,
    d1: usize,
    d2: usize,
    src_s1: usize,
    dst_s1: usize,
    src_o: usize,
    dst_o: usize,
) -> Result<()> {
    if d1 == 0 || d2 == 0 {
        return Ok(());
    }
    src.device.bind()?;
    let (src_ptr, dst_ptr, el_size) = match (&src.slice, &mut dst.slice) {
        (RocmStorageSlice::U8(s), RocmStorageSlice::U8(d)) => (s.as_ptr(), d.as_ptr(), 1usize),
        (RocmStorageSlice::U32(s), RocmStorageSlice::U32(d)) => (s.as_ptr(), d.as_ptr(), 4),
        (RocmStorageSlice::I16(s), RocmStorageSlice::I16(d)) => (s.as_ptr(), d.as_ptr(), 2),
        (RocmStorageSlice::I32(s), RocmStorageSlice::I32(d)) => (s.as_ptr(), d.as_ptr(), 4),
        (RocmStorageSlice::I64(s), RocmStorageSlice::I64(d)) => (s.as_ptr(), d.as_ptr(), 8),
        (RocmStorageSlice::BF16(s), RocmStorageSlice::BF16(d)) => (s.as_ptr(), d.as_ptr(), 2),
        (RocmStorageSlice::F16(s), RocmStorageSlice::F16(d)) => (s.as_ptr(), d.as_ptr(), 2),
        (RocmStorageSlice::F32(s), RocmStorageSlice::F32(d)) => (s.as_ptr(), d.as_ptr(), 4),
        (RocmStorageSlice::F64(s), RocmStorageSlice::F64(d)) => (s.as_ptr(), d.as_ptr(), 8),
        (RocmStorageSlice::F8E4M3(s), RocmStorageSlice::F8E4M3(d)) => (s.as_ptr(), d.as_ptr(), 1),
        _ => crate::bail!("dtype mismatch in copy2d"),
    };
    let src_ptr = unsafe { src_ptr.add(src_o * el_size) };
    let dst_ptr = unsafe { dst_ptr.add(dst_o * el_size) };
    let width = d2 * el_size;
    let spitch = src_s1 * el_size;
    let dpitch = dst_s1 * el_size;
    // Stream-ordered, like the contiguous path in `copy_strided_src`.
    let result = unsafe {
        bindings::hipMemcpy2DAsync(
            dst_ptr,
            dpitch,
            src_ptr,
            spitch,
            width,
            d1,
            bindings::hipMemcpyKind_hipMemcpyDeviceToDevice,
            src.device.stream().as_raw(),
        )
    };
    if result != bindings::hipError_t_hipSuccess {
        crate::bail!("hipMemcpy2DAsync failed with error {}", result);
    }
    Ok(())
}

pub(super) fn const_set(
    storage: &mut RocmStorage,
    val: crate::scalar::Scalar,
    layout: &Layout,
) -> Result<()> {
    let shape = layout.shape();
    let dims = shape.dims();
    let el_count = shape.elem_count();
    if el_count == 0 {
        return Ok(());
    }

    let ds = dims_and_strides(&storage.device, layout)?;
    let (grid, block) = launch_config_layout(&storage.device, el_count, ds.is_null());

    macro_rules! const_set {
        ($variant:ident, $suffix:expr, $ty:ty, $val:expr) => {{
            let mem = match &mut storage.slice {
                RocmStorageSlice::$variant(m) => m,
                _ => crate::bail!("dtype mismatch in const_set"),
            };
            let func_name = format!("const_set_{}", $suffix);
            let out_ptr = unsafe { mem.ptr_at(layout.start_offset()) };
            let scalar_val: $ty = $val;
            let ds_ptr: *const usize = ds.as_ptr();
            unsafe {
                launch_kernel(
                    &storage.device,
                    &kernels::FILL,
                    &func_name,
                    grid,
                    block,
                    &mut [
                        &el_count as *const usize as *mut std::ffi::c_void,
                        &dims.len() as *const usize as *mut std::ffi::c_void,
                        (&ds_ptr) as *const *const usize as *mut std::ffi::c_void,
                        &scalar_val as *const $ty as *mut std::ffi::c_void,
                        (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }};
    }

    match (&mut storage.slice, val) {
        (RocmStorageSlice::U8(_), crate::scalar::Scalar::U8(v)) => const_set!(U8, "u8", u8, v),
        (RocmStorageSlice::U32(_), crate::scalar::Scalar::U32(v)) => const_set!(U32, "u32", u32, v),
        (RocmStorageSlice::I64(_), crate::scalar::Scalar::I64(v)) => const_set!(I64, "i64", i64, v),
        (RocmStorageSlice::F32(_), crate::scalar::Scalar::F32(v)) => const_set!(F32, "f32", f32, v),
        (RocmStorageSlice::F64(_), crate::scalar::Scalar::F64(v)) => const_set!(F64, "f64", f64, v),
        (RocmStorageSlice::BF16(_), crate::scalar::Scalar::BF16(v)) => {
            const_set!(BF16, "bf16", bf16, v)
        }
        (RocmStorageSlice::F16(_), crate::scalar::Scalar::F16(v)) => const_set!(F16, "f16", f16, v),
        (RocmStorageSlice::I16(_), crate::scalar::Scalar::I16(v)) => const_set!(I16, "i16", i16, v),
        (RocmStorageSlice::I32(_), crate::scalar::Scalar::I32(v)) => const_set!(I32, "i32", i32, v),
        // `RocmStorageSlice::F8E4M3` keeps its payload as bytes, and F8E4M3
        // is exactly one byte, so `ptr_at` on the u8 buffer still lands on
        // element `start_offset`.
        (RocmStorageSlice::F8E4M3(_), crate::scalar::Scalar::F8E4M3(v)) => {
            const_set!(F8E4M3, "f8_e4m3", float8::F8E4M3, v)
        }
        _ => crate::bail!("dtype mismatch in const_set"),
    }

    Ok(())
}
