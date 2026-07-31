//! Elementwise op launchers: buffer copies, unary, binary and comparison ops.
//!
//! Argument order always comes from the kernel macro in `candle-kernels/src`,
//! never from what a neighbouring launcher happens to do.

use super::launch::{launch_config_layout, launch_kernel};
use super::params::{dims_and_strides, dims_and_strides_pair};
use super::{
    kernels, params_from_vec, try_kernel_name, Map1, Map2, Map2Any, RocmDevice, RocmStorage,
    RocmStorageSlice, SendSyncDeviceMemory, S,
};
use crate::op::CmpOp;
use crate::{Layout, Result};

/// Duplicates a whole device buffer.
///
/// The clone keeps the source allocation's length so that absolute element
/// offsets stay valid: `try_clone` hands the copy back to a caller that keeps
/// using the *original* layout, start offset and strides included.
pub(super) struct CloneBuffer;

impl Map1 for CloneBuffer {
    fn f<T: Copy + Send + Sync + 'static>(
        &self,
        src: &SendSyncDeviceMemory<T>,
        dev: &RocmDevice,
        _: &Layout,
    ) -> Result<SendSyncDeviceMemory<T>> {
        let mut dst = dev.alloc::<T>(src.count())?;
        dst.copy_from_device(src)?;
        Ok(dst)
    }
}

impl<U: crate::op::UnaryOpT> Map1 for U {
    fn f<T: Copy + Send + Sync + 'static>(
        &self,
        src: &SendSyncDeviceMemory<T>,
        dev: &RocmDevice,
        layout: &Layout,
    ) -> Result<SendSyncDeviceMemory<T>> {
        let shape = layout.shape();
        let dims = shape.dims();
        let elem_count = shape.elem_count();

        let func_name = try_kernel_name::<T>(U::KERNEL)?;
        let ds = dims_and_strides(dev, layout)?;
        let output = dev.alloc::<T>(elem_count)?;
        let (grid, block) = launch_config_layout(dev, elem_count, ds.is_null());

        unsafe {
            let src_ptr = src.ptr_at(layout.start_offset());
            let out_ptr = output.as_ptr();
            let ds_ptr: *const usize = ds.as_ptr();

            launch_kernel(
                dev,
                &kernels::UNARY,
                &func_name,
                grid,
                block,
                &mut [
                    &elem_count as *const usize as *mut std::ffi::c_void,
                    &dims.len() as *const usize as *mut std::ffi::c_void,
                    (&ds_ptr) as *const *const usize as *mut std::ffi::c_void,
                    (&src_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(output)
    }
}

impl<U: crate::op::BinaryOpT> Map2 for U {
    fn f<T: Copy + Send + Sync + 'static>(
        &self,
        lhs: &SendSyncDeviceMemory<T>,
        lhs_l: &Layout,
        rhs: &SendSyncDeviceMemory<T>,
        rhs_l: &Layout,
        dev: &RocmDevice,
    ) -> Result<SendSyncDeviceMemory<T>> {
        let shape = lhs_l.shape();
        let dims = shape.dims();
        let elem_count = shape.elem_count();

        let func_name = try_kernel_name::<T>(U::KERNEL)?;
        let ds = dims_and_strides_pair(dev, lhs_l, rhs_l)?;
        let output = dev.alloc::<T>(elem_count)?;
        let (grid, block) = launch_config_layout(dev, elem_count, ds.is_null());

        unsafe {
            let lhs_ptr = lhs.ptr_at(lhs_l.start_offset());
            let rhs_ptr = rhs.ptr_at(rhs_l.start_offset());
            let out_ptr = output.as_ptr();
            let ds_ptr: *const usize = ds.as_ptr();

            launch_kernel(
                dev,
                &kernels::BINARY,
                &func_name,
                grid,
                block,
                &mut [
                    &elem_count as *const usize as *mut std::ffi::c_void,
                    &dims.len() as *const usize as *mut std::ffi::c_void,
                    (&ds_ptr) as *const *const usize as *mut std::ffi::c_void,
                    (&lhs_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&rhs_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(output)
    }
}

pub(super) struct Cmp(pub CmpOp);

impl Map2Any for Cmp {
    fn f<T: Copy + Send + Sync + 'static>(
        &self,
        lhs: &SendSyncDeviceMemory<T>,
        lhs_l: &Layout,
        rhs: &SendSyncDeviceMemory<T>,
        rhs_l: &Layout,
        dev: &RocmDevice,
    ) -> Result<S> {
        let shape = lhs_l.shape();
        let dims = shape.dims();
        let elem_count = shape.elem_count();

        let name = match self.0 {
            CmpOp::Eq => "eq",
            CmpOp::Ne => "ne",
            CmpOp::Lt => "lt",
            CmpOp::Le => "le",
            CmpOp::Gt => "gt",
            CmpOp::Ge => "ge",
        };
        let func_name = try_kernel_name::<T>(name)?;
        let ds = dims_and_strides_pair(dev, lhs_l, rhs_l)?;
        // `BINARY_OP_OUT` always writes u8 regardless of the input dtype.
        let output = dev.alloc::<u8>(elem_count)?;
        let (grid, block) = launch_config_layout(dev, elem_count, ds.is_null());

        unsafe {
            let lhs_ptr = lhs.ptr_at(lhs_l.start_offset());
            let rhs_ptr = rhs.ptr_at(rhs_l.start_offset());
            let out_ptr = output.as_ptr();
            let ds_ptr: *const usize = ds.as_ptr();

            launch_kernel(
                dev,
                &kernels::BINARY,
                &func_name,
                grid,
                block,
                &mut [
                    &elem_count as *const usize as *mut std::ffi::c_void,
                    &dims.len() as *const usize as *mut std::ffi::c_void,
                    (&ds_ptr) as *const *const usize as *mut std::ffi::c_void,
                    (&lhs_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&rhs_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(S::U8(output))
    }
}

/// `where_cond`: selects from `t`/`f` per the predicate held in `.0`.
///
/// The predicate dtype is dispatched here and the value dtype by `Map2`, so the
/// two are independent — hence the `where_{ids}_{dtype}` kernel names.
pub(super) struct WhereCond<'a>(pub &'a RocmStorage, pub &'a Layout);

impl Map2 for WhereCond<'_> {
    fn f<T: Copy + Send + Sync + 'static>(
        &self,
        t: &SendSyncDeviceMemory<T>,
        layout_t: &Layout,
        f: &SendSyncDeviceMemory<T>,
        layout_f: &Layout,
        dev: &RocmDevice,
    ) -> Result<SendSyncDeviceMemory<T>> {
        let ids_l = self.1;
        let (ids_ptr, name) = match &self.0.slice {
            RocmStorageSlice::U8(s) => (unsafe { s.ptr_at(ids_l.start_offset()) }, "where_u8"),
            RocmStorageSlice::U32(s) => (unsafe { s.ptr_at(ids_l.start_offset()) }, "where_u32"),
            RocmStorageSlice::I64(s) => (unsafe { s.ptr_at(ids_l.start_offset()) }, "where_i64"),
            slice => crate::bail!(
                "where conditions should be u8/u32/i64, got {:?}",
                slice.dtype()
            ),
        };

        let shape = ids_l.shape();
        let dims = shape.dims();
        let el = shape.elem_count();

        let func_name = try_kernel_name::<T>(name)?;
        // `WHERE_OP` dereferences `info` unconditionally to test contiguity, so
        // unlike the binary ops it never accepts a null pointer here.
        let ds = params_from_vec(
            dev,
            [dims, ids_l.stride(), layout_t.stride(), layout_f.stride()].concat(),
        )?;
        let output = dev.alloc::<T>(el)?;
        // `WHERE_OP` always takes an `info` buffer, so contiguity has to be
        // tested here rather than read off the buffer as the others do.
        let contiguous =
            ids_l.is_contiguous() && layout_t.is_contiguous() && layout_f.is_contiguous();
        let (grid, block) = launch_config_layout(dev, el, contiguous);

        unsafe {
            let t_ptr = t.ptr_at(layout_t.start_offset());
            let f_ptr = f.ptr_at(layout_f.start_offset());
            let out_ptr = output.as_ptr();
            let ds_ptr = ds.as_ptr();

            launch_kernel(
                dev,
                &kernels::TERNARY,
                &func_name,
                grid,
                block,
                &mut [
                    &el as *const usize as *mut std::ffi::c_void,
                    &dims.len() as *const usize as *mut std::ffi::c_void,
                    (&ds_ptr) as *const *const usize as *mut std::ffi::c_void,
                    (&ids_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&t_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&f_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(output)
    }
}
