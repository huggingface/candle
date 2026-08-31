//! Launchers for the ops that carry a scalar operand: `affine`, `powf`, `elu`.
//!
//! `UNARY_OP1` (uelu/upowf) takes its scalar *before* the buffers while
//! `AFFINE_OP` takes both of its scalars *after* them. Read the macro in
//! `candle-kernels/src` before touching either argument list.

use super::launch::{launch_config_layout, launch_kernel};
use super::params::dims_and_strides;
use super::{kernels, try_kernel_name, RocmDevice, RocmStorageSlice, SendSyncDeviceMemory};
use crate::{Layout, Result, WithDType};

pub(crate) struct Affine(pub f64, pub f64);

impl Affine {
    pub(super) fn map(
        &self,
        s: &RocmStorageSlice,
        d: &RocmDevice,
        l: &Layout,
    ) -> Result<RocmStorageSlice> {
        let out = match s {
            RocmStorageSlice::U8(s) => RocmStorageSlice::U8(self.f(s, d, l)?),
            RocmStorageSlice::U32(s) => RocmStorageSlice::U32(self.f(s, d, l)?),
            RocmStorageSlice::I16(s) => RocmStorageSlice::I16(self.f(s, d, l)?),
            RocmStorageSlice::I32(s) => RocmStorageSlice::I32(self.f(s, d, l)?),
            RocmStorageSlice::I64(s) => RocmStorageSlice::I64(self.f(s, d, l)?),
            RocmStorageSlice::BF16(s) => RocmStorageSlice::BF16(self.f(s, d, l)?),
            RocmStorageSlice::F16(s) => RocmStorageSlice::F16(self.f(s, d, l)?),
            RocmStorageSlice::F32(s) => RocmStorageSlice::F32(self.f(s, d, l)?),
            RocmStorageSlice::F64(s) => RocmStorageSlice::F64(self.f(s, d, l)?),
            RocmStorageSlice::F8E4M3(s) => RocmStorageSlice::F8E4M3(self.f(s, d, l)?),
        };
        Ok(out)
    }

    fn f<T: Copy + Send + Sync + WithDType + 'static>(
        &self,
        src: &SendSyncDeviceMemory<T>,
        dev: &RocmDevice,
        layout: &Layout,
    ) -> Result<SendSyncDeviceMemory<T>> {
        let shape = layout.shape();
        let dims = shape.dims();
        let elem_count = shape.elem_count();

        let func_name = try_kernel_name::<T>("affine")?;
        let ds = dims_and_strides(dev, layout)?;
        let output = dev.alloc::<T>(elem_count)?;
        let (grid, block) = launch_config_layout(dev, elem_count, ds.is_null());

        let mul_val = T::from_f64(self.0);
        let add_val = T::from_f64(self.1);

        unsafe {
            let src_ptr = src.ptr_at(layout.start_offset());
            let out_ptr = output.as_ptr();
            let ds_ptr: *const usize = ds.as_ptr();

            launch_kernel(
                dev,
                &kernels::AFFINE,
                &func_name,
                grid,
                block,
                &mut [
                    &elem_count as *const usize as *mut std::ffi::c_void,
                    &dims.len() as *const usize as *mut std::ffi::c_void,
                    (&ds_ptr) as *const *const usize as *mut std::ffi::c_void,
                    (&src_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    &mul_val as *const T as *mut std::ffi::c_void,
                    &add_val as *const T as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(output)
    }
}

pub(super) struct Powf(pub f64);

impl Powf {
    pub(super) fn map(
        &self,
        s: &RocmStorageSlice,
        d: &RocmDevice,
        l: &Layout,
    ) -> Result<RocmStorageSlice> {
        let out = match s {
            RocmStorageSlice::U8(s) => RocmStorageSlice::U8(self.f(s, d, l)?),
            RocmStorageSlice::U32(s) => RocmStorageSlice::U32(self.f(s, d, l)?),
            RocmStorageSlice::I16(s) => RocmStorageSlice::I16(self.f(s, d, l)?),
            RocmStorageSlice::I32(s) => RocmStorageSlice::I32(self.f(s, d, l)?),
            RocmStorageSlice::I64(s) => RocmStorageSlice::I64(self.f(s, d, l)?),
            RocmStorageSlice::BF16(s) => RocmStorageSlice::BF16(self.f(s, d, l)?),
            RocmStorageSlice::F16(s) => RocmStorageSlice::F16(self.f(s, d, l)?),
            RocmStorageSlice::F32(s) => RocmStorageSlice::F32(self.f(s, d, l)?),
            RocmStorageSlice::F64(s) => RocmStorageSlice::F64(self.f(s, d, l)?),
            RocmStorageSlice::F8E4M3(s) => RocmStorageSlice::F8E4M3(self.f(s, d, l)?),
        };
        Ok(out)
    }

    fn f<T: Copy + Send + Sync + WithDType + 'static>(
        &self,
        src: &SendSyncDeviceMemory<T>,
        dev: &RocmDevice,
        layout: &Layout,
    ) -> Result<SendSyncDeviceMemory<T>> {
        let shape = layout.shape();
        let dims = shape.dims();
        let elem_count = shape.elem_count();

        let func_name = try_kernel_name::<T>("upowf")?;
        let ds = dims_and_strides(dev, layout)?;
        let output = dev.alloc::<T>(elem_count)?;
        let (grid, block) = launch_config_layout(dev, elem_count, ds.is_null());

        let scalar_val = T::from_f64(self.0);

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
                    // UNARY_OP1 takes its scalar before the buffers, unlike
                    // AFFINE_OP which takes them after.
                    &scalar_val as *const T as *mut std::ffi::c_void,
                    (&src_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(output)
    }
}

pub(super) struct Elu(pub f64);

impl Elu {
    pub(super) fn map(
        &self,
        s: &RocmStorageSlice,
        d: &RocmDevice,
        l: &Layout,
    ) -> Result<RocmStorageSlice> {
        let out = match s {
            RocmStorageSlice::U8(s) => RocmStorageSlice::U8(self.f(s, d, l)?),
            RocmStorageSlice::U32(s) => RocmStorageSlice::U32(self.f(s, d, l)?),
            RocmStorageSlice::I16(s) => RocmStorageSlice::I16(self.f(s, d, l)?),
            RocmStorageSlice::I32(s) => RocmStorageSlice::I32(self.f(s, d, l)?),
            RocmStorageSlice::I64(s) => RocmStorageSlice::I64(self.f(s, d, l)?),
            RocmStorageSlice::BF16(s) => RocmStorageSlice::BF16(self.f(s, d, l)?),
            RocmStorageSlice::F16(s) => RocmStorageSlice::F16(self.f(s, d, l)?),
            RocmStorageSlice::F32(s) => RocmStorageSlice::F32(self.f(s, d, l)?),
            RocmStorageSlice::F64(s) => RocmStorageSlice::F64(self.f(s, d, l)?),
            RocmStorageSlice::F8E4M3(s) => RocmStorageSlice::F8E4M3(self.f(s, d, l)?),
        };
        Ok(out)
    }

    fn f<T: Copy + Send + Sync + WithDType + 'static>(
        &self,
        src: &SendSyncDeviceMemory<T>,
        dev: &RocmDevice,
        layout: &Layout,
    ) -> Result<SendSyncDeviceMemory<T>> {
        let shape = layout.shape();
        let dims = shape.dims();
        let elem_count = shape.elem_count();

        let func_name = try_kernel_name::<T>("uelu")?;
        let ds = dims_and_strides(dev, layout)?;
        let output = dev.alloc::<T>(elem_count)?;
        let (grid, block) = launch_config_layout(dev, elem_count, ds.is_null());

        let alpha_val = T::from_f64(self.0);

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
                    // UNARY_OP1 takes its scalar before the buffers, unlike
                    // AFFINE_OP which takes them after.
                    &alpha_val as *const T as *mut std::ffi::c_void,
                    (&src_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(output)
    }
}
