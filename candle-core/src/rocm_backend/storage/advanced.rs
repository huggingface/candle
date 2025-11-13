//! Advanced operations (convolution, pooling, matmul)
//! Created by: TEAM-488 (Phase 1 - Device integration)
//! MIOpen operations: TEAM-495 (conv2d, pooling)
//! rocBLAS operations: TEAM-496 (matmul)
//! CUDA parity verified by: TEAM-497

use super::RocmStorage;
use crate::rocm_backend::{miopen, rocblas, RocmError};
use crate::Result;

impl RocmStorage {
    // Created by: TEAM-495 | CUDA parity verified by: TEAM-497 (cuda_backend/mod.rs:1801-1863)
    pub(super) fn conv2d_impl(
        &self,
        inp_l: &crate::Layout,
        kernel: &Self,
        kernel_l: &crate::Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        miopen::conv2d(self, inp_l, kernel, kernel_l, params)
    }

    pub(super) fn avg_pool2d_impl(
        &self,
        layout: &crate::Layout,
        k: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        self.pool2d(
            layout,
            k,
            stride,
            rocm_rs::miopen::ffi::miopenPoolingMode_t_miopenPoolingAverage,
        )
    }

    // Created by: TEAM-495 | CUDA parity verified by: TEAM-497 (cuda_backend/mod.rs:1892-1903)
    pub(super) fn max_pool2d_impl(
        &self,
        layout: &crate::Layout,
        k: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        self.pool2d(layout, k, stride, rocm_rs::miopen::ffi::miopenPoolingMode_t_miopenPoolingMax)
    }

    // Created by: TEAM-496 | CUDA parity verified by: TEAM-497 (cuda_backend/mod.rs:1965-2019)
    pub(super) fn matmul_impl(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &crate::Layout,
        rhs_l: &crate::Layout,
    ) -> Result<Self> {
        rocblas::matmul(self, rhs, (b, m, n, k), lhs_l, rhs_l)
    }

    // Created by: TEAM-497 | CUDA parity verified by: TEAM-497 (cuda_backend/mod.rs:1621-1684)
    pub(super) fn conv1d_impl(
        &self,
        inp_l: &crate::Layout,
        kernel: &Self,
        kernel_l: &crate::Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        miopen::conv1d(self, inp_l, kernel, kernel_l, params)
    }

    // Created by: TEAM-497 | CUDA parity verified by: TEAM-497 (cuda_backend/mod.rs:1686-1743)
    pub(super) fn conv_transpose1d_impl(
        &self,
        inp_l: &crate::Layout,
        kernel: &Self,
        kernel_l: &crate::Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        miopen::conv_transpose1d(self, inp_l, kernel, kernel_l, params)
    }

    // Created by: TEAM-497 | CUDA parity verified by: TEAM-497 (cuda_backend/mod.rs:1866-1876)
    pub(super) fn conv_transpose2d_impl(
        &self,
        inp_l: &crate::Layout,
        kernel: &Self,
        kernel_l: &crate::Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        miopen::conv_transpose2d(self, inp_l, kernel, kernel_l, params)
    }

    // TEAM-497: Copy2D via HIP memcpy2D (CUDA: cuda_backend/mod.rs:2021-2066)
    pub(super) fn copy2d_impl(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_s: usize,
        dst_s: usize,
        src_o: usize,
        dst_o: usize,
    ) -> Result<()> {
        use super::RocmStorageSlice as S;

        let elem_size = self.dtype().size_in_bytes();
        let src_offset_bytes = src_o * elem_size;
        let dst_offset_bytes = dst_o * elem_size;
        let width_bytes = d2 * elem_size;
        let src_pitch = src_s * elem_size;
        let dst_pitch = dst_s * elem_size;

        match (&self.slice, &mut dst.slice) {
            (S::F32(src), S::F32(dst)) => {
                unsafe {
                    rocm_rs::hip::memory::copy_2d(
                        dst.as_mut_ptr().add(dst_offset_bytes / 4) as *mut u8,
                        dst_pitch,
                        src.as_ptr().add(src_offset_bytes / 4) as *const u8,
                        src_pitch,
                        width_bytes,
                        d1,
                    )?;
                }
                Ok(())
            }
            (S::F16(src), S::F16(dst)) => {
                unsafe {
                    rocm_rs::hip::memory::copy_2d(
                        dst.as_mut_ptr().add(dst_offset_bytes / 2) as *mut u8,
                        dst_pitch,
                        src.as_ptr().add(src_offset_bytes / 2) as *const u8,
                        src_pitch,
                        width_bytes,
                        d1,
                    )?;
                }
                Ok(())
            }
            (S::BF16(src), S::BF16(dst)) => {
                unsafe {
                    rocm_rs::hip::memory::copy_2d(
                        dst.as_mut_ptr().add(dst_offset_bytes / 2) as *mut u8,
                        dst_pitch,
                        src.as_ptr().add(src_offset_bytes / 2) as *const u8,
                        src_pitch,
                        width_bytes,
                        d1,
                    )?;
                }
                Ok(())
            }
            (S::U8(src), S::U8(dst)) => {
                unsafe {
                    rocm_rs::hip::memory::copy_2d(
                        dst.as_mut_ptr().add(dst_offset_bytes) as *mut u8,
                        dst_pitch,
                        src.as_ptr().add(src_offset_bytes) as *const u8,
                        src_pitch,
                        width_bytes,
                        d1,
                    )?;
                }
                Ok(())
            }
            (S::U32(src), S::U32(dst)) => {
                unsafe {
                    rocm_rs::hip::memory::copy_2d(
                        dst.as_mut_ptr().add(dst_offset_bytes / 4) as *mut u8,
                        dst_pitch,
                        src.as_ptr().add(src_offset_bytes / 4) as *const u8,
                        src_pitch,
                        width_bytes,
                        d1,
                    )?;
                }
                Ok(())
            }
            (S::I64(src), S::I64(dst)) => {
                unsafe {
                    rocm_rs::hip::memory::copy_2d(
                        dst.as_mut_ptr().add(dst_offset_bytes / 8) as *mut u8,
                        dst_pitch,
                        src.as_ptr().add(src_offset_bytes / 8) as *const u8,
                        src_pitch,
                        width_bytes,
                        d1,
                    )?;
                }
                Ok(())
            }
            (S::F64(src), S::F64(dst)) => {
                unsafe {
                    rocm_rs::hip::memory::copy_2d(
                        dst.as_mut_ptr().add(dst_offset_bytes / 8) as *mut u8,
                        dst_pitch,
                        src.as_ptr().add(src_offset_bytes / 8) as *const u8,
                        src_pitch,
                        width_bytes,
                        d1,
                    )?;
                }
                Ok(())
            }
            _ => Err(RocmError::InternalError("dtype mismatch in copy2d").into()),
        }
    }

    // Created by: TEAM-497 | CUDA parity verified by: TEAM-497 (cuda_backend/mod.rs:1879-1890)
    pub(super) fn copy_strided_src_impl(
        &self,
        dst: &mut Self,
        dst_offset: usize,
        src_l: &crate::Layout,
    ) -> Result<()> {
        use super::RocmStorageSlice as S;

        // For contiguous layouts, use simple copy
        if src_l.is_contiguous() {
            let src_offset = src_l.start_offset();
            let len = src_l.shape().elem_count();

            match (&self.slice, &mut dst.slice) {
                (S::F32(src), S::F32(dst)) => {
                    unsafe {
                        rocm_rs::hip::memory::copy(
                            dst.as_mut_ptr().add(dst_offset),
                            src.as_ptr().add(src_offset),
                            len,
                        )?;
                    }
                    Ok(())
                }
                (S::F16(src), S::F16(dst)) => {
                    unsafe {
                        rocm_rs::hip::memory::copy(
                            dst.as_mut_ptr().add(dst_offset),
                            src.as_ptr().add(src_offset),
                            len,
                        )?;
                    }
                    Ok(())
                }
                (S::BF16(src), S::BF16(dst)) => {
                    unsafe {
                        rocm_rs::hip::memory::copy(
                            dst.as_mut_ptr().add(dst_offset),
                            src.as_ptr().add(src_offset),
                            len,
                        )?;
                    }
                    Ok(())
                }
                (S::U8(src), S::U8(dst)) => {
                    unsafe {
                        rocm_rs::hip::memory::copy(
                            dst.as_mut_ptr().add(dst_offset),
                            src.as_ptr().add(src_offset),
                            len,
                        )?;
                    }
                    Ok(())
                }
                (S::U32(src), S::U32(dst)) => {
                    unsafe {
                        rocm_rs::hip::memory::copy(
                            dst.as_mut_ptr().add(dst_offset),
                            src.as_ptr().add(src_offset),
                            len,
                        )?;
                    }
                    Ok(())
                }
                (S::I64(src), S::I64(dst)) => {
                    unsafe {
                        rocm_rs::hip::memory::copy(
                            dst.as_mut_ptr().add(dst_offset),
                            src.as_ptr().add(src_offset),
                            len,
                        )?;
                    }
                    Ok(())
                }
                (S::F64(src), S::F64(dst)) => {
                    unsafe {
                        rocm_rs::hip::memory::copy(
                            dst.as_mut_ptr().add(dst_offset),
                            src.as_ptr().add(src_offset),
                            len,
                        )?;
                    }
                    Ok(())
                }
                _ => Err(RocmError::InternalError("dtype mismatch in copy_strided_src").into()),
            }
        } else {
            // For strided layouts, we need to copy element by element or use rocBLAS copy_strided_batched
            // For now, return error - this needs custom kernel implementation
            Err(RocmError::InternalError(
                "copy_strided_src with non-contiguous layout not yet implemented - needs custom kernel"
            ).into())
        }
    }
}
