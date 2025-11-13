//! rocBLAS integration for ROCm backend
//!
//! Created by: TEAM-496 (Module refactoring - extracted from mod.rs)
//! CUDA parity verified by: TEAM-498
//!
//! This module provides matrix multiplication operations using AMD's rocBLAS library.
//! Matches cuda_backend BLAS operations pattern (cuBLAS equivalent).

use crate::rocm_backend::{RocmDevice, RocmError, RocmStorageSlice as S};
use crate::{Layout, Result};
use half::{bf16, f16};
use rocm_rs::rocblas::{Handle, Operation};

// TEAM-496 | CUDA parity: cuda_backend/mod.rs:1965-2019 (matmul method)
/// Matrix multiplication using rocBLAS GEMM
/// Matches cuda_backend/mod.rs::matmul pattern
pub(crate) fn matmul(
    storage: &crate::rocm_backend::RocmStorage,
    rhs: &crate::rocm_backend::RocmStorage,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_l: &Layout,
    rhs_l: &Layout,
) -> Result<crate::rocm_backend::RocmStorage> {
    let elem_count = b * m * n;
    let dev = storage.device();

    // Create rocBLAS handle
    let handle = Handle::new()
        .map_err(|e| RocmError::InternalError(&format!("rocBLAS handle creation failed: {:?}", e)))?;

    let slice = match (&storage.slice, &rhs.slice) {
        (S::F32(lhs), S::F32(rhs_slice)) => {
            let lhs_slice = &lhs.slice(lhs_l.start_offset()..);
            let rhs_slice_data = &rhs_slice.slice(rhs_l.start_offset()..);
            let mut out = unsafe { dev.hip_device().alloc::<f32>(elem_count)? };

            // rocBLAS GEMM: C = alpha * op(A) * op(B) + beta * C
            // We compute: out = 1.0 * rhs * lhs + 0.0 * out
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;

            unsafe {
                rocm_rs::rocblas::level3::gemm_strided_batched(
                    &handle,
                    Operation::None, // No transpose for rhs
                    Operation::None, // No transpose for lhs
                    n as i32,
                    m as i32,
                    k as i32,
                    &alpha,
                    rhs_slice_data.as_ptr(),
                    n as i32,        // lda
                    (n * k) as i64,  // stride_a
                    lhs_slice.as_ptr(),
                    k as i32,        // ldb
                    (m * k) as i64,  // stride_b
                    &beta,
                    out.as_mut_ptr(),
                    n as i32,        // ldc
                    (m * n) as i64,  // stride_c
                    b as i32,        // batch_count
                )
                .map_err(|e| RocmError::InternalError(&format!("rocBLAS GEMM failed: {:?}", e)))?;
            }

            S::F32(out)
        }
        (S::F64(lhs), S::F64(rhs_slice)) => {
            let lhs_slice = &lhs.slice(lhs_l.start_offset()..);
            let rhs_slice_data = &rhs_slice.slice(rhs_l.start_offset()..);
            let mut out = unsafe { dev.hip_device().alloc::<f64>(elem_count)? };

            let alpha: f64 = 1.0;
            let beta: f64 = 0.0;

            unsafe {
                rocm_rs::rocblas::level3::gemm_strided_batched(
                    &handle,
                    Operation::None,
                    Operation::None,
                    n as i32,
                    m as i32,
                    k as i32,
                    &alpha,
                    rhs_slice_data.as_ptr(),
                    n as i32,
                    (n * k) as i64,
                    lhs_slice.as_ptr(),
                    k as i32,
                    (m * k) as i64,
                    &beta,
                    out.as_mut_ptr(),
                    n as i32,
                    (m * n) as i64,
                    b as i32,
                )
                .map_err(|e| RocmError::InternalError(&format!("rocBLAS GEMM failed: {:?}", e)))?;
            }

            S::F64(out)
        }
        (S::F16(lhs), S::F16(rhs_slice)) => {
            let lhs_slice = &lhs.slice(lhs_l.start_offset()..);
            let rhs_slice_data = &rhs_slice.slice(rhs_l.start_offset()..);
            let mut out = unsafe { dev.hip_device().alloc::<f16>(elem_count)? };

            let alpha = f16::ONE;
            let beta = f16::ZERO;

            unsafe {
                rocm_rs::rocblas::level3::gemm_strided_batched(
                    &handle,
                    Operation::None,
                    Operation::None,
                    n as i32,
                    m as i32,
                    k as i32,
                    &alpha,
                    rhs_slice_data.as_ptr(),
                    n as i32,
                    (n * k) as i64,
                    lhs_slice.as_ptr(),
                    k as i32,
                    (m * k) as i64,
                    &beta,
                    out.as_mut_ptr(),
                    n as i32,
                    (m * n) as i64,
                    b as i32,
                )
                .map_err(|e| RocmError::InternalError(&format!("rocBLAS GEMM failed: {:?}", e)))?;
            }

            S::F16(out)
        }
        (S::BF16(lhs), S::BF16(rhs_slice)) => {
            let lhs_slice = &lhs.slice(lhs_l.start_offset()..);
            let rhs_slice_data = &rhs_slice.slice(rhs_l.start_offset()..);
            let mut out = unsafe { dev.hip_device().alloc::<bf16>(elem_count)? };

            let alpha = bf16::ONE;
            let beta = bf16::ZERO;

            unsafe {
                rocm_rs::rocblas::level3::gemm_strided_batched(
                    &handle,
                    Operation::None,
                    Operation::None,
                    n as i32,
                    m as i32,
                    k as i32,
                    &alpha,
                    rhs_slice_data.as_ptr(),
                    n as i32,
                    (n * k) as i64,
                    lhs_slice.as_ptr(),
                    k as i32,
                    (m * k) as i64,
                    &beta,
                    out.as_mut_ptr(),
                    n as i32,
                    (m * n) as i64,
                    b as i32,
                )
                .map_err(|e| RocmError::InternalError(&format!("rocBLAS GEMM failed: {:?}", e)))?;
            }

            S::BF16(out)
        }
        _ => return Err(RocmError::InternalError("dtype mismatch in matmul").into()),
    };

    let device = dev.clone();
    Ok(crate::rocm_backend::RocmStorage { slice, device })
}
