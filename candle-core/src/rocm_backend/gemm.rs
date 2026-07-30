//! rocBLAS GEMM plumbing behind [`BackendStorage::matmul`].
//!
//! Split out of `mod.rs`, which is far over the workspace 400-line cap.
//!
//! `gemm_strided_batched_ex` is declared by hand: rocm-rs 0.5 exposes no safe
//! wrapper for the `_ex` entry point that f16/bf16 need.
//!
//! [`gemm_config`] mirrors `cuda_backend`'s, including its handling of a zero
//! batch stride — which candle produces whenever an operand is broadcast over
//! the batch (`broadcast_matmul`, and the col2im path in
//! [`super::ops_conv`]). rocBLAS handles those correctly; `cpu_backend`'s
//! `MatMul` does not, see `tests_conv::conv_transpose1d_covers_both_paths`.

use rocm_rs::rocblas::{self, level3::GemmStridedBatchedType, types::Operation};

use super::{RocmError, RocmStorage, RocmStorageSlice};
use crate::{Layout, Result};
use half::{bf16, f16};

struct GemmConfig<T> {
    alpha: T,
    beta: T,
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldc: i32,
    transa: Operation,
    transb: Operation,
}

struct StridedBatchedConfig<T> {
    batch_size: i32,
    gemm: GemmConfig<T>,
    stride_a: i64,
    stride_b: i64,
    stride_c: i64,
}

fn gemm_config<T: Copy>(
    alpha: T,
    beta: T,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_l: &Layout,
    rhs_l: &Layout,
) -> std::result::Result<StridedBatchedConfig<T>, RocmError> {
    let lhs_stride = lhs_l.stride();
    let rhs_stride = rhs_l.stride();
    let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
    let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
    let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
    let lhs_m2 = lhs_stride[lhs_stride.len() - 2];

    let (lda, transa) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
        (n as i32, Operation::None)
    } else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
        (k as i32, Operation::Transpose)
    } else {
        return Err(RocmError::MatMulNonContiguous {
            lhs_stride: Box::new(lhs_l.clone()),
            rhs_stride: Box::new(rhs_l.clone()),
            mnk: (m, n, k),
        });
    };

    let (ldb, transb) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
        (k as i32, Operation::None)
    } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
        (m as i32, Operation::Transpose)
    } else {
        return Err(RocmError::MatMulNonContiguous {
            lhs_stride: Box::new(lhs_l.clone()),
            rhs_stride: Box::new(rhs_l.clone()),
            mnk: (m, n, k),
        });
    };

    let gemm = GemmConfig {
        alpha,
        beta,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        lda,
        ldb,
        ldc: n as i32,
        transa,
        transb,
    };

    let stride_b: usize = match lhs_stride[..lhs_stride.len() - 2] {
        [s1, stride] if s1 == stride * lhs_l.dims()[1] => stride,
        [_, stride] if lhs_l.dims()[0] == 1 => stride,
        [stride, _] if lhs_l.dims()[1] == 1 => stride,
        [stride] => stride,
        [] => m * k,
        _ => {
            return Err(RocmError::MatMulNonContiguous {
                lhs_stride: Box::new(lhs_l.clone()),
                rhs_stride: Box::new(rhs_l.clone()),
                mnk: (m, n, k),
            })
        }
    };
    let stride_a: usize = match rhs_stride[..rhs_stride.len() - 2] {
        [s1, stride] if s1 == stride * rhs_l.dims()[1] => stride,
        [_, stride] if rhs_l.dims()[0] == 1 => stride,
        [stride, _] if rhs_l.dims()[1] == 1 => stride,
        [stride] => stride,
        [] => n * k,
        _ => {
            return Err(RocmError::MatMulNonContiguous {
                lhs_stride: Box::new(lhs_l.clone()),
                rhs_stride: Box::new(rhs_l.clone()),
                mnk: (m, n, k),
            })
        }
    };
    Ok(StridedBatchedConfig {
        batch_size: b as i32,
        gemm,
        stride_a: stride_a as i64,
        stride_b: stride_b as i64,
        stride_c: (m * n) as i64,
    })
}

unsafe fn gemm_strided_batched<T: GemmStridedBatchedType>(
    blas: &rocblas::Handle,
    cfg: StridedBatchedConfig<T>,
    a: *const std::ffi::c_void,
    b: *const std::ffi::c_void,
    c: *mut std::ffi::c_void,
) -> std::result::Result<(), RocmError> {
    rocblas::gemm_strided_batched(
        blas,
        cfg.gemm.transa,
        cfg.gemm.transb,
        cfg.gemm.m,
        cfg.gemm.n,
        cfg.gemm.k,
        &cfg.gemm.alpha,
        a as *const T,
        cfg.gemm.lda,
        cfg.stride_a,
        b as *const T,
        cfg.gemm.ldb,
        cfg.stride_b,
        &cfg.gemm.beta,
        c as *mut T,
        cfg.gemm.ldc,
        cfg.stride_c,
        cfg.batch_size,
    )
    .map_err(|e| RocmError::Rocblas(e.to_string()))
}

struct GemmExConfig {
    alpha: f32,
    beta: f32,
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldc: i32,
    transa: Operation,
    transb: Operation,
}

struct StridedBatchedExConfig {
    batch_size: i32,
    gemm: GemmExConfig,
    stride_a: i64,
    stride_b: i64,
    stride_c: i64,
}

fn gemm_ex_config(
    alpha: f32,
    beta: f32,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_l: &Layout,
    rhs_l: &Layout,
) -> std::result::Result<StridedBatchedExConfig, RocmError> {
    let inner = gemm_config(alpha, beta, (b, m, n, k), lhs_l, rhs_l)?;
    Ok(StridedBatchedExConfig {
        batch_size: inner.batch_size,
        gemm: GemmExConfig {
            alpha: inner.gemm.alpha,
            beta: inner.gemm.beta,
            m: inner.gemm.m,
            n: inner.gemm.n,
            k: inner.gemm.k,
            lda: inner.gemm.lda,
            ldb: inner.gemm.ldb,
            ldc: inner.gemm.ldc,
            transa: inner.gemm.transa,
            transb: inner.gemm.transb,
        },
        stride_a: inner.stride_a,
        stride_b: inner.stride_b,
        stride_c: inner.stride_c,
    })
}

unsafe fn gemm_strided_batched_ex(
    blas: &rocblas::Handle,
    cfg: StridedBatchedExConfig,
    a: *const std::ffi::c_void,
    b: *const std::ffi::c_void,
    c: *mut std::ffi::c_void,
    datatype: rocm_rs::rocblas::ffi::rocblas_datatype,
) -> std::result::Result<(), RocmError> {
    use rocm_rs::rocblas::ffi;
    use rocm_rs::rocblas::utils::GemmAlgo;

    let status = unsafe {
        rocblas_gemm_strided_batched_ex(
            blas.as_raw(),
            cfg.gemm.transa.into(),
            cfg.gemm.transb.into(),
            cfg.gemm.m,
            cfg.gemm.n,
            cfg.gemm.k,
            &cfg.gemm.alpha as *const f32 as *const std::ffi::c_void,
            a,
            datatype,
            cfg.gemm.lda,
            cfg.stride_a,
            b,
            datatype,
            cfg.gemm.ldb,
            cfg.stride_b,
            &cfg.gemm.beta as *const f32 as *const std::ffi::c_void,
            c,
            datatype,
            cfg.gemm.ldc,
            cfg.stride_c,
            c,
            datatype,
            cfg.gemm.ldc,
            cfg.stride_c,
            cfg.batch_size,
            ffi::rocblas_datatype__rocblas_datatype_f32_r,
            GemmAlgo::Standard.into(),
            0,
            0,
        )
    };
    if status != ffi::rocblas_status__rocblas_status_success {
        return Err(RocmError::Rocblas(format!(
            "rocblas_gemm_strided_batched_ex failed with status {}",
            status
        )));
    }
    Ok(())
}

extern "C" {
    fn rocblas_gemm_strided_batched_ex(
        handle: rocm_rs::rocblas::ffi::rocblas_handle,
        transA: rocm_rs::rocblas::ffi::rocblas_operation,
        transB: rocm_rs::rocblas::ffi::rocblas_operation,
        m: rocm_rs::rocblas::ffi::rocblas_int,
        n: rocm_rs::rocblas::ffi::rocblas_int,
        k: rocm_rs::rocblas::ffi::rocblas_int,
        alpha: *const std::ffi::c_void,
        a: *const std::ffi::c_void,
        a_type: rocm_rs::rocblas::ffi::rocblas_datatype,
        lda: rocm_rs::rocblas::ffi::rocblas_int,
        stride_a: rocm_rs::rocblas::ffi::rocblas_stride,
        b: *const std::ffi::c_void,
        b_type: rocm_rs::rocblas::ffi::rocblas_datatype,
        ldb: rocm_rs::rocblas::ffi::rocblas_int,
        stride_b: rocm_rs::rocblas::ffi::rocblas_stride,
        beta: *const std::ffi::c_void,
        c: *const std::ffi::c_void,
        c_type: rocm_rs::rocblas::ffi::rocblas_datatype,
        ldc: rocm_rs::rocblas::ffi::rocblas_int,
        stride_c: rocm_rs::rocblas::ffi::rocblas_stride,
        d: *mut std::ffi::c_void,
        d_type: rocm_rs::rocblas::ffi::rocblas_datatype,
        ldd: rocm_rs::rocblas::ffi::rocblas_int,
        stride_d: rocm_rs::rocblas::ffi::rocblas_stride,
        batch_count: rocm_rs::rocblas::ffi::rocblas_int,
        compute_type: rocm_rs::rocblas::ffi::rocblas_datatype,
        algo: rocm_rs::rocblas::ffi::rocblas_gemm_algo,
        solution_index: i32,
        flags: u32,
    ) -> rocm_rs::rocblas::ffi::rocblas_status;
}

macro_rules! dispatch_matmul {
    ($self:expr, $rhs:expr, $b:expr, $m:expr, $n:expr, $k:expr, $lhs_l:expr, $rhs_l:expr, $dev:expr,
     $(($variant:ident, $rust_ty:ty, $alpha:expr, $zero:expr, $cfg_fn:expr, $gemm_fn:expr $(, $ex_datatype:expr)?)),+ $(,)?) => {{
        let elem_count = $b * $m * $n;
        let lhs_ptr = unsafe { $self.slice.offset_ptr($lhs_l.start_offset()) };
        let rhs_ptr = unsafe { $rhs.slice.offset_ptr($rhs_l.start_offset()) };
        let device = $dev.clone();
        let slice = match (&$self.slice, &$rhs.slice) {
            $(
                (RocmStorageSlice::$variant(_), RocmStorageSlice::$variant(_)) => {
                    let cfg = $cfg_fn($alpha, $zero, ($b, $m, $n, $k), $lhs_l, $rhs_l)?;
                    let out = $dev.alloc::<$rust_ty>(elem_count)?;
                    unsafe { $gemm_fn(&$dev.blas, cfg, rhs_ptr, lhs_ptr, out.as_ptr() $(, $ex_datatype)?)?; }
                    RocmStorageSlice::$variant(out)
                }
            )+
            _ => return Err(RocmError::Internal("dtype mismatch in matmul".into()).into()),
        };
        Ok(RocmStorage { slice, device })
    }};
}

pub(super) fn matmul(
    lhs: &RocmStorage,
    rhs: &RocmStorage,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_l: &Layout,
    rhs_l: &Layout,
) -> Result<RocmStorage> {
    use rocm_rs::rocblas::ffi;
    lhs.device.bind()?;
    dispatch_matmul!(
        lhs,
        rhs,
        b,
        m,
        n,
        k,
        lhs_l,
        rhs_l,
        &lhs.device,
        (F32, f32, 1.0f32, 0.0f32, gemm_config, gemm_strided_batched),
        (F64, f64, 1.0f64, 0.0f64, gemm_config, gemm_strided_batched),
        (
            F16,
            f16,
            1.0f32,
            0.0f32,
            gemm_ex_config,
            gemm_strided_batched_ex,
            ffi::rocblas_datatype__rocblas_datatype_f16_r
        ),
        (
            BF16,
            bf16,
            1.0f32,
            0.0f32,
            gemm_ex_config,
            gemm_strided_batched_ex,
            ffi::rocblas_datatype__rocblas_datatype_bf16_r
        ),
    )
}
