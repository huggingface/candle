//! Implementation of Backend traits for ROCm device
//!
use crate::backend::BackendStorage;
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result};
pub use candle_rocm_kernels as kernels;
use half::{bf16, f16};
pub use rocm_rs;
use rocm_rs::hip::bindings;
use rocm_rs::rocblas::{self, level3::GemmStridedBatchedType, types::Operation};

mod device;
mod error;
mod miopen;
mod ops_elementwise;
mod ops_reduce;
mod ops_scalar;
#[cfg(test)]
mod tests;
mod wrappers;
pub use device::{DeviceId, RocmDevice};
pub use error::{RocmError, WrapErr};
pub use wrappers::SendSyncDeviceMemory;
pub mod utils;
pub use utils::{Map1, Map1Any, Map2, Map2Any, Map2InPlace, Map3, S};

use ops_elementwise::{CloneBuffer, Cmp, WhereCond};
use ops_reduce::{index_select_typed, FastReduce};
pub(crate) use ops_scalar::Affine;
use ops_scalar::{Elu, Powf};

pub enum RocmStorageSlice {
    U8(SendSyncDeviceMemory<u8>),
    U32(SendSyncDeviceMemory<u32>),
    I16(SendSyncDeviceMemory<i16>),
    I32(SendSyncDeviceMemory<i32>),
    I64(SendSyncDeviceMemory<i64>),
    BF16(SendSyncDeviceMemory<bf16>),
    F16(SendSyncDeviceMemory<f16>),
    F32(SendSyncDeviceMemory<f32>),
    F64(SendSyncDeviceMemory<f64>),
    F8E4M3(SendSyncDeviceMemory<u8>),
}

/// `RocmStorageSlice::F8E4M3` stores its payload as `u8`, so every byte-view
/// shortcut in this backend (and in `device.rs`) is only correct while F8E4M3 is
/// exactly one byte wide.
const _: () = assert!(std::mem::size_of::<float8::F8E4M3>() == 1);

impl std::fmt::Debug for RocmStorageSlice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RocmStorageSlice::U8(m) => write!(f, "U8({} bytes)", m.size()),
            RocmStorageSlice::U32(m) => write!(f, "U32({} bytes)", m.size()),
            RocmStorageSlice::I16(m) => write!(f, "I16({} bytes)", m.size()),
            RocmStorageSlice::I32(m) => write!(f, "I32({} bytes)", m.size()),
            RocmStorageSlice::I64(m) => write!(f, "I64({} bytes)", m.size()),
            RocmStorageSlice::BF16(m) => write!(f, "BF16({} bytes)", m.size()),
            RocmStorageSlice::F16(m) => write!(f, "F16({} bytes)", m.size()),
            RocmStorageSlice::F32(m) => write!(f, "F32({} bytes)", m.size()),
            RocmStorageSlice::F64(m) => write!(f, "F64({} bytes)", m.size()),
            RocmStorageSlice::F8E4M3(m) => write!(f, "F8E4M3({} bytes)", m.size()),
        }
    }
}

impl RocmStorageSlice {
    pub fn dtype(&self) -> DType {
        match self {
            RocmStorageSlice::U8(_) => DType::U8,
            RocmStorageSlice::U32(_) => DType::U32,
            RocmStorageSlice::I16(_) => DType::I16,
            RocmStorageSlice::I32(_) => DType::I32,
            RocmStorageSlice::I64(_) => DType::I64,
            RocmStorageSlice::BF16(_) => DType::BF16,
            RocmStorageSlice::F16(_) => DType::F16,
            RocmStorageSlice::F32(_) => DType::F32,
            RocmStorageSlice::F64(_) => DType::F64,
            RocmStorageSlice::F8E4M3(_) => DType::F8E4M3,
        }
    }

    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        match self {
            RocmStorageSlice::U8(m) => m.as_ptr(),
            RocmStorageSlice::U32(m) => m.as_ptr(),
            RocmStorageSlice::I16(m) => m.as_ptr(),
            RocmStorageSlice::I32(m) => m.as_ptr(),
            RocmStorageSlice::I64(m) => m.as_ptr(),
            RocmStorageSlice::BF16(m) => m.as_ptr(),
            RocmStorageSlice::F16(m) => m.as_ptr(),
            RocmStorageSlice::F32(m) => m.as_ptr(),
            RocmStorageSlice::F64(m) => m.as_ptr(),
            RocmStorageSlice::F8E4M3(m) => m.as_ptr(),
        }
    }

    fn elem_size(&self) -> usize {
        match self {
            RocmStorageSlice::U8(_) | RocmStorageSlice::F8E4M3(_) => 1,
            RocmStorageSlice::I16(_) | RocmStorageSlice::BF16(_) | RocmStorageSlice::F16(_) => 2,
            RocmStorageSlice::U32(_) | RocmStorageSlice::I32(_) | RocmStorageSlice::F32(_) => 4,
            RocmStorageSlice::I64(_) | RocmStorageSlice::F64(_) => 8,
        }
    }

    unsafe fn offset_ptr(&self, offset: usize) -> *mut std::ffi::c_void {
        self.as_ptr().add(offset * self.elem_size())
    }
}

pub struct RocmStorage {
    pub slice: RocmStorageSlice,
    pub device: RocmDevice,
}

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
            lhs_stride: lhs_l.clone(),
            rhs_stride: rhs_l.clone(),
            mnk: (m, n, k),
        });
    };

    let (ldb, transb) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
        (k as i32, Operation::None)
    } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
        (m as i32, Operation::Transpose)
    } else {
        return Err(RocmError::MatMulNonContiguous {
            lhs_stride: lhs_l.clone(),
            rhs_stride: rhs_l.clone(),
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
                lhs_stride: lhs_l.clone(),
                rhs_stride: rhs_l.clone(),
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
                lhs_stride: lhs_l.clone(),
                rhs_stride: rhs_l.clone(),
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
        Ok(Self { slice, device })
    }};
}

macro_rules! dispatch_miopen_conv {
    ($self:expr, $kernel:expr, $l:expr, $kernel_l:expr, $dst_el:expr, $device:expr, $handle:expr, $func:ident, $($arg:expr),* $(,)?) => {{
        let device = $device.clone();
        let slice = match (&$self.slice, &$kernel.slice) {
            (RocmStorageSlice::F32(s), RocmStorageSlice::F32(w)) => {
                let x_ptr = unsafe { s.ptr_at($l.start_offset()) } as *mut _;
                let w_ptr = unsafe { w.ptr_at($kernel_l.start_offset()) } as *mut _;
                let o = device.alloc_zeros::<f32>($dst_el)?;
                $func::<f32>($handle, x_ptr, w_ptr, o.as_ptr() as *mut _, $($arg),*)?;
                RocmStorageSlice::F32(o)
            }
            (RocmStorageSlice::F16(s), RocmStorageSlice::F16(w)) => {
                let x_ptr = unsafe { s.ptr_at($l.start_offset()) } as *mut _;
                let w_ptr = unsafe { w.ptr_at($kernel_l.start_offset()) } as *mut _;
                let o = device.alloc_zeros::<f16>($dst_el)?;
                $func::<f16>($handle, x_ptr, w_ptr, o.as_ptr() as *mut _, $($arg),*)?;
                RocmStorageSlice::F16(o)
            }
            (RocmStorageSlice::BF16(s), RocmStorageSlice::BF16(w)) => {
                let x_ptr = unsafe { s.ptr_at($l.start_offset()) } as *mut _;
                let w_ptr = unsafe { w.ptr_at($kernel_l.start_offset()) } as *mut _;
                let o = device.alloc_zeros::<bf16>($dst_el)?;
                $func::<bf16>($handle, x_ptr, w_ptr, o.as_ptr() as *mut _, $($arg),*)?;
                RocmStorageSlice::BF16(o)
            }
            (RocmStorageSlice::F64(s), RocmStorageSlice::F64(w)) => {
                let x_ptr = unsafe { s.ptr_at($l.start_offset()) } as *mut _;
                let w_ptr = unsafe { w.ptr_at($kernel_l.start_offset()) } as *mut _;
                let o = device.alloc_zeros::<f64>($dst_el)?;
                $func::<f64>($handle, x_ptr, w_ptr, o.as_ptr() as *mut _, $($arg),*)?;
                RocmStorageSlice::F64(o)
            }
            _ => return Err(crate::Error::Msg(
                "conv only supports f32, f16, bf16, f64 for ROCm".to_string(),
            )),
        };
        Ok(Self { slice, device })
    }};
}

macro_rules! cast_launch {
    ($dev:expr, $grid:expr, $block:expr, $el:expr, $dims_len:expr, $ds_ptr:expr, $src_ptr:expr, $src_dtype:expr, $rust_type:ty, $variant:ident) => {{
        let out = $dev.alloc::<$rust_type>($el)?;
        let out_ptr = out.as_ptr() as *mut std::ffi::c_void;
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
                    (&$src_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                ],
            )?;
        }
        RocmStorageSlice::$variant(out)
    }};
}

/// Kernel-name suffix candle-kernels uses for the Rust type `T`.
///
/// `bf16` has to be probed before `f16` — `half::bf16`'s type name contains
/// both.
fn dtype_suffix<T: Copy + Send + Sync + 'static>() -> Option<&'static str> {
    let type_name = std::any::type_name::<T>();
    let suffix = if type_name.contains("f32") {
        "f32"
    } else if type_name.contains("f64") {
        "f64"
    } else if type_name.contains("u8") {
        "u8"
    } else if type_name.contains("u32") {
        "u32"
    } else if type_name.contains("i64") {
        "i64"
    } else if type_name.contains("bf16") {
        "bf16"
    } else if type_name.contains("f16") {
        "f16"
    } else if type_name.contains("i16") {
        "i16"
    } else if type_name.contains("i32") {
        "i32"
    } else {
        return None;
    };
    Some(suffix)
}

/// Name of the `kernel` variant compiled for `T`, or an error when candle
/// ships no kernels for that dtype.
pub fn try_kernel_name<T: Copy + Send + Sync + 'static>(kernel: &str) -> Result<String> {
    match dtype_suffix::<T>() {
        Some(suffix) => Ok(format!("{}_{}", kernel, suffix)),
        None => Err(RocmError::Internal(format!(
            "unsupported dtype {} for kernel {}",
            std::any::type_name::<T>(),
            kernel
        ))
        .into()),
    }
}

/// Infallible variant kept for the public API used by candle-nn.
///
/// An unsupported dtype yields a name no module defines, so the caller gets a
/// "kernel not found" error at launch instead of aborting the process.
pub fn kernel_name<T: Copy + Send + Sync + 'static>(kernel: &str) -> String {
    let suffix = dtype_suffix::<T>().unwrap_or("unsupported_dtype");
    format!("{}_{}", kernel, suffix)
}

/// Grid and block dimensions for an elementwise kernel over `num_elems`.
///
/// The grid is capped at 65535 blocks, so for large inputs it launches *fewer*
/// threads than there are elements. That is only sound for kernels written as a
/// grid-stride loop, which every elementwise kernel in `candle-kernels` is:
///
/// ```text
/// for (i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x)
/// ```
///
/// A kernel that instead maps one thread to one element and returns early would
/// silently leave the tail of its output untouched. Size the grid by hand for
/// those — `rope` in `reduce.cu` is one such kernel.
pub fn launch_config(num_elems: usize) -> (rocm_rs::hip::Dim3, rocm_rs::hip::Dim3) {
    const BLOCK_SIZE: u32 = 256;
    let num_blocks = (num_elems as u32).div_ceil(BLOCK_SIZE);
    let grid_dim = num_blocks.min(65535);
    (
        rocm_rs::hip::Dim3::from(grid_dim),
        rocm_rs::hip::Dim3::from(BLOCK_SIZE),
    )
}

unsafe fn launch_kernel(
    dev: &RocmDevice,
    module: &kernels::Module,
    func_name: &str,
    grid: rocm_rs::hip::Dim3,
    block: rocm_rs::hip::Dim3,
    args: &mut [*mut std::ffi::c_void],
) -> Result<()> {
    let kernel_manager = dev
        .kernel_manager()
        .lock()
        .map_err(|_| crate::Error::Msg("Failed to lock kernel manager".to_string()))?;
    let module = kernel_manager
        .get_or_load(module)
        .map_err(|e| crate::Error::Msg(e.to_string()))?;
    let kernel = module
        .get_function(func_name)
        .map_err(|e| crate::Error::Msg(format!("Kernel {} not found: {}", func_name, e)))?;
    kernel
        .launch(grid, block, 0, Some(&dev.stream), args)
        .map_err(|e| crate::Error::Msg(format!("Kernel launch failed: {}", e)))
}

fn dims_and_strides(
    dev: &RocmDevice,
    layout: &Layout,
    n_strides: usize,
) -> Result<Option<SendSyncDeviceMemory<usize>>> {
    if layout.is_contiguous() {
        return Ok(None);
    }
    let dims = layout.shape().dims();
    let strides = layout.stride();
    let mut data = Vec::with_capacity(dims.len() + n_strides * dims.len());
    for &d in dims {
        data.push(d as usize);
    }
    for _ in 0..n_strides {
        for &s in strides {
            data.push(s as usize);
        }
    }
    Ok(Some(dev.clone_htod(&data)?))
}

fn dims_and_strides_pair(
    dev: &RocmDevice,
    l1: &Layout,
    l2: &Layout,
) -> Result<Option<SendSyncDeviceMemory<usize>>> {
    if l1.is_contiguous() && l2.is_contiguous() {
        return Ok(None);
    }
    let dims = l1.shape().dims();
    let mut data = Vec::with_capacity(dims.len() * 3);
    for &d in dims {
        data.push(d as usize);
    }
    for &s in l1.stride() {
        data.push(s as usize);
    }
    for &s in l2.stride() {
        data.push(s as usize);
    }
    Ok(Some(dev.clone_htod(&data)?))
}

impl std::fmt::Debug for RocmStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RocmStorage {{ slice: {:?}, device: {:?} }}",
            self.slice, self.device
        )
    }
}

impl BackendStorage for RocmStorage {
    type Device = RocmDevice;

    fn try_clone(&self, layout: &Layout) -> Result<Self> {
        let device = self.device.clone();
        let slice = match &self.slice {
            // `Map1` refuses F8E4M3 (it shares the u8 storage and would resolve
            // to the u8 kernels), but a raw buffer copy is dtype agnostic.
            RocmStorageSlice::F8E4M3(s) => {
                RocmStorageSlice::F8E4M3(CloneBuffer.f(s, &device, layout)?)
            }
            slice => CloneBuffer.map(slice, &device, layout)?,
        };
        Ok(Self { slice, device })
    }

    fn dtype(&self) -> DType {
        self.slice.dtype()
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        match &self.slice {
            RocmStorageSlice::U8(s) => Ok(CpuStorage::U8(self.device.clone_dtoh(s)?.into())),
            RocmStorageSlice::U32(s) => Ok(CpuStorage::U32(self.device.clone_dtoh(s)?.into())),
            RocmStorageSlice::I16(s) => Ok(CpuStorage::I16(self.device.clone_dtoh(s)?.into())),
            RocmStorageSlice::I32(s) => Ok(CpuStorage::I32(self.device.clone_dtoh(s)?.into())),
            RocmStorageSlice::I64(s) => Ok(CpuStorage::I64(self.device.clone_dtoh(s)?.into())),
            RocmStorageSlice::BF16(s) => Ok(CpuStorage::BF16(self.device.clone_dtoh(s)?.into())),
            RocmStorageSlice::F16(s) => Ok(CpuStorage::F16(self.device.clone_dtoh(s)?.into())),
            RocmStorageSlice::F32(s) => Ok(CpuStorage::F32(self.device.clone_dtoh(s)?.into())),
            RocmStorageSlice::F64(s) => Ok(CpuStorage::F64(self.device.clone_dtoh(s)?.into())),
            RocmStorageSlice::F8E4M3(s) => {
                let bytes = self.device.clone_dtoh(s)?;
                let v: Vec<float8::F8E4M3> =
                    bytes.into_iter().map(float8::F8E4M3::from_bits).collect();
                Ok(CpuStorage::F8E4M3(v.into()))
            }
        }
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        let device = self.device.clone();
        let slice = Affine(mul, add).map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        let device = self.device.clone();
        let slice = Powf(e).map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        let device = self.device.clone();
        let slice = Elu(alpha).map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    fn reduce_op(&self, op: ReduceOp, l: &Layout, sum_dims: &[usize]) -> Result<Self> {
        let device = self.device.clone();
        let slice = FastReduce(sum_dims, op).map(&self.slice, &device, l)?;
        Ok(Self { slice, device })
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, l1: &Layout, l2: &Layout) -> Result<Self> {
        let device = self.device.clone();
        let slice = Cmp(op).map(&self.slice, l1, &rhs.slice, l2, &device)?;
        Ok(Self { slice, device })
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        let shape = layout.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let dev = self.device.clone();

        let ds = dims_and_strides(&dev, layout, 1)?;
        let start_o = layout.start_offset();
        let src_ptr = unsafe { self.slice.offset_ptr(start_o) };

        let (grid, block) = launch_config(el);
        let ds_ptr: *const usize = ds
            .as_ref()
            .map(|d| d.as_ptr() as *const usize)
            .unwrap_or(std::ptr::null());

        let src_dtype = self.slice.dtype();
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
            DType::U32 => cast_launch!(
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
            ),
            DType::I64 => cast_launch!(
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
            ),
            DType::BF16 => cast_launch!(
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
            ),
            DType::F16 => cast_launch!(
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
            ),
            DType::F32 => cast_launch!(
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
            ),
            DType::F64 => cast_launch!(
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
            ),
            DType::I16 | DType::I32 => {
                return Err(crate::Error::Msg(
                    "i16/i32 dtypes are not supported for to_dtype on ROCm".to_string(),
                ))
            }
            DType::F8E4M3 | DType::F4 | DType::F6E2M3 | DType::F6E3M2 | DType::F8E8M0 => {
                return Err(crate::Error::Msg(format!(
                    "{:?} dtype is not supported for to_dtype on ROCm",
                    dtype
                )))
            }
        };

        Ok(Self { slice, device: dev })
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        let device = self.device.clone();
        let slice = B::V.map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    fn binary_impl<B: BinaryOpT>(&self, rhs: &Self, l1: &Layout, l2: &Layout) -> Result<Self> {
        let device = self.device.clone();
        let slice = B::V.map(&self.slice, l1, &rhs.slice, l2, &device)?;
        Ok(Self { slice, device })
    }

    fn where_cond(&self, l: &Layout, t: &Self, lt: &Layout, f: &Self, lf: &Layout) -> Result<Self> {
        let device = self.device.clone();
        let slice = WhereCond(self, l).map(&t.slice, lt, &f.slice, lf, &device)?;
        Ok(Self { slice, device })
    }

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        use crate::rocm_backend::miopen::conv2d_forward;

        let device = self.device();
        let miopen_handle = device.miopen();
        let dst_el = params.b_size * params.c_out * params.l_out();

        dispatch_miopen_conv!(
            self,
            kernel,
            l,
            kernel_l,
            dst_el,
            device,
            &miopen_handle.0,
            conv2d_forward,
            params.b_size,
            params.c_in,
            params.c_out,
            1,
            params.l_in,
            1,
            params.k_size,
            1,
            params.l_out(),
            params.padding,
            0,
            params.stride,
            1,
            params.dilation,
            1,
        )
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        use crate::rocm_backend::miopen::conv_transpose1d_forward;

        let device = self.device();
        let miopen_handle = device.miopen();
        let dst_el = params.b_size * params.c_out * params.l_out();

        dispatch_miopen_conv!(
            self,
            kernel,
            l,
            kernel_l,
            dst_el,
            device,
            &miopen_handle.0,
            conv_transpose1d_forward,
            params.b_size,
            params.c_in,
            params.c_out,
            params.l_in,
            params.k_size,
            params.l_out(),
            params.padding,
            params.output_padding,
            params.stride,
            params.dilation,
        )
    }

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        use crate::rocm_backend::miopen::conv2d_forward;

        let device = self.device();
        let miopen_handle = device.miopen();
        let out_h = params.out_h();
        let out_w = params.out_w();
        let dst_el = params.b_size * params.c_out * out_h * out_w;

        dispatch_miopen_conv!(
            self,
            kernel,
            l,
            kernel_l,
            dst_el,
            device,
            &miopen_handle.0,
            conv2d_forward,
            params.b_size,
            params.c_in,
            params.c_out,
            params.i_h,
            params.i_w,
            params.k_h,
            params.k_w,
            out_h,
            out_w,
            params.padding,
            params.padding,
            params.stride,
            params.stride,
            params.dilation,
            params.dilation,
        )
    }

    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kl: &Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        Err(crate::Error::Msg(
            "conv_transpose2d not yet implemented for ROCm".to_string(),
        ))
    }

    fn avg_pool2d(&self, _l: &Layout, _k: (usize, usize), _s: (usize, usize)) -> Result<Self> {
        Err(crate::Error::Msg(
            "avg_pool2d not yet implemented for ROCm".to_string(),
        ))
    }

    fn max_pool2d(&self, _l: &Layout, _k: (usize, usize), _s: (usize, usize)) -> Result<Self> {
        Err(crate::Error::Msg(
            "max_pool2d not yet implemented for ROCm".to_string(),
        ))
    }

    fn upsample_nearest1d(&self, _l: &Layout, _sz: usize) -> Result<Self> {
        Err(crate::Error::Msg(
            "upsample_nearest1d not yet implemented for ROCm".to_string(),
        ))
    }

    fn upsample_nearest2d(&self, _l: &Layout, _w: usize, _h: usize) -> Result<Self> {
        Err(crate::Error::Msg(
            "upsample_nearest2d not yet implemented for ROCm".to_string(),
        ))
    }

    fn upsample_bilinear2d(
        &self,
        _l: &Layout,
        _w: usize,
        _h: usize,
        _align: bool,
        _fh: Option<f64>,
        _fv: Option<f64>,
    ) -> Result<Self> {
        Err(crate::Error::Msg(
            "upsample_bilinear2d not yet implemented for ROCm".to_string(),
        ))
    }

    fn gather(&self, _l: &Layout, _idx: &Self, _il: &Layout, _dim: usize) -> Result<Self> {
        Err(crate::Error::Msg(
            "gather not yet implemented for ROCm".to_string(),
        ))
    }

    fn scatter_set(
        &mut self,
        _l: &Layout,
        _val: &Self,
        _vl: &Layout,
        _idx: &Self,
        _il: &Layout,
        _dim: usize,
    ) -> Result<()> {
        Err(crate::Error::Msg(
            "scatter_set not yet implemented for ROCm".to_string(),
        ))
    }

    fn scatter_add_set(
        &mut self,
        _l: &Layout,
        _val: &Self,
        _vl: &Layout,
        _idx: &Self,
        _il: &Layout,
        _dim: usize,
    ) -> Result<()> {
        Err(crate::Error::Msg(
            "scatter_add_set not yet implemented for ROCm".to_string(),
        ))
    }

    fn index_select(&self, idx: &Self, src_l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        let device = self.device.clone();
        let left_size: usize = src_l.dims()[..dim].iter().product();
        let right_size: usize = src_l.dims()[dim + 1..].iter().product();
        let src_dim_size = src_l.dims()[dim];
        let ids_dim_size = ids_l.shape().elem_count();
        let dst_el = ids_dim_size * left_size * right_size;

        let ids_dims = ids_l.shape().dims();
        let ds = device.clone_htod(&[ids_dims, ids_l.stride()].concat())?;

        let src_ptr = match src_l.contiguous_offsets() {
            Some((o1, _)) => unsafe { self.slice.offset_ptr(o1) },
            None => Err(crate::Error::RequiresContiguous { op: "index-select" }.bt())?,
        };

        let (ids_prefix, ids_ptr) = match &idx.slice {
            RocmStorageSlice::U32(s) => ("is_u32", unsafe { s.ptr_at(ids_l.start_offset()) }
                as *mut std::ffi::c_void),
            RocmStorageSlice::U8(s) => ("is_u8", unsafe { s.ptr_at(ids_l.start_offset()) }
                as *mut std::ffi::c_void),
            RocmStorageSlice::I64(s) => ("is_i64", unsafe { s.ptr_at(ids_l.start_offset()) }
                as *mut std::ffi::c_void),
            _ => crate::bail!("index_select ids should be u8, u32, or i64"),
        };

        let slice = match &self.slice {
            RocmStorageSlice::F32(_) => RocmStorageSlice::F32(index_select_typed::<f32>(
                ids_prefix,
                ids_ptr,
                &ds,
                src_ptr,
                left_size,
                src_dim_size,
                ids_dim_size,
                right_size,
                dst_el,
                &device,
            )?),
            RocmStorageSlice::F64(_) => RocmStorageSlice::F64(index_select_typed::<f64>(
                ids_prefix,
                ids_ptr,
                &ds,
                src_ptr,
                left_size,
                src_dim_size,
                ids_dim_size,
                right_size,
                dst_el,
                &device,
            )?),
            RocmStorageSlice::U8(_) => RocmStorageSlice::U8(index_select_typed::<u8>(
                ids_prefix,
                ids_ptr,
                &ds,
                src_ptr,
                left_size,
                src_dim_size,
                ids_dim_size,
                right_size,
                dst_el,
                &device,
            )?),
            RocmStorageSlice::U32(_) => RocmStorageSlice::U32(index_select_typed::<u32>(
                ids_prefix,
                ids_ptr,
                &ds,
                src_ptr,
                left_size,
                src_dim_size,
                ids_dim_size,
                right_size,
                dst_el,
                &device,
            )?),
            RocmStorageSlice::I64(_) => RocmStorageSlice::I64(index_select_typed::<i64>(
                ids_prefix,
                ids_ptr,
                &ds,
                src_ptr,
                left_size,
                src_dim_size,
                ids_dim_size,
                right_size,
                dst_el,
                &device,
            )?),
            RocmStorageSlice::BF16(_) => RocmStorageSlice::BF16(index_select_typed::<half::bf16>(
                ids_prefix,
                ids_ptr,
                &ds,
                src_ptr,
                left_size,
                src_dim_size,
                ids_dim_size,
                right_size,
                dst_el,
                &device,
            )?),
            RocmStorageSlice::F16(_) => RocmStorageSlice::F16(index_select_typed::<half::f16>(
                ids_prefix,
                ids_ptr,
                &ds,
                src_ptr,
                left_size,
                src_dim_size,
                ids_dim_size,
                right_size,
                dst_el,
                &device,
            )?),
            RocmStorageSlice::I16(_) | RocmStorageSlice::I32(_) | RocmStorageSlice::F8E4M3(_) => {
                crate::bail!("index_select does not support this dtype for ROCm")
            }
        };
        Ok(Self { slice, device })
    }

    fn index_add(
        &self,
        _l: &Layout,
        _idx: &Self,
        _il: &Layout,
        _val: &Self,
        _vl: &Layout,
        _dim: usize,
    ) -> Result<Self> {
        Err(crate::Error::Msg(
            "index_add not yet implemented for ROCm".to_string(),
        ))
    }

    fn matmul(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        use rocm_rs::rocblas::ffi;
        dispatch_matmul!(
            self,
            rhs,
            b,
            m,
            n,
            k,
            lhs_l,
            rhs_l,
            &self.device,
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

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        let src_shape = src_l.shape();
        let dims = src_shape.dims();
        let el_count = src_shape.elem_count();
        if el_count == 0 {
            return Ok(());
        }

        if src_l.is_contiguous() {
            let (src_ptr, el_size) = match &self.slice {
                RocmStorageSlice::U8(s) => (s.as_ptr(), 1usize),
                RocmStorageSlice::U32(s) => (s.as_ptr(), 4),
                RocmStorageSlice::I16(s) => (s.as_ptr(), 2),
                RocmStorageSlice::I32(s) => (s.as_ptr(), 4),
                RocmStorageSlice::I64(s) => (s.as_ptr(), 8),
                RocmStorageSlice::BF16(s) => (s.as_ptr(), 2),
                RocmStorageSlice::F16(s) => (s.as_ptr(), 2),
                RocmStorageSlice::F32(s) => (s.as_ptr(), 4),
                RocmStorageSlice::F64(s) => (s.as_ptr(), 8),
                RocmStorageSlice::F8E4M3(s) => (s.as_ptr(), 1),
            };
            let (dst_ptr, _) = match &mut dst.slice {
                RocmStorageSlice::U8(s) => (s.as_ptr(), 1usize),
                RocmStorageSlice::U32(s) => (s.as_ptr(), 4),
                RocmStorageSlice::I16(s) => (s.as_ptr(), 2),
                RocmStorageSlice::I32(s) => (s.as_ptr(), 4),
                RocmStorageSlice::I64(s) => (s.as_ptr(), 8),
                RocmStorageSlice::BF16(s) => (s.as_ptr(), 2),
                RocmStorageSlice::F16(s) => (s.as_ptr(), 2),
                RocmStorageSlice::F32(s) => (s.as_ptr(), 4),
                RocmStorageSlice::F64(s) => (s.as_ptr(), 8),
                RocmStorageSlice::F8E4M3(s) => (s.as_ptr(), 1),
            };
            let src_ptr = unsafe { src_ptr.add(src_l.start_offset() * el_size) };
            let dst_ptr = unsafe { dst_ptr.add(dst_offset * el_size) };
            let byte_count = el_count * el_size;
            let result = unsafe {
                bindings::hipMemcpy(
                    dst_ptr,
                    src_ptr,
                    byte_count,
                    bindings::hipMemcpyKind_hipMemcpyDeviceToDevice,
                )
            };
            if result != bindings::hipError_t_hipSuccess {
                crate::bail!("hipMemcpy failed with error {}", result);
            }
            return Ok(());
        }

        let (grid, block) = launch_config(el_count);
        let ds = dims_and_strides(&self.device, src_l, 1)?;

        macro_rules! copy_strided {
            ($variant:ident, $suffix:expr, $ty:ty) => {{
                let (src_mem, dst_mem) = match (&self.slice, &mut dst.slice) {
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
                let ds_ptr: *const usize = ds
                    .as_ref()
                    .map(|d| d.as_ptr() as *const usize)
                    .unwrap_or(std::ptr::null());
                unsafe {
                    launch_kernel(
                        &self.device,
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

        match &self.slice {
            RocmStorageSlice::U8(_) => copy_strided!(U8, "u8", u8),
            RocmStorageSlice::U32(_) => copy_strided!(U32, "u32", u32),
            RocmStorageSlice::I16(_) => copy_strided!(I16, "i16", i16),
            RocmStorageSlice::I32(_) => copy_strided!(I32, "i32", i32),
            RocmStorageSlice::I64(_) => copy_strided!(I64, "i64", i64),
            RocmStorageSlice::BF16(_) => copy_strided!(BF16, "bf16", bf16),
            RocmStorageSlice::F16(_) => copy_strided!(F16, "f16", f16),
            RocmStorageSlice::F32(_) => copy_strided!(F32, "f32", f32),
            RocmStorageSlice::F64(_) => copy_strided!(F64, "f64", f64),
            RocmStorageSlice::F8E4M3(_) => {
                crate::bail!("copy_strided_src not supported for F8E4M3 on ROCm")
            }
        }

        Ok(())
    }

    fn copy2d(
        &self,
        dst: &mut Self,
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
        let (src_ptr, dst_ptr, el_size) = match (&self.slice, &mut dst.slice) {
            (RocmStorageSlice::U8(s), RocmStorageSlice::U8(d)) => (s.as_ptr(), d.as_ptr(), 1usize),
            (RocmStorageSlice::U32(s), RocmStorageSlice::U32(d)) => (s.as_ptr(), d.as_ptr(), 4),
            (RocmStorageSlice::I16(s), RocmStorageSlice::I16(d)) => (s.as_ptr(), d.as_ptr(), 2),
            (RocmStorageSlice::I32(s), RocmStorageSlice::I32(d)) => (s.as_ptr(), d.as_ptr(), 4),
            (RocmStorageSlice::I64(s), RocmStorageSlice::I64(d)) => (s.as_ptr(), d.as_ptr(), 8),
            (RocmStorageSlice::BF16(s), RocmStorageSlice::BF16(d)) => (s.as_ptr(), d.as_ptr(), 2),
            (RocmStorageSlice::F16(s), RocmStorageSlice::F16(d)) => (s.as_ptr(), d.as_ptr(), 2),
            (RocmStorageSlice::F32(s), RocmStorageSlice::F32(d)) => (s.as_ptr(), d.as_ptr(), 4),
            (RocmStorageSlice::F64(s), RocmStorageSlice::F64(d)) => (s.as_ptr(), d.as_ptr(), 8),
            (RocmStorageSlice::F8E4M3(s), RocmStorageSlice::F8E4M3(d)) => {
                (s.as_ptr(), d.as_ptr(), 1)
            }
            _ => crate::bail!("dtype mismatch in copy2d"),
        };
        let src_ptr = unsafe { src_ptr.add(src_o * el_size) };
        let dst_ptr = unsafe { dst_ptr.add(dst_o * el_size) };
        let width = d2 * el_size;
        let spitch = src_s1 * el_size;
        let dpitch = dst_s1 * el_size;
        let result = unsafe {
            bindings::hipMemcpy2D(
                dst_ptr,
                dpitch,
                src_ptr,
                spitch,
                width,
                d1,
                bindings::hipMemcpyKind_hipMemcpyDeviceToDevice,
            )
        };
        if result != bindings::hipError_t_hipSuccess {
            crate::bail!("hipMemcpy2D failed with error {}", result);
        }
        Ok(())
    }

    fn const_set(&mut self, val: crate::scalar::Scalar, layout: &Layout) -> Result<()> {
        let shape = layout.shape();
        let dims = shape.dims();
        let el_count = shape.elem_count();
        if el_count == 0 {
            return Ok(());
        }

        let (grid, block) = launch_config(el_count);
        let ds = dims_and_strides(&self.device, layout, 1)?;

        macro_rules! const_set {
            ($variant:ident, $suffix:expr, $ty:ty, $val:expr) => {{
                let mem = match &mut self.slice {
                    RocmStorageSlice::$variant(m) => m,
                    _ => crate::bail!("dtype mismatch in const_set"),
                };
                let func_name = format!("const_set_{}", $suffix);
                let out_ptr = unsafe { mem.ptr_at(layout.start_offset()) };
                let scalar_val: $ty = $val;
                let ds_ptr: *const usize = ds
                    .as_ref()
                    .map(|d| d.as_ptr() as *const usize)
                    .unwrap_or(std::ptr::null());
                unsafe {
                    launch_kernel(
                        &self.device,
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

        match (&mut self.slice, val) {
            (RocmStorageSlice::U8(_), crate::scalar::Scalar::U8(v)) => const_set!(U8, "u8", u8, v),
            (RocmStorageSlice::U32(_), crate::scalar::Scalar::U32(v)) => {
                const_set!(U32, "u32", u32, v)
            }
            (RocmStorageSlice::I64(_), crate::scalar::Scalar::I64(v)) => {
                const_set!(I64, "i64", i64, v)
            }
            (RocmStorageSlice::F32(_), crate::scalar::Scalar::F32(v)) => {
                const_set!(F32, "f32", f32, v)
            }
            (RocmStorageSlice::F64(_), crate::scalar::Scalar::F64(v)) => {
                const_set!(F64, "f64", f64, v)
            }
            (RocmStorageSlice::BF16(_), crate::scalar::Scalar::BF16(v)) => {
                const_set!(BF16, "bf16", bf16, v)
            }
            (RocmStorageSlice::F16(_), crate::scalar::Scalar::F16(v)) => {
                const_set!(F16, "f16", f16, v)
            }
            (RocmStorageSlice::I16(_), crate::scalar::Scalar::I16(v)) => {
                const_set!(I16, "i16", i16, v)
            }
            (RocmStorageSlice::I32(_), crate::scalar::Scalar::I32(v)) => {
                const_set!(I32, "i32", i32, v)
            }
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
}
