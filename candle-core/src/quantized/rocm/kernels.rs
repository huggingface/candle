//! Launchers for the quantized kernels of `candle-kernels/src/quantized.cu`.
//!
//! The ROCm backend compiles the very same `.cu` sources the CUDA backend uses,
//! so the launch geometry is taken from `quantized/cuda.rs`. None of these
//! kernels is a grid-stride loop, so `rocm_backend::launch_config` must not be
//! used for them: each maps `blockIdx` straight onto a block of the quantized
//! payload and a grid that is too small silently leaves the tail of the output
//! untouched.
//!
//! Dequantization and row gathering live here; the two matrix-vector paths are
//! in [`super::dmmv`] and [`super::mmvq`].

use crate::quantized::GgmlDType;
use crate::rocm_backend::rocm_rs::hip::Dim3;
use crate::rocm_backend::{kernels, RocmDevice, SendSyncDeviceMemory};
use crate::Result;
use half::f16;
use std::ffi::c_void;

/// Rows of a quantized matrix are padded up to a multiple of this many
/// elements. Kernels that consume `q8_1`-quantized activations read whole
/// padded rows, so the allocation has to carry the padding even though the
/// logical payload stops earlier.
pub(crate) const MATRIX_ROW_PADDING: usize = 512;

const DEQUANTIZE_BLOCK_SIZE: usize = 256;
const GET_ROWS_BLOCK_SIZE: usize = 256;

/// Warp width every kernel in `quantized.cu` is compiled for (`#define
/// WARP_SIZE 32`). RDNA3 runs wave32, so a block's x-dimension of 32 matches
/// the wavefront exactly; correctness on wave64 parts is the caller's to verify.
pub(super) const WARP_SIZE: usize = 32;

/// `hipModuleLaunchKernel` takes an array of pointers *to* the argument values.
pub(super) fn arg<T>(v: &T) -> *mut c_void {
    v as *const T as *mut c_void
}

pub(super) fn launch_err(name: &str, e: impl std::fmt::Display) -> crate::Error {
    crate::Error::Msg(format!("quantized kernel {name} launch failed: {e}"))
}

/// Kernel name, whether it is a K-quant, block size and grid size for one
/// dequantization pass over `elem_count` elements.
fn dequantize_plan(
    dtype: GgmlDType,
    elem_count: usize,
    suffix: &str,
) -> Result<(String, bool, usize, usize)> {
    let nb = elem_count.div_ceil(256);
    let nb_half = elem_count.div_ceil(2 * DEQUANTIZE_BLOCK_SIZE);
    let (name, is_k, block_dim, num_blocks) = match dtype {
        GgmlDType::Q4_0 => ("dequantize_block_q4_0", false, 32, nb),
        GgmlDType::Q4_1 => ("dequantize_block_q4_1", false, 32, nb),
        GgmlDType::Q5_0 => (
            "dequantize_block_q5_0",
            false,
            DEQUANTIZE_BLOCK_SIZE,
            nb_half,
        ),
        GgmlDType::Q5_1 => (
            "dequantize_block_q5_1",
            false,
            DEQUANTIZE_BLOCK_SIZE,
            nb_half,
        ),
        GgmlDType::Q8_0 => ("dequantize_block_q8_0", false, 32, nb),
        GgmlDType::Q2K => ("dequantize_block_q2_K", true, 64, nb),
        GgmlDType::Q3K => ("dequantize_block_q3_K", true, 64, nb),
        GgmlDType::Q4K => ("dequantize_block_q4_K", true, 32, nb),
        GgmlDType::Q5K => ("dequantize_block_q5_K", true, 64, nb),
        GgmlDType::Q6K => ("dequantize_block_q6_K", true, 64, nb),
        GgmlDType::Q8K => ("dequantize_block_q8_K", true, 32, nb),
        _ => crate::bail!("unsupported dtype for ROCm dequantize {dtype:?}"),
    };
    Ok((format!("{name}_{suffix}"), is_k, block_dim, num_blocks))
}

/// True when `dtype` has a dequantization kernel; the remaining ggml dtypes
/// have to go through the CPU.
pub(crate) fn has_dequantize_kernel(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q4_0
            | GgmlDType::Q4_1
            | GgmlDType::Q5_0
            | GgmlDType::Q5_1
            | GgmlDType::Q8_0
            | GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q4K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
            | GgmlDType::Q8K
    )
}

/// Dequantize `data` into the freshly allocated `dst`.
///
/// The two kernel families differ in arity — this is exactly the kind of
/// mismatch that produces silent wrong answers, so both shapes are spelled out:
///
/// ```text
/// K-quants: dequantize_block_qN_K_{f32,f16}(const void *vx, dst_t *y)
/// others:   dequantize_block_qN_M_{f32,f16}(const void *vx, dst_t *y, const int k)
/// ```
///
/// `k` is *not* the element count for every dtype: the `q5_0`/`q5_1` kernels
/// forward to the generic two-elements-per-thread `dequantize_block`, which
/// bounds-checks against a raw element count, while `q4_0`/`q4_1`/`q8_0` check
/// against a count of 32-element sub-blocks.
fn dequantize_into<T>(
    data: &SendSyncDeviceMemory<u8>,
    dst: &SendSyncDeviceMemory<T>,
    dtype: GgmlDType,
    elem_count: usize,
    suffix: &str,
    dev: &RocmDevice,
) -> Result<()> {
    let (name, is_k, block_dim, num_blocks) = dequantize_plan(dtype, elem_count, suffix)?;
    if num_blocks == 0 {
        return Ok(());
    }
    let func = dev.get_or_load_func(&name, &kernels::QUANTIZED)?;
    let src_ptr = data.as_ptr();
    let dst_ptr = dst.as_ptr();
    let nb32 = match dtype {
        GgmlDType::Q5_0 | GgmlDType::Q5_1 => elem_count as i32,
        _ => (elem_count / 32) as i32,
    };
    let mut args = vec![arg(&src_ptr), arg(&dst_ptr)];
    if !is_k {
        args.push(arg(&nb32));
    }
    func.launch(
        Dim3::new_1d(num_blocks as u32),
        Dim3::new_1d(block_dim as u32),
        0,
        Some(dev.stream()),
        &mut args,
    )
    .map_err(|e| launch_err(&name, e))
}

pub(crate) fn dequantize_f32(
    data: &SendSyncDeviceMemory<u8>,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &RocmDevice,
) -> Result<SendSyncDeviceMemory<f32>> {
    let dst = dev.alloc::<f32>(elem_count)?;
    dequantize_into(data, &dst, dtype, elem_count, "f32", dev)?;
    Ok(dst)
}

pub(crate) fn dequantize_f16(
    data: &SendSyncDeviceMemory<u8>,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &RocmDevice,
) -> Result<SendSyncDeviceMemory<f16>> {
    let dst = dev.alloc::<f16>(elem_count)?;
    dequantize_into(data, &dst, dtype, elem_count, "f16", dev)?;
    Ok(dst)
}

/// Kernel name, block size, grid y and whether the kernel strides over `gridDim.y`.
fn get_rows_plan(dtype: GgmlDType, hidden: usize) -> Result<(&'static str, usize, usize, bool)> {
    let plan = match dtype {
        GgmlDType::F32 => (
            "get_rows_f32",
            GET_ROWS_BLOCK_SIZE,
            hidden.div_ceil(GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::F16 => (
            "get_rows_f16",
            GET_ROWS_BLOCK_SIZE,
            hidden.div_ceil(GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::BF16 => (
            "get_rows_bf16",
            GET_ROWS_BLOCK_SIZE,
            hidden.div_ceil(GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::Q4_0 => (
            "get_rows_q4_0",
            GET_ROWS_BLOCK_SIZE,
            hidden.div_ceil(2 * GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::Q4_1 => (
            "get_rows_q4_1",
            GET_ROWS_BLOCK_SIZE,
            hidden.div_ceil(2 * GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::Q5_0 => (
            "get_rows_q5_0",
            GET_ROWS_BLOCK_SIZE,
            hidden.div_ceil(2 * GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::Q5_1 => (
            "get_rows_q5_1",
            GET_ROWS_BLOCK_SIZE,
            hidden.div_ceil(2 * GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::Q8_0 => (
            "get_rows_q8_0",
            GET_ROWS_BLOCK_SIZE,
            hidden.div_ceil(2 * GET_ROWS_BLOCK_SIZE),
            true,
        ),
        // The K-quant kernels dequantize exactly one block per `blockIdx.y`, so
        // the grid *is* the block count and cannot be shortened.
        GgmlDType::Q2K => ("get_rows_q2_K", 64, hidden / dtype.block_size(), false),
        GgmlDType::Q3K => ("get_rows_q3_K", 64, hidden / dtype.block_size(), false),
        GgmlDType::Q4K => ("get_rows_q4_K", 32, hidden / dtype.block_size(), false),
        GgmlDType::Q5K => ("get_rows_q5_K", 64, hidden / dtype.block_size(), false),
        GgmlDType::Q6K => ("get_rows_q6_K", 64, hidden / dtype.block_size(), false),
        _ => crate::bail!("unsupported dtype for ROCm quantized embedding {dtype:?}"),
    };
    Ok(plan)
}

/// Gather `ids_len` rows of `hidden` elements out of the quantized `data`.
///
/// ```text
/// get_rows_*(const void *src0, const uint32_t *src1, float *dst,
///            const int64_t ne00, const size_t nb01)
/// ```
///
/// `ne00` is the row width in elements and `nb01` the row stride in *bytes*.
pub(crate) fn get_rows(
    data: &SendSyncDeviceMemory<u8>,
    dtype: GgmlDType,
    hidden: usize,
    ids: &SendSyncDeviceMemory<u32>,
    ids_offset: usize,
    ids_len: usize,
    dev: &RocmDevice,
) -> Result<SendSyncDeviceMemory<f32>> {
    let (name, block_dim, block_num_y, can_stride_y) = get_rows_plan(dtype, hidden)?;
    let dst = dev.alloc::<f32>(ids_len * hidden)?;
    if ids_len == 0 || hidden == 0 {
        return Ok(dst);
    }
    if !can_stride_y && block_num_y > u16::MAX as usize {
        crate::bail!("quantized embedding hidden size {hidden} exceeds the grid y limit")
    }
    let grid_y = if can_stride_y {
        block_num_y.min(u16::MAX as usize)
    } else {
        block_num_y
    };
    let func = dev.get_or_load_func(name, &kernels::QUANTIZED)?;
    let src_ptr = data.as_ptr();
    let ids_ptr = unsafe { ids.ptr_at(ids_offset) };
    let dst_ptr = dst.as_ptr();
    let ne00 = hidden as i64;
    let nb01 = hidden * dtype.type_size() / dtype.block_size();
    let mut args = vec![
        arg(&src_ptr),
        arg(&ids_ptr),
        arg(&dst_ptr),
        arg(&ne00),
        arg(&nb01),
    ];
    func.launch(
        Dim3::new_2d(ids_len as u32, grid_y as u32),
        Dim3::new_1d(block_dim as u32),
        0,
        Some(dev.stream()),
        &mut args,
    )
    .map_err(|e| launch_err(name, e))?;
    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `k` means different things per dtype; getting it wrong dequantizes only
    /// a fraction of the tensor (or reads past the end).
    #[test]
    fn dequantize_plan_shapes() -> Result<()> {
        // 32-element blocks: one 256-thread-equivalent block per 256 elements.
        let (name, is_k, block, grid) = dequantize_plan(GgmlDType::Q4_0, 1024, "f32")?;
        assert_eq!(name, "dequantize_block_q4_0_f32");
        assert!(!is_k);
        assert_eq!((block, grid), (32, 4));

        // q5_* forward to the generic two-per-thread kernel.
        let (name, is_k, block, grid) = dequantize_plan(GgmlDType::Q5_0, 1024, "f16")?;
        assert_eq!(name, "dequantize_block_q5_0_f16");
        assert!(!is_k);
        assert_eq!((block, grid), (256, 2));

        // K-quants take no `k` argument at all.
        let (name, is_k, block, grid) = dequantize_plan(GgmlDType::Q6K, 1024, "f32")?;
        assert_eq!(name, "dequantize_block_q6_K_f32");
        assert!(is_k);
        assert_eq!((block, grid), (64, 4));

        assert!(dequantize_plan(GgmlDType::Q8_1, 1024, "f32").is_err());
        Ok(())
    }

    #[test]
    fn get_rows_plan_shapes() -> Result<()> {
        let (name, block, grid_y, stride) = get_rows_plan(GgmlDType::Q4_0, 1024)?;
        assert_eq!(name, "get_rows_q4_0");
        assert_eq!((block, grid_y, stride), (256, 2, true));

        // One block of the grid per 256-element super-block, no folding.
        let (name, block, grid_y, stride) = get_rows_plan(GgmlDType::Q4K, 1024)?;
        assert_eq!(name, "get_rows_q4_K");
        assert_eq!((block, grid_y, stride), (32, 4, false));

        assert!(get_rows_plan(GgmlDType::Q8K, 1024).is_err());
        Ok(())
    }
}
