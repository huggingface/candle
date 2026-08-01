//! MMQ: the tiled `q8_1` matmul, for a batch past what [`super::mmvq`] covers.
//!
//! This is prefill. The vector kernels stop at a batch of eight, so before this
//! path existed a prompt of any length dequantized the whole weight matrix into
//! the activation's dtype and ran a rocBLAS GEMM against it — `n * k` elements
//! written and read back per matmul, per layer, for a result that then only
//! needed `b_size` rows of it.
//!
//! `mul_mat_q` blocks the weights instead: each workgroup loads an
//! `mmq_y x WARP_SIZE` tile of packed blocks into LDS, dequantizes nothing, and
//! accumulates `mmq_x` output columns through `__dp4a`. The activations reach it
//! as `q8_1` from [`super::q8_1`], the same requantization MMVQ uses.
//!
//! Kernel signature and launch geometry are taken from
//! `quantized/cuda.rs::mul_mat_via_q8_1` rather than inferred. The tile sizes
//! are not a constant: `quantized.cu` carries one set per architecture, and
//! [`candle_rocm_kernels::MmqTiles`] reports which set the device's kernels were
//! compiled with so [`plan`] can reproduce it.

use super::kernels::{arg, launch_err, WARP_SIZE};
use super::q8_1::{buffer_bytes, force_dmmv, pad, quantize_q8_1};
use super::QRocmStorage;
use crate::backend::BackendStorage;
use crate::quantized::GgmlDType;
use crate::rocm_backend::rocm_rs::hip::Dim3;
use crate::rocm_backend::{
    kernels, RocmDevice, RocmStorage, RocmStorageSlice, SendSyncDeviceMemory,
};
use crate::{DType, Layout, Result};
use candle_rocm_kernels::MmqTiles;

use super::kernels::MATRIX_ROW_PADDING;

/// Entry-point name for `dtype`, and the `k` granularity its column loop steps
/// in, or `None` for a dtype with no `mul_mat_q*` kernel at all.
///
/// `k` must be a multiple of the step. The kernel walks the shared dimension as
/// `for (ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_warp)` with no bound
/// on the last step, so a `k` that leaves a partial step reads into the *next*
/// weight row — a silently wrong result, not a fault. The value is
/// `qk * (WARP_SIZE / qi)` for the dtype's `QK`/`QI` in `quantized.cu`, so
/// unlike the tile sizes it does not move with the architecture.
fn kernel(dtype: GgmlDType) -> Option<(&'static str, usize)> {
    let plan = match dtype {
        GgmlDType::Q4_0 => ("mul_mat_q4_0", 256),
        GgmlDType::Q4_1 => ("mul_mat_q4_1", 256),
        GgmlDType::Q5_0 => ("mul_mat_q5_0", 256),
        GgmlDType::Q5_1 => ("mul_mat_q5_1", 256),
        GgmlDType::Q8_0 => ("mul_mat_q8_0", 128),
        GgmlDType::Q2K => ("mul_mat_q2_K", 512),
        GgmlDType::Q3K => ("mul_mat_q3_K", 512),
        GgmlDType::Q4K => ("mul_mat_q4_K", 256),
        GgmlDType::Q5K => ("mul_mat_q5_K", 256),
        GgmlDType::Q6K => ("mul_mat_q6_K", 256),
        _ => return None,
    };
    Some(plan)
}

/// What one `mul_mat_q*` launch needs. Everything here except `name` and
/// `k_step` is per *architecture* as well as per dtype: `quantized.cu` carries
/// one tile set per architecture and the host has to launch whichever the
/// kernel was compiled with.
struct Plan {
    name: &'static str,
    /// Output columns per workgroup (`MMQ_X_*`).
    mmq_x: usize,
    /// Output rows per workgroup (`MMQ_Y_*`).
    mmq_y: usize,
    /// Warps per workgroup (`NWARPS_*`), and so the y-extent of the block.
    nwarps: usize,
    /// See [`kernel`].
    k_step: usize,
}

/// The geometry `quantized.cu` compiles `dtype`'s kernel with under `tiles`,
/// mirroring the `MMQ_X_OF`/`MMQ_Y_OF`/`NWARPS_OF` selection there.
///
/// `nwarps` is uniform within a set — 4 for every `NWARPS_*_AMPERE`, 8 for every
/// `NWARPS_*_RDNA2` — so it is not tabulated per dtype.
fn plan(dtype: GgmlDType, tiles: MmqTiles) -> Option<Plan> {
    let (name, k_step) = kernel(dtype)?;
    let (mmq_x, mmq_y) = match tiles {
        MmqTiles::Ampere => match dtype {
            GgmlDType::Q5_0 | GgmlDType::Q5_1 | GgmlDType::Q8_0 => (128, 64),
            GgmlDType::Q3K => (128, 128),
            GgmlDType::Q6K => (64, 64),
            _ => (64, 128),
        },
        MmqTiles::Rdna2 => match dtype {
            GgmlDType::Q3K => (128, 64),
            _ => (64, 128),
        },
    };
    let nwarps = match tiles {
        MmqTiles::Ampere => 4,
        MmqTiles::Rdna2 => 8,
    };
    Some(Plan {
        name,
        mmq_x,
        mmq_y,
        nwarps,
        k_step,
    })
}

/// `(mmq_x, mmq_y, nwarps)` for [`super::tests_mmq`], which checks the table
/// above against the `#define`s in `quantized.cu` itself.
#[cfg(test)]
pub(super) fn geometry(dtype: GgmlDType, tiles: MmqTiles) -> Option<(usize, usize, usize)> {
    plan(dtype, tiles).map(|p| (p.mmq_x, p.mmq_y, p.nwarps))
}

/// Whether `(dtype, k)` can go through MMQ at all.
///
/// Architecture-independent: the tile sizes move with it but `k_step` does not,
/// and `n` and `b_size` are unconstrained either way — unlike the MMVQ kernels,
/// `mul_mat_q` bounds-checks both coordinates of its store and clamps the
/// activation column it loads, so a workgroup that overhangs the output is
/// harmless.
pub(super) fn supports(dtype: GgmlDType, k: usize) -> bool {
    match kernel(dtype) {
        Some((_, k_step)) => k > 0 && k.is_multiple_of(k_step),
        None => false,
    }
}

/// Weight elements at or above which MMQ beats the dense dequantize, or `None`
/// for a dtype where it never does.
///
/// Being *able* to run MMQ is not a reason to. The dense path dequantizes `n*k`
/// weights and hands rocBLAS a plain GEMM, and rocBLAS is good: MMQ only comes
/// out ahead once that materialisation is large enough to dominate, and for
/// three dtypes it never does. Measured on gfx1101, `dense_ms / mmq_ms` (above
/// 1 means MMQ wins), each cell the range over batches of 9, 32, 128 and 512 —
/// `bench_prefill_paths` is what prints it:
///
/// ```text
/// weights   0.25 Mi     1 Mi        4 Mi        16 Mi       131 Mi
/// Q4_0      0.61-0.75   0.66-1.25   1.10-1.35   1.54-1.94   1.69-3.27
/// Q4_1      0.63-0.84   0.74-1.25   1.07-1.31   1.52-1.85   1.62-3.12
/// Q8_0      0.54-0.79   0.58-1.08   0.94-1.10   1.25-1.49   1.35-1.51
/// Q2K       0.43-0.55   0.48-0.81   0.75-0.86   0.97-1.23   1.12-2.34
/// Q4K       0.44-0.56   0.49-0.76   0.75-0.84   0.93-1.13   1.03-1.99
/// Q6K       0.86-1.01   0.96-1.17   1.22-1.54   1.22-2.12   1.15-2.53
/// Q5_0      0.46-0.59   0.51-0.84   0.77-0.83   0.99-1.03   0.89-1.06
/// Q5_1      0.46-0.59   0.51-0.83   0.78-0.84   0.98-1.03   0.89-1.05
/// Q3K       0.20-0.29   0.22-0.41   0.33-0.35   0.42-0.47   0.49-0.54
/// Q5K       0.41-0.54   0.45-0.49   0.47-0.76   0.54-1.04   0.51-1.11
/// ```
///
/// The last four never earn it. Q5_0/Q5_1 flatten out at a wash, and Q3K and
/// Q5K are worse than dense at every size measured — Q3K by more than 2x
/// throughout — so they keep the dense path.
///
/// The table above was measured with the Ampere tile set, i.e. before gfx11
/// started compiling these kernels at `nwarps = 8`; the thresholds are due a
/// re-measurement under the RDNA geometry.
///
/// The thresholds sit where the ratio first stops favouring dense rather than
/// where it becomes decisive, because the timings understate the case for MMQ:
/// the dense path also *allocates* the dequantized matrix, 524 MB of transient
/// for a 4096x32000 `lm_head` at f32, which the ratio does not show.
fn min_work(dtype: GgmlDType) -> Option<usize> {
    match dtype {
        GgmlDType::Q4_0 | GgmlDType::Q4_1 | GgmlDType::Q6K => Some(4 << 20),
        GgmlDType::Q8_0 | GgmlDType::Q2K | GgmlDType::Q4K => Some(16 << 20),
        _ => None,
    }
}

/// Whether MMQ is the faster path for `n * k` weight elements of `dtype`.
pub(super) fn is_faster(dtype: GgmlDType, work: usize) -> bool {
    matches!(min_work(dtype), Some(min) if work >= min)
}

/// `data` (the `(nrows, ncols)` packed weights) times `b_size` activation rows.
///
/// ```text
/// mul_mat_q*(const void *vx, const void *vy, float *dst,
///            const int ncols_x, const int nrows_x,
///            const int ncols_y, const int nrows_y, const int nrows_dst)
/// ```
///
/// `nrows_y` is the *padded* activation width: the kernel derives
/// `blocks_per_col_y = nrows_y / QK8_1` and uses it as the row stride into the
/// q8_1 buffer, which [`quantize_q8_1`] laid out at the padded width. The
/// output is written as `dst[col*nrows_dst + row]`, i.e. `b_size` rows of
/// `nrows` — the contiguous `(b_size, n)` the caller wants.
#[allow(clippy::too_many_arguments)]
pub(super) fn mul_mat_via_q8_1(
    data: &SendSyncDeviceMemory<u8>,
    data_len: usize,
    y: &SendSyncDeviceMemory<f32>,
    y_offset: usize,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    b_size: usize,
    dev: &RocmDevice,
) -> Result<SendSyncDeviceMemory<f32>> {
    let data_elems = data_len / dtype.type_size() * dtype.block_size();
    if data_elems < ncols * nrows {
        crate::bail!("quantized mmq: data holds {data_elems} elems, need {ncols}x{nrows}")
    }
    let plan = match plan(dtype, dev.mmq_tiles()) {
        Some(plan) => plan,
        None => crate::bail!("no ROCm mmq kernel for {dtype:?}"),
    };
    if !ncols.is_multiple_of(plan.k_step) {
        crate::bail!(
            "quantized mmq: k={ncols} is not a multiple of {} for {dtype:?}",
            plan.k_step
        )
    }

    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let y_q8_1 = dev.alloc_zeros::<u8>(buffer_bytes(ncols, b_size))?;
    quantize_q8_1(y, y_offset, &y_q8_1, ncols, b_size, dev)?;

    let func = dev.get_or_load_func(plan.name, &kernels::QUANTIZED)?;
    let dst = dev.alloc_zeros::<f32>(nrows * b_size)?;
    let src_ptr = data.as_ptr();
    let y_ptr = y_q8_1.as_ptr();
    let dst_ptr = dst.as_ptr();
    let ncols_x = ncols as i32;
    let nrows_x = nrows as i32;
    let ncols_y = b_size as i32;
    let nrows_y = ncols_padded as i32;
    let nrows_dst = nrows as i32;
    let mut args = vec![
        arg(&src_ptr),
        arg(&y_ptr),
        arg(&dst_ptr),
        arg(&ncols_x),
        arg(&nrows_x),
        arg(&ncols_y),
        arg(&nrows_y),
        arg(&nrows_dst),
    ];
    func.launch(
        Dim3::new_2d(
            nrows.div_ceil(plan.mmq_y) as u32,
            b_size.div_ceil(plan.mmq_x) as u32,
        ),
        Dim3::new_2d(WARP_SIZE as u32, plan.nwarps as u32),
        0,
        Some(dev.stream()),
        &mut args,
    )
    .map_err(|e| launch_err(plan.name, e))?;
    Ok(dst)
}

/// The MMQ dispatch for [`QRocmStorage::fwd`], or `None` when the shape, dtype
/// or activation is one this path does not cover — or does not pay for — and
/// the caller should fall through to the dense dequantize.
///
/// Every batch the vector paths declined comes here, not only the large ones:
/// MMQ is correct at any `b_size`, and the batch is not what decides whether it
/// is worth taking. The weight matrix is; see [`min_work`].
pub(super) fn try_fwd(
    q: &QRocmStorage,
    n: usize,
    k: usize,
    b_size: usize,
    storage: &RocmStorage,
    layout: &Layout,
) -> Result<Option<SendSyncDeviceMemory<f32>>> {
    if force_dmmv() || !supports(q.dtype, k) || !is_faster(q.dtype, n * k) || b_size == 0 {
        return Ok(None);
    }
    let (o1, o2) = match layout.contiguous_offsets() {
        Some(offsets) => offsets,
        None => return Ok(None),
    };
    if o2 - o1 != b_size * k {
        return Ok(None);
    }

    // `quantize_q8_1` reads f32, so an f16/bf16 activation is cast first. That
    // is `b_size * k` elements — for a prefill of 1024 tokens against a
    // 4096x4096 weight, three orders of magnitude below what the dense fallback
    // would materialise.
    let cast;
    let (y, y_offset) = match &storage.slice {
        RocmStorageSlice::F32(y) => (y, o1),
        RocmStorageSlice::F16(_) | RocmStorageSlice::BF16(_) => {
            cast = storage.to_dtype(layout, DType::F32)?;
            match &cast.slice {
                RocmStorageSlice::F32(y) => (y, 0usize),
                _ => return Ok(None),
            }
        }
        _ => return Ok(None),
    };

    let dst = mul_mat_via_q8_1(
        &q.data, q.len, y, y_offset, q.dtype, k, n, b_size, &q.device,
    )?;
    Ok(Some(dst))
}
