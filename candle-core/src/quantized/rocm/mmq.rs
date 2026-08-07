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
/// for a dtype with no `mul_mat_q*` kernel at all.
///
/// Being *able* to run MMQ is not a reason to. The dense path dequantizes `n*k`
/// weights and hands rocBLAS a plain GEMM, and rocBLAS is good: MMQ only comes
/// out ahead once that materialisation is large enough to dominate.
///
/// **Measured on gfx1101 (RDNA3, Navi 32) with the RDNA tile geometry**, i.e.
/// the `nwarps = 8` kernels that no longer spill. Both halves of that matter.
/// The same sweep under the Ampere tile set put seven of these ten dtypes below
/// 1.0 at every size and Q3K below 0.55 throughout, which is where the previous
/// thresholds came from; anyone porting this to another chip, or changing what
/// [`MmqTiles`] reports, has to re-measure rather than trust the table.
///
/// `dense_ms / mmq_ms`, above 1 means MMQ wins. Each cell is the range over
/// batches of 9, 32, 128 and 512 and over three independent runs, each of which
/// is itself the median of five timing repeats; `bench_prefill_paths` prints
/// it. Individual repeats spread up to ~45% at the smallest sizes, where a call
/// is 35 µs and mostly launch overhead, but the medians reproduce across whole
/// runs to under 2%, so the cells below are stable to about ±0.03.
///
/// ```text
/// weights   0.25 Mi     0.5 Mi      1 Mi        2 Mi        4 Mi        16 Mi       125 Mi
/// Q4_0      1.11-1.39   1.16-1.88   1.54-2.02   1.93-2.64   2.07-2.88   2.50-3.52   2.69-5.95
/// Q4_1      1.12-1.30   1.17-1.62   1.47-1.94   1.83-2.52   1.95-2.73   2.33-3.36   2.52-5.51
/// Q5_0      1.16-1.32   1.18-1.55   1.48-1.98   1.80-2.55   1.94-2.76   2.15-3.46   2.30-5.13
/// Q5_1      1.14-1.29   1.16-1.50   1.44-1.95   1.74-2.51   1.86-2.69   2.09-3.33   2.25-4.89
/// Q8_0      1.12-1.34   1.15-1.72   1.48-1.94   1.85-2.48   2.01-2.64   2.55-3.55   2.68-6.24
/// Q4K       1.05-1.19   1.07-1.35   1.31-1.78   1.47-2.24   1.68-2.39   1.80-2.84   1.89-4.23
/// Q5K       1.11-1.27   1.13-1.47   1.42-1.97   1.70-2.51   1.81-2.71   1.99-3.14   2.16-4.71
/// Q6K       0.92-1.12   0.92-1.15   1.15-1.54   1.25-1.87   1.35-1.91   1.47-2.40   1.50-3.31
/// Q3K       0.78-0.97   0.79-0.99   0.92-1.10   1.04-1.23   1.21-1.29   1.30-1.35   1.43-1.70
/// Q2K       0.75-0.85   0.75-0.94   0.87-1.23   0.89-1.40   0.99-1.39   1.17-1.66   1.32-2.76
/// ```
///
/// Every dtype now earns MMQ somewhere, so the only `None` left is "no kernel".
/// Seven win at every size swept and are floored at 0.25 Mi rather than at
/// zero, because that is where the sweep bottoms out and nothing smaller was
/// measured — no real weight matrix is below it anyway. The other three cross
/// over inside the sweep, at the first size whose *worst* batch reaches 1.0:
/// Q6K at 1 Mi, Q3K at 2 Mi, Q2K at 4 Mi.
///
/// Two of those crossings are soft, and are called for MMQ deliberately:
///
/// * Q2K at 4 Mi is a coin flip at batch 128 — 1.04, 0.99, 0.99 over the three
///   runs — while its other three batches sit at 1.12-1.39. At 2 Mi that same
///   batch is a flat 0.89 in all three runs, which is a real loss, so 4 Mi is
///   the first size that is not worse than dense.
/// * Q3K at 1 Mi depends on the shape, not just the product: 0.92-0.96 at
///   1024x1024 but 1.03-1.10 at 512x2048. That is the one place the sweep
///   contradicts "the product alone decides", and it is why Q3K waits for 2 Mi,
///   where every shape and batch clears 1.04.
///
/// A wall-clock wash is not a wash overall, which is what makes those calls
/// safe: the dense path also *allocates* the dequantized matrix — 16 MB of
/// transient at 4 Mi weights, 524 MB for a 4096x32000 `lm_head` at f32 — and
/// MMQ allocates none of it. So every threshold here sits where the ratio stops
/// favouring dense rather than where MMQ becomes decisive. What that allocation
/// does *not* buy is permission to go below the measured floor: at 0.5 Mi Q6K
/// is a genuine 0.92 and saves 2 MB, which is not a trade worth making.
fn min_work(dtype: GgmlDType) -> Option<usize> {
    let min = match dtype {
        GgmlDType::Q4_0
        | GgmlDType::Q4_1
        | GgmlDType::Q5_0
        | GgmlDType::Q5_1
        | GgmlDType::Q8_0
        | GgmlDType::Q4K
        | GgmlDType::Q5K => 256 << 10,
        GgmlDType::Q6K => 1 << 20,
        GgmlDType::Q3K => 2 << 20,
        GgmlDType::Q2K => 4 << 20,
        _ => return None,
    };
    Some(min)
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
