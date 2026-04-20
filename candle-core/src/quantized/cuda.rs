use super::{GgmlDType, QStorage};
use crate::quantized::k_quants::GgmlType;
use crate::{backend::BackendDevice, cuda_backend::WrapErr};
use crate::{builder_arg as barg, CudaDevice, CudaStorage, Result};
use half::f16;

use cudarc::driver::{CudaSlice, CudaView, PushKernelArg};

#[derive(Clone, Debug)]
struct PaddedCudaSlice {
    inner: CudaSlice<u8>,
    len: usize,
}

#[derive(Clone, Debug)]
pub struct QCudaStorage {
    data: PaddedCudaSlice,
    dtype: GgmlDType,
    device: CudaDevice,
}

static FORCE_DMMV: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_force_dmmv(f: bool) {
    FORCE_DMMV.store(f, std::sync::atomic::Ordering::Relaxed)
}

pub const WARP_SIZE: usize = 32;
pub const MMQ_X_Q4_0_AMPERE: usize = 4;
pub const MMQ_Y_Q4_0_AMPERE: usize = 32;
pub const NWARPS_Q4_0_AMPERE: usize = 4;
pub const GGML_CUDA_MMV_X: usize = 32;
pub const GGML_CUDA_MMV_Y: usize = 1;
pub const CUDA_QUANTIZE_BLOCK_SIZE: usize = 256;
pub const CUDA_DEQUANTIZE_BLOCK_SIZE: usize = 256;
pub const MATRIX_ROW_PADDING: usize = 512;

fn ceil_div(p: usize, q: usize) -> usize {
    p.div_ceil(q)
}

fn pad(p: usize, q: usize) -> usize {
    ceil_div(p, q) * q
}

fn quantize_q8_1(
    src: &CudaView<f32>,
    dst: &mut CudaSlice<u8>,
    k: usize,
    ky: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let kx_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks = ceil_div(kx_padded, CUDA_QUANTIZE_BLOCK_SIZE);

    let total_rows = ky;
    // Get Q8_1 metadata.
    let q8_1_block_size = GgmlDType::Q8_1.block_size();
    let q8_1_type_size = GgmlDType::Q8_1.type_size();

    // Calculate the size of the output buffer in bytes.
    let num_blocks_per_row = kx_padded / q8_1_block_size;
    let dst_row_size_bytes = num_blocks_per_row * q8_1_type_size;

    const CHUNK_SIZE: usize = 65535; // gridDim.y limit
    let func = dev.get_or_load_func("quantize_q8_1", &candle_kernels::QUANTIZED)?;

    let mut rows_processed = 0;
    while rows_processed < total_rows {
        // --- calculate the number of rows for this chunk ---
        let remaining_rows = total_rows - rows_processed;
        // This is our gridDim.y, now <= 65535
        let rows_in_chunk = std::cmp::min(CHUNK_SIZE, remaining_rows);

        // --- slice the source (f32) tensor by elements ---
        let src_start_elem = rows_processed * k;
        let src_num_elems = rows_in_chunk * k;
        let src_chunk = src.slice(src_start_elem..(src_start_elem + src_num_elems));

        // --- slice the destination (u8) tensor by bytes ---
        let dst_start_byte = rows_processed * dst_row_size_bytes;
        let dst_num_bytes = rows_in_chunk * dst_row_size_bytes;
        let dst_chunk = dst.slice(dst_start_byte..(dst_start_byte + dst_num_bytes));

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_blocks as u32, rows_in_chunk as u32, 1),
            block_dim: (CUDA_QUANTIZE_BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = func.builder();
        builder.arg(&src_chunk);
        builder.arg(&dst_chunk);
        barg!(builder, k as i32, kx_padded as i32);
        unsafe { builder.launch(cfg) }.w()?;

        rows_processed += rows_in_chunk;
    }

    Ok(())
}

/// GPU-native Q8_0 quantizer. Writes directly into a `PaddedCudaSlice`-shaped
/// destination buffer (one `block_q8_0` per 32 source elements followed by
/// MATRIX_ROW_PADDING worth of zero-filled blocks — the caller must zero
/// the padding region before this call).
///
/// Source tensor is treated as `ky` rows of `k` elements each. The trailing
/// `kx_padded - k` elements of each row are implicitly treated as 0 by the
/// kernel, so they never contribute to the per-block amax.
fn quantize_q8_0_f32(
    src: &CudaView<f32>,
    dst: &mut CudaSlice<u8>,
    k: usize,
    ky: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let kx_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks = ceil_div(kx_padded, CUDA_QUANTIZE_BLOCK_SIZE);

    let total_rows = ky;
    let q8_0_block_size = GgmlDType::Q8_0.block_size();
    let q8_0_type_size = GgmlDType::Q8_0.type_size();
    let num_blocks_per_row = kx_padded / q8_0_block_size;
    let dst_row_size_bytes = num_blocks_per_row * q8_0_type_size;

    const CHUNK_SIZE: usize = 65535;
    let func = dev.get_or_load_func("quantize_q8_0", &candle_kernels::QUANTIZED)?;

    let mut rows_processed = 0;
    while rows_processed < total_rows {
        let remaining_rows = total_rows - rows_processed;
        let rows_in_chunk = std::cmp::min(CHUNK_SIZE, remaining_rows);

        let src_start_elem = rows_processed * k;
        let src_num_elems = rows_in_chunk * k;
        let src_chunk = src.slice(src_start_elem..(src_start_elem + src_num_elems));

        let dst_start_byte = rows_processed * dst_row_size_bytes;
        let dst_num_bytes = rows_in_chunk * dst_row_size_bytes;
        let dst_chunk = dst.slice(dst_start_byte..(dst_start_byte + dst_num_bytes));

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_blocks as u32, rows_in_chunk as u32, 1),
            block_dim: (CUDA_QUANTIZE_BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = func.builder();
        builder.arg(&src_chunk);
        builder.arg(&dst_chunk);
        barg!(builder, k as i32, kx_padded as i32);
        unsafe { builder.launch(cfg) }.w()?;

        rows_processed += rows_in_chunk;
    }

    Ok(())
}

/// f16 variant of `quantize_q8_0_f32`. Avoids an f16→f32 cast + extra tensor
/// allocation on the hot path when the source activations are already f16.
fn quantize_q8_0_f16(
    src: &CudaView<f16>,
    dst: &mut CudaSlice<u8>,
    k: usize,
    ky: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let kx_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks = ceil_div(kx_padded, CUDA_QUANTIZE_BLOCK_SIZE);

    let total_rows = ky;
    let q8_0_block_size = GgmlDType::Q8_0.block_size();
    let q8_0_type_size = GgmlDType::Q8_0.type_size();
    let num_blocks_per_row = kx_padded / q8_0_block_size;
    let dst_row_size_bytes = num_blocks_per_row * q8_0_type_size;

    const CHUNK_SIZE: usize = 65535;
    let func = dev.get_or_load_func("quantize_q8_0_f16", &candle_kernels::QUANTIZED)?;

    let mut rows_processed = 0;
    while rows_processed < total_rows {
        let remaining_rows = total_rows - rows_processed;
        let rows_in_chunk = std::cmp::min(CHUNK_SIZE, remaining_rows);

        let src_start_elem = rows_processed * k;
        let src_num_elems = rows_in_chunk * k;
        let src_chunk = src.slice(src_start_elem..(src_start_elem + src_num_elems));

        let dst_start_byte = rows_processed * dst_row_size_bytes;
        let dst_num_bytes = rows_in_chunk * dst_row_size_bytes;
        let dst_chunk = dst.slice(dst_start_byte..(dst_start_byte + dst_num_bytes));

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_blocks as u32, rows_in_chunk as u32, 1),
            block_dim: (CUDA_QUANTIZE_BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = func.builder();
        builder.arg(&src_chunk);
        builder.arg(&dst_chunk);
        barg!(builder, k as i32, kx_padded as i32);
        unsafe { builder.launch(cfg) }.w()?;

        rows_processed += rows_in_chunk;
    }

    Ok(())
}

/// GPU-native Q4_0 quantizer (f32 source). Mirrors `quantize_q8_0_f32` but
/// emits 18-byte blocks (1 half scale + 16 bytes of nibble-packed quants) for
/// half the bytes per element. Used for KV cache storage at long contexts
/// where the Q8_0 memory footprint doesn't fit.
fn quantize_q4_0_f32(
    src: &CudaView<f32>,
    dst: &mut CudaSlice<u8>,
    k: usize,
    ky: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let kx_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks = ceil_div(kx_padded, CUDA_QUANTIZE_BLOCK_SIZE);

    let total_rows = ky;
    let q4_0_block_size = GgmlDType::Q4_0.block_size();
    let q4_0_type_size = GgmlDType::Q4_0.type_size();
    let num_blocks_per_row = kx_padded / q4_0_block_size;
    let dst_row_size_bytes = num_blocks_per_row * q4_0_type_size;

    const CHUNK_SIZE: usize = 65535;
    let func = dev.get_or_load_func("quantize_q4_0", &candle_kernels::QUANTIZED)?;

    let mut rows_processed = 0;
    while rows_processed < total_rows {
        let remaining_rows = total_rows - rows_processed;
        let rows_in_chunk = std::cmp::min(CHUNK_SIZE, remaining_rows);

        let src_start_elem = rows_processed * k;
        let src_num_elems = rows_in_chunk * k;
        let src_chunk = src.slice(src_start_elem..(src_start_elem + src_num_elems));

        let dst_start_byte = rows_processed * dst_row_size_bytes;
        let dst_num_bytes = rows_in_chunk * dst_row_size_bytes;
        let dst_chunk = dst.slice(dst_start_byte..(dst_start_byte + dst_num_bytes));

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_blocks as u32, rows_in_chunk as u32, 1),
            block_dim: (CUDA_QUANTIZE_BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = func.builder();
        builder.arg(&src_chunk);
        builder.arg(&dst_chunk);
        barg!(builder, k as i32, kx_padded as i32);
        unsafe { builder.launch(cfg) }.w()?;

        rows_processed += rows_in_chunk;
    }

    Ok(())
}

fn quantize_q4_0_f16(
    src: &CudaView<f16>,
    dst: &mut CudaSlice<u8>,
    k: usize,
    ky: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let kx_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks = ceil_div(kx_padded, CUDA_QUANTIZE_BLOCK_SIZE);

    let total_rows = ky;
    let q4_0_block_size = GgmlDType::Q4_0.block_size();
    let q4_0_type_size = GgmlDType::Q4_0.type_size();
    let num_blocks_per_row = kx_padded / q4_0_block_size;
    let dst_row_size_bytes = num_blocks_per_row * q4_0_type_size;

    const CHUNK_SIZE: usize = 65535;
    let func = dev.get_or_load_func("quantize_q4_0_f16", &candle_kernels::QUANTIZED)?;

    let mut rows_processed = 0;
    while rows_processed < total_rows {
        let remaining_rows = total_rows - rows_processed;
        let rows_in_chunk = std::cmp::min(CHUNK_SIZE, remaining_rows);

        let src_start_elem = rows_processed * k;
        let src_num_elems = rows_in_chunk * k;
        let src_chunk = src.slice(src_start_elem..(src_start_elem + src_num_elems));

        let dst_start_byte = rows_processed * dst_row_size_bytes;
        let dst_num_bytes = rows_in_chunk * dst_row_size_bytes;
        let dst_chunk = dst.slice(dst_start_byte..(dst_start_byte + dst_num_bytes));

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_blocks as u32, rows_in_chunk as u32, 1),
            block_dim: (CUDA_QUANTIZE_BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = func.builder();
        builder.arg(&src_chunk);
        builder.arg(&dst_chunk);
        barg!(builder, k as i32, kx_padded as i32);
        unsafe { builder.launch(cfg) }.w()?;

        rows_processed += rows_in_chunk;
    }

    Ok(())
}

/// Attention-score gemv: `Q @ K^T` where K is a Q8_0 blob (row-major over
/// tokens, stride `n_kv_stride_blocks` block_q8_0 units) and Q is per-step
/// queries on-device in f32.
///
/// Q is quantized to Q8_1 internally via the existing GPU quantize kernel
/// before dispatch. Use this instead of `mul_mat_vec_via_q8_1` when k is
/// small (head_dim 64/128/256) and n is large (up to 128K) — the generic
/// gemv is tuned for large-k LLM weights and runs 5-7× slower than F16 at
/// attention score shapes.
///
/// Returns a contiguous `float[b_size, n_kv]` tensor.
///
/// `n_kv_stride_blocks` allows striding between K rows when the caller
/// packs a multi-kv-head cache as `[seq, n_kv_heads, head_dim]` — in that
/// case pass `n_kv_heads * head_dim / 32` so the kernel hops over the
/// sibling kv-heads of each token.
pub fn attn_score_q8_0_q8_1_raw(
    k_blob: &CudaSlice<u8>,
    k_byte_offset: usize,
    n_kv_stride_blocks: usize,
    q_f32: &CudaView<f32>,
    head_dim: usize,
    n_kv: usize,
    b_size: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    if !head_dim.is_multiple_of(32) {
        crate::bail!("attn_score_q8_0_q8_1: head_dim {head_dim} not a multiple of 32");
    }
    if q_f32.len() != b_size * head_dim {
        crate::bail!(
            "attn_score_q8_0_q8_1: q_f32 has {} elements, expected {}",
            q_f32.len(),
            b_size * head_dim
        );
    }

    // Step 1: quantize Q to Q8_1 on-device (hot path allocation).
    let k_padded = pad(head_dim, MATRIX_ROW_PADDING);
    let q_q8_1_bytes =
        k_padded * b_size * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
    let mut q_q8_1 = unsafe { dev.alloc::<u8>(q_q8_1_bytes)? };
    quantize_q8_1(q_f32, &mut q_q8_1, head_dim, b_size, dev)?;
    let kernel_name = match head_dim {
        64 => "attn_score_q8_0_q8_1_hd64",
        128 => "attn_score_q8_0_q8_1_hd128",
        256 => "attn_score_q8_0_q8_1_hd256",
        _ => crate::bail!(
            "attn_score_q8_0_q8_1: unsupported head_dim {head_dim} (have 64/128/256)",
        ),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;

    // Q stride between consecutive batch rows, in `block_q8_1` units.
    // `quantize_q8_1` pads each row up to MATRIX_ROW_PADDING elements, so
    // the physical stride is (k_padded / block_size), not hd_blocks.
    let b_stride_blocks = k_padded / GgmlDType::Q8_1.block_size();

    // One warp per n row; batches iterate inside the kernel so K is loaded
    // once per n and reused across all queries. With 4 warps per block, 128
    // threads per block — good occupancy without over-subscribing regs.
    const WARPS_PER_BLOCK: u32 = 4;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: ((n_kv as u32).div_ceil(WARPS_PER_BLOCK), 1, 1),
        block_dim: (WARP_SIZE as u32, WARPS_PER_BLOCK, 1),
        shared_mem_bytes: 0,
    };

    let dst = unsafe { dev.alloc::<f32>(b_size * n_kv)? };
    // Slice K starting at the caller-supplied byte offset.
    let k_slice = if k_byte_offset == 0 {
        k_blob.slice(..)
    } else {
        k_blob.slice(k_byte_offset..)
    };
    let mut builder = func.builder();
    builder.arg(&k_slice);
    builder.arg(&q_q8_1);
    builder.arg(&dst);
    barg!(
        builder,
        n_kv as i32,
        n_kv_stride_blocks as i32,
        b_stride_blocks as i32,
        b_size as i32
    );
    unsafe { builder.launch(cfg) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

/// GQA variant: compute attention scores for **all** kv-heads in a single
/// kernel launch.
///
/// K layout: `[seq_kv, n_kv_heads, head_dim]` as Q8_0 blocks — the packing
/// used by `Q8KvCache`. Pass `n_kv_stride_blocks = n_kv_heads * head_dim / 32`
/// (total token stride) and the function derives the per-kv-head offset
/// internally.
///
/// Q layout: `[n_q_heads, head_dim]` f32 on-device. `n_q_heads` must equal
/// `n_kv_heads * n_q_per_kv`. Q is quantized to Q8_1 once, re-used across
/// all kv-heads.
///
/// Output: `float[n_q_heads, n_kv]` scores. Query head `h` corresponds to
/// kv-head `h / n_q_per_kv`.
#[allow(clippy::too_many_arguments)]
pub fn attn_score_q8_0_q8_1_gqa(
    k_blob: &CudaSlice<u8>,
    q_f32: &CudaView<f32>,
    head_dim: usize,
    n_kv: usize,
    n_kv_heads: usize,
    n_q_per_kv: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    if !head_dim.is_multiple_of(32) {
        crate::bail!("attn_score_q8_0_q8_1_gqa: head_dim {head_dim} not a multiple of 32");
    }
    let n_q_heads = n_kv_heads * n_q_per_kv;
    if q_f32.len() != n_q_heads * head_dim {
        crate::bail!(
            "attn_score_q8_0_q8_1_gqa: q_f32 has {} elements, expected {}",
            q_f32.len(),
            n_q_heads * head_dim
        );
    }
    let kernel_name = match head_dim {
        64 => "attn_score_q8_0_q8_1_gqa_hd64",
        128 => "attn_score_q8_0_q8_1_gqa_hd128",
        256 => "attn_score_q8_0_q8_1_gqa_hd256",
        _ => crate::bail!(
            "attn_score_q8_0_q8_1_gqa: unsupported head_dim {head_dim} (have 64/128/256)",
        ),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;

    // Quantize Q to Q8_1 once for all kv-heads.
    let k_padded = pad(head_dim, MATRIX_ROW_PADDING);
    let q_q8_1_bytes =
        k_padded * n_q_heads * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
    let mut q_q8_1 = unsafe { dev.alloc::<u8>(q_q8_1_bytes)? };
    quantize_q8_1(q_f32, &mut q_q8_1, head_dim, n_q_heads, dev)?;

    let hd_blocks = head_dim / 32;
    let n_kv_stride_blocks = n_kv_heads * hd_blocks; // full token stride
    let kv_head_stride_blocks = hd_blocks;
    let q_stride_blocks = k_padded / GgmlDType::Q8_1.block_size();

    const WARPS_PER_BLOCK: u32 = 4;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (
            (n_kv as u32).div_ceil(WARPS_PER_BLOCK),
            n_kv_heads as u32,
            1,
        ),
        block_dim: (WARP_SIZE as u32, WARPS_PER_BLOCK, 1),
        shared_mem_bytes: 0,
    };

    let dst = unsafe { dev.alloc::<f32>(n_q_heads * n_kv)? };
    let mut builder = func.builder();
    builder.arg(k_blob);
    builder.arg(&q_q8_1);
    builder.arg(&dst);
    barg!(
        builder,
        n_kv as i32,
        n_kv_stride_blocks as i32,
        kv_head_stride_blocks as i32,
        q_stride_blocks as i32,
        n_q_per_kv as i32
    );
    unsafe { builder.launch(cfg) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

/// V-path attention output: `out = probs @ V` where V is a Q8_0 cache laid
/// out `[seq_kv, n_kv_heads, head_dim]` (same packing as `Q8KvCache`).
///
/// `probs` is f32 `[n_q_heads, seq_kv]` (softmax output). `n_q_heads` must
/// equal `n_kv_heads * n_q_per_kv`. Returns f32 `[n_q_heads, head_dim]`.
///
/// The reduction is along `seq_kv`, orthogonal to Q8_0's head_dim block
/// direction — so we can't reuse candle's standard `mul_mat_vec_q` kernels
/// (which reduce along the block axis). Each lane owns one head_dim output
/// and sweeps seq_kv internally.
#[allow(clippy::too_many_arguments)]
pub fn attn_output_q8_0_f32_gqa(
    v_blob: &CudaSlice<u8>,
    probs_f32: &CudaView<f32>,
    head_dim: usize,
    seq_kv: usize,
    n_kv_heads: usize,
    n_q_per_kv: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    if !head_dim.is_multiple_of(32) {
        crate::bail!("attn_output_q8_0_f32_gqa: head_dim {head_dim} not a multiple of 32");
    }
    let n_q_heads = n_kv_heads * n_q_per_kv;
    if probs_f32.len() != n_q_heads * seq_kv {
        crate::bail!(
            "attn_output_q8_0_f32_gqa: probs has {} elements, expected {}",
            probs_f32.len(),
            n_q_heads * seq_kv
        );
    }
    let kernel_name = match (head_dim, n_q_per_kv) {
        (64, 1) => "attn_output_q8_0_f32_hd64_nq1",
        (64, 4) => "attn_output_q8_0_f32_hd64_nq4",
        (64, 8) => "attn_output_q8_0_f32_hd64_nq8",
        (128, 1) => "attn_output_q8_0_f32_hd128_nq1",
        (128, 4) => "attn_output_q8_0_f32_hd128_nq4",
        (128, 8) => "attn_output_q8_0_f32_hd128_nq8",
        (256, 1) => "attn_output_q8_0_f32_hd256_nq1",
        (256, 4) => "attn_output_q8_0_f32_hd256_nq4",
        (256, 8) => "attn_output_q8_0_f32_hd256_nq8",
        _ => crate::bail!(
            "attn_output_q8_0_f32_gqa: unsupported combo head_dim={head_dim} n_q_per_kv={n_q_per_kv} (want {{64,128,256}} × {{1,4,8}})",
        ),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;

    let hd_blocks = head_dim / 32;
    let n_kv_stride_blocks = n_kv_heads * hd_blocks;
    let kv_head_stride_blocks = hd_blocks;

    // Keep this in lock-step with ATTN_OUTPUT_WARPS in the .cu file.
    // qh is iterated inside the kernel so V is loaded once per (kv, block_idx)
    // tile and reused across n_q_per_kv query heads — critical for GQA
    // groups with n_q_per_kv > 1 (qwen3 = 4, qwen3-coder = 8, Llama-3 70B = 8).
    const WARPS_PER_BLOCK: u32 = 32;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (hd_blocks as u32, n_kv_heads as u32, 1),
        block_dim: (WARP_SIZE as u32, WARPS_PER_BLOCK, 1),
        shared_mem_bytes: 0,
    };

    let dst = unsafe { dev.alloc::<f32>(n_q_heads * head_dim)? };
    let mut builder = func.builder();
    builder.arg(v_blob);
    builder.arg(probs_f32);
    builder.arg(&dst);
    barg!(
        builder,
        seq_kv as i32,
        n_kv_stride_blocks as i32,
        kv_head_stride_blocks as i32,
        n_q_per_kv as i32
    );
    unsafe { builder.launch(cfg) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

/// Q4_0 K-path attention score, KIVI layout.
///
/// K state is split between two buffers:
///   - `k_blocks`: `[seq_blocks, n_kv, head_dim]` of 18-byte Q4_0 cells.
///     Each cell holds 32 consecutive seq positions of one (head, channel).
///   - `k_residual`: `[n_kv, head_dim, 32]` f16 — the not-yet-flushed
///     partial seq block. Values at slot `s` correspond to seq position
///     `full_blocks * 32 + s`.
///
/// Q is f32 `[n_q_heads, head_dim]`. Returns f32 `[n_q_heads, seq_kv]`
/// scores covering the full `seq_kv` range (flushed blocks + residual).
///
/// The kernel handles the partial block internally — no separate host-side
/// matmul + concat is needed. Grid dimension 0 covers `ceil(seq_kv/32)`
/// blocks; the last block is the partial one when `seq_kv % 32 != 0`.
#[allow(clippy::too_many_arguments)]
pub fn attn_score_q4_0_f32_kivi(
    k_blocks: &CudaSlice<u8>,
    k_residual: &CudaSlice<f16>,
    q_f32: &CudaView<f32>,
    head_dim: usize,
    seq_kv: usize,
    n_kv_heads: usize,
    n_q_per_kv: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    if !head_dim.is_multiple_of(32) {
        crate::bail!("attn_score_q4_0_f32_kivi: head_dim {head_dim} not a multiple of 32");
    }
    let n_q_heads = n_kv_heads * n_q_per_kv;
    if q_f32.len() != n_q_heads * head_dim {
        crate::bail!(
            "attn_score_q4_0_f32_kivi: Q has {} elems, expected {}",
            q_f32.len(),
            n_q_heads * head_dim
        );
    }
    let full_blocks = seq_kv / 32;
    let total_blocks = seq_kv.div_ceil(32);
    let kernel_name = match (head_dim, n_q_per_kv) {
        (64, 1) => "attn_score_q4_0_f32_kivi_hd64_nq1",
        (64, 4) => "attn_score_q4_0_f32_kivi_hd64_nq4",
        (64, 8) => "attn_score_q4_0_f32_kivi_hd64_nq8",
        (128, 1) => "attn_score_q4_0_f32_kivi_hd128_nq1",
        (128, 4) => "attn_score_q4_0_f32_kivi_hd128_nq4",
        (128, 8) => "attn_score_q4_0_f32_kivi_hd128_nq8",
        (256, 1) => "attn_score_q4_0_f32_kivi_hd256_nq1",
        (256, 4) => "attn_score_q4_0_f32_kivi_hd256_nq4",
        (256, 8) => "attn_score_q4_0_f32_kivi_hd256_nq8",
        _ => crate::bail!(
            "attn_score_q4_0_f32_kivi: unsupported (head_dim={head_dim}, n_q_per_kv={n_q_per_kv})",
        ),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;

    // Grid = (total_blocks, n_kv_heads). Last block of the x axis is the
    // partial residual block when seq_kv is not a multiple of 32; the
    // kernel handles it via the `sb == full_blocks` path.
    const N_WARPS: u32 = 16;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (total_blocks as u32, n_kv_heads as u32, 1),
        block_dim: (WARP_SIZE as u32, N_WARPS, 1),
        shared_mem_bytes: (N_WARPS * WARP_SIZE as u32 * 4) as u32,
    };

    let dst = unsafe { dev.alloc::<f32>(n_q_heads * seq_kv)? };
    let mut builder = func.builder();
    builder.arg(k_blocks);
    builder.arg(k_residual);
    builder.arg(q_f32);
    builder.arg(&dst);
    barg!(
        builder,
        seq_kv as i32,
        full_blocks as i32,
        n_kv_heads as i32,
        n_q_per_kv as i32
    );
    unsafe { builder.launch(cfg) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

/// Q4_0 variant of `attn_output_q8_0_f32_gqa`. Same call shape, V laid out
/// `[seq_kv, n_kv_heads, head_dim]` in 18-byte Q4_0 blocks (half the byte
/// count of Q8_0 for the same element count).
#[allow(clippy::too_many_arguments)]
pub fn attn_output_q4_0_f32_gqa(
    v_blob: &CudaSlice<u8>,
    probs_f32: &CudaView<f32>,
    head_dim: usize,
    seq_kv: usize,
    n_kv_heads: usize,
    n_q_per_kv: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    if !head_dim.is_multiple_of(32) {
        crate::bail!("attn_output_q4_0_f32_gqa: head_dim {head_dim} not a multiple of 32");
    }
    let n_q_heads = n_kv_heads * n_q_per_kv;
    if probs_f32.len() != n_q_heads * seq_kv {
        crate::bail!(
            "attn_output_q4_0_f32_gqa: probs has {} elements, expected {}",
            probs_f32.len(),
            n_q_heads * seq_kv
        );
    }
    let kernel_name = match (head_dim, n_q_per_kv) {
        (64, 1) => "attn_output_q4_0_f32_hd64_nq1",
        (64, 4) => "attn_output_q4_0_f32_hd64_nq4",
        (64, 8) => "attn_output_q4_0_f32_hd64_nq8",
        (128, 1) => "attn_output_q4_0_f32_hd128_nq1",
        (128, 4) => "attn_output_q4_0_f32_hd128_nq4",
        (128, 8) => "attn_output_q4_0_f32_hd128_nq8",
        (256, 1) => "attn_output_q4_0_f32_hd256_nq1",
        (256, 4) => "attn_output_q4_0_f32_hd256_nq4",
        (256, 8) => "attn_output_q4_0_f32_hd256_nq8",
        _ => crate::bail!(
            "attn_output_q4_0_f32_gqa: unsupported combo head_dim={head_dim} n_q_per_kv={n_q_per_kv}",
        ),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;

    let hd_blocks = head_dim / 32;
    let n_kv_stride_blocks = n_kv_heads * hd_blocks;
    let kv_head_stride_blocks = hd_blocks;

    const WARPS_PER_BLOCK: u32 = 32;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (hd_blocks as u32, n_kv_heads as u32, 1),
        block_dim: (WARP_SIZE as u32, WARPS_PER_BLOCK, 1),
        shared_mem_bytes: 0,
    };

    let dst = unsafe { dev.alloc::<f32>(n_q_heads * head_dim)? };
    let mut builder = func.builder();
    builder.arg(v_blob);
    builder.arg(probs_f32);
    builder.arg(&dst);
    barg!(
        builder,
        seq_kv as i32,
        n_kv_stride_blocks as i32,
        kv_head_stride_blocks as i32,
        n_q_per_kv as i32
    );
    unsafe { builder.launch(cfg) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

/// Convenience: call the attention-score gemv against a pre-built QCudaStorage
/// K. Assumes the K cache is laid out row-major over tokens with stride
/// `head_dim / 32` blocks (the layout produced by calling `QTensor::quantize`
/// on a `[seq, head_dim]` tensor).
pub fn attn_score_q8_0_q8_1(
    k: &QCudaStorage,
    q_f32: &CudaView<f32>,
    head_dim: usize,
    n_kv: usize,
    b_size: usize,
) -> Result<CudaStorage> {
    if k.dtype != GgmlDType::Q8_0 {
        crate::bail!("attn_score_q8_0_q8_1: K must be Q8_0, got {:?}", k.dtype);
    }
    let hd_blocks = head_dim / 32;
    attn_score_q8_0_q8_1_raw(
        &k.data.inner,
        0,
        hd_blocks,
        q_f32,
        head_dim,
        n_kv,
        b_size,
        &k.device,
    )
}

fn dequantize_f32(
    data: &PaddedCudaSlice,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let nb = elem_count.div_ceil(256);
    let (kernel_name, is_k, block_dim, num_blocks) = match dtype {
        GgmlDType::Q4_0 => ("dequantize_block_q4_0_f32", false, 32, nb),
        GgmlDType::Q4_1 => ("dequantize_block_q4_1_f32", false, 32, nb),
        GgmlDType::Q5_0 => (
            "dequantize_block_q5_0_f32",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q5_1 => (
            "dequantize_block_q5_1_f32",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q8_0 => ("dequantize_block_q8_0_f32", false, 32, nb),
        GgmlDType::Q2K => ("dequantize_block_q2_K_f32", true, 64, nb),
        GgmlDType::Q3K => ("dequantize_block_q3_K_f32", true, 64, nb),
        GgmlDType::Q4K => ("dequantize_block_q4_K_f32", true, 32, nb),
        GgmlDType::Q5K => ("dequantize_block_q5_K_f32", true, 64, nb),
        GgmlDType::Q6K => ("dequantize_block_q6_K_f32", true, 64, nb),
        GgmlDType::Q8K => ("dequantize_block_q8_K_f32", true, 32, nb),
        _ => crate::bail!("unsupported dtype for dequantize {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f32>(elem_count)? };
    // See e.g.
    // https://github.com/ggerganov/llama.cpp/blob/cbbd1efa06f8c09f9dff58ff9d9af509cc4c152b/ggml-cuda.cu#L7270
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (block_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    if is_k {
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        unsafe { builder.launch(cfg) }.w()?;
    } else {
        let nb32 = match dtype {
            GgmlDType::Q5_0 | GgmlDType::Q5_1 => elem_count,
            _ => elem_count / 32,
        };
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        barg!(builder, nb32 as i32);
        unsafe { builder.launch(cfg) }.w()?;
    }
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

/// Dequantize a Q8_0 byte blob straight to an F16 CudaStorage. Intended
/// for callers that hold a raw `CudaSlice<u8>` of Q8_0 blocks (for
/// example a KV cache that owns the storage directly rather than via a
/// QCudaStorage/QTensor) and need an f16 view for downstream attention
/// kernels (like flash-attn) that don't consume quantized inputs.
///
/// `elem_count` is the logical (dequantized) element count — must be a
/// multiple of 32 (Q8_0 block size).
pub fn dequantize_q8_0_blob_f16(
    data: &CudaSlice<u8>,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    if !elem_count.is_multiple_of(32) {
        crate::bail!("dequantize_q8_0_blob_f16: elem_count {elem_count} must be multiple of 32");
    }
    let nb = elem_count.div_ceil(256);
    let func = dev.get_or_load_func("dequantize_block_q8_0_f16", &candle_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f16>(elem_count)? };
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (nb as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let nb32 = (elem_count / 32) as i32;
    let mut builder = func.builder();
    builder.arg(data);
    builder.arg(&dst);
    barg!(builder, nb32);
    unsafe { builder.launch(cfg) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

/// Dequantize a Q4_0 byte blob to F16. Mirror of `dequantize_q8_0_blob_f16`
/// for the Q4 KV cache: each 18-byte block becomes 32 F16 elements. Used by
/// the "dequant then standard attention" fallback path until the fused Q4
/// attention kernels land.
pub fn dequantize_q4_0_blob_f16(
    data: &CudaSlice<u8>,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    if !elem_count.is_multiple_of(32) {
        crate::bail!("dequantize_q4_0_blob_f16: elem_count {elem_count} must be multiple of 32");
    }
    let nb = elem_count.div_ceil(256);
    let func = dev.get_or_load_func("dequantize_block_q4_0_f16", &candle_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f16>(elem_count)? };
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (nb as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let nb32 = (elem_count / 32) as i32;
    let mut builder = func.builder();
    builder.arg(data);
    builder.arg(&dst);
    barg!(builder, nb32);
    unsafe { builder.launch(cfg) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

fn dequantize_f16(
    data: &PaddedCudaSlice,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let nb = elem_count.div_ceil(256);
    let (kernel_name, is_k, block_dim, num_blocks) = match dtype {
        GgmlDType::Q4_0 => ("dequantize_block_q4_0_f16", false, 32, nb),
        GgmlDType::Q4_1 => ("dequantize_block_q4_1_f16", false, 32, nb),
        GgmlDType::Q5_0 => (
            "dequantize_block_q5_0_f16",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q5_1 => (
            "dequantize_block_q5_1_f16",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q8_0 => ("dequantize_block_q8_0_f16", false, 32, nb),
        GgmlDType::Q2K => ("dequantize_block_q2_K_f16", true, 64, nb),
        GgmlDType::Q3K => ("dequantize_block_q3_K_f16", true, 64, nb),
        GgmlDType::Q4K => ("dequantize_block_q4_K_f16", true, 32, nb),
        GgmlDType::Q5K => ("dequantize_block_q5_K_f16", true, 64, nb),
        GgmlDType::Q6K => ("dequantize_block_q6_K_f16", true, 64, nb),
        GgmlDType::Q8K => ("dequantize_block_q8_K_f16", true, 32, nb),
        _ => crate::bail!("unsupported dtype for dequantize {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f16>(elem_count)? };
    // See e.g.
    // https://github.com/ggerganov/llama.cpp/blob/cbbd1efa06f8c09f9dff58ff9d9af509cc4c152b/ggml-cuda.cu#L7270
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (block_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    if is_k {
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        unsafe { builder.launch(cfg) }.w()?;
    } else {
        let nb32 = match dtype {
            GgmlDType::Q5_0 | GgmlDType::Q5_1 => elem_count,
            _ => elem_count / 32,
        };
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        barg!(builder, nb32 as i32);
        unsafe { builder.launch(cfg) }.w()?;
    }
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

fn dequantize_mul_mat_vec(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < ncols * nrows {
        crate::bail!("unexpected data size {}, ncols {ncols} {nrows}", data_elems)
    }
    if y.len() != ncols {
        crate::bail!("unexpected y size {}, ncols {ncols} {nrows}", y.len())
    }
    let kernel_name = match dtype {
        GgmlDType::Q4_0 => "dequantize_mul_mat_vec_q4_0_cuda",
        GgmlDType::Q4_1 => "dequantize_mul_mat_vec_q4_1_cuda",
        GgmlDType::Q5_0 => "dequantize_mul_mat_vec_q5_0_cuda",
        GgmlDType::Q5_1 => "dequantize_mul_mat_vec_q5_1_cuda",
        GgmlDType::Q8_0 => "dequantize_mul_mat_vec_q8_0_cuda",
        GgmlDType::Q2K => "dequantize_mul_mat_vec_q2_k",
        GgmlDType::Q3K => "dequantize_mul_mat_vec_q3_k",
        GgmlDType::Q4K => "dequantize_mul_mat_vec_q4_k",
        GgmlDType::Q5K => "dequantize_mul_mat_vec_q5_k",
        GgmlDType::Q6K => "dequantize_mul_mat_vec_q6_k",
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f32>(nrows)? };
    let block_num_y = ceil_div(nrows, GGML_CUDA_MMV_Y);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (block_num_y as u32, 1, 1),
        block_dim: (WARP_SIZE as u32, GGML_CUDA_MMV_Y as u32, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(&data.inner);
    builder.arg(y);
    builder.arg(&dst);
    barg!(builder, ncols as i32, nrows as i32);
    unsafe { builder.launch(cfg) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

fn mul_mat_vec_via_q8_1(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    b_size: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < ncols * nrows {
        crate::bail!("unexpected data size {}, ncols {ncols} {nrows}", data_elems)
    }
    if y.len() != ncols * b_size {
        crate::bail!("unexpected y size {}, ncols {ncols} {nrows}", y.len())
    }
    if b_size == 0 || b_size > 8 {
        crate::bail!("only bsize between 1 and 8 are supported, got {b_size}")
    }
    // Start by quantizing y
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let y_size_in_bytes =
        b_size * ncols_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
    let mut y_q8_1 = unsafe { dev.alloc::<u8>(y_size_in_bytes)? };
    quantize_q8_1(y, &mut y_q8_1, ncols, b_size, dev)?;

    let kernel_name = match dtype {
        GgmlDType::Q4_0 => "mul_mat_vec_q4_0_q8_1_cuda",
        GgmlDType::Q4_1 => "mul_mat_vec_q4_1_q8_1_cuda",
        GgmlDType::Q5_0 => "mul_mat_vec_q5_0_q8_1_cuda",
        GgmlDType::Q5_1 => "mul_mat_vec_q5_1_q8_1_cuda",
        GgmlDType::Q8_0 => "mul_mat_vec_q8_0_q8_1_cuda",
        GgmlDType::Q2K => "mul_mat_vec_q2_K_q8_1_cuda",
        GgmlDType::Q3K => "mul_mat_vec_q3_K_q8_1_cuda",
        GgmlDType::Q4K => "mul_mat_vec_q4_K_q8_1_cuda",
        GgmlDType::Q5K => "mul_mat_vec_q5_K_q8_1_cuda",
        GgmlDType::Q6K => "mul_mat_vec_q6_K_q8_1_cuda",
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    let kernel_name = format!("{kernel_name}{b_size}");
    let func = dev.get_or_load_func(&kernel_name, &candle_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f32>(nrows * b_size)? };
    // https://github.com/ggerganov/llama.cpp/blob/facb8b56f8fd3bb10a693bf0943ae9d69d0828ef/ggml-cuda/mmvq.cu#L98
    let (nblocks, nwarps) = match b_size {
        1 => (nrows as u32, 4),
        2..=4 => ((nrows as u32).div_ceil(2), 4),
        5..=8 => ((nrows as u32).div_ceil(2), 2),
        _ => crate::bail!("unexpected bsize {b_size}"),
    };
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (nblocks, 1, 1),
        block_dim: (WARP_SIZE as u32, nwarps, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(&data.inner);
    builder.arg(&y_q8_1);
    builder.arg(&dst);
    barg!(
        builder,
        /* ncols_x */ ncols as i32,
        /* nrows_x */ nrows as i32,
        /* nrows_y */ ncols_padded as i32,
        /* nrows_dst */ nrows as i32
    );
    unsafe { builder.launch(cfg) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

#[allow(clippy::too_many_arguments)]
fn mul_mat_via_q8_1(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    x_rows: usize,
    x_cols: usize,
    y_rows: usize,
    y_cols: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < x_rows * x_cols {
        crate::bail!("unexpected lhs size {}, {x_rows} {x_cols}", data_elems)
    }
    if y.len() != y_rows * y_cols {
        crate::bail!("unexpected y size {}, {y_rows} {y_cols}", y.len())
    }
    if x_cols != y_rows {
        crate::bail!("unexpected x/y size {x_rows} {x_cols} {y_rows} {y_cols}")
    }
    let k = x_cols;
    // Start by quantizing y
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let y_size_in_bytes =
        k_padded * y_cols * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
    let mut y_q8_1 = unsafe { dev.alloc::<u8>(y_size_in_bytes)? };
    quantize_q8_1(y, &mut y_q8_1, k, y_cols, dev)?;

    let (kernel_name, mmq_x, mmq_y) = match dtype {
        GgmlDType::Q4_0 => ("mul_mat_q4_0", 64, 128),
        GgmlDType::Q4_1 => ("mul_mat_q4_1", 64, 128),
        GgmlDType::Q5_0 => ("mul_mat_q5_0", 128, 64),
        GgmlDType::Q5_1 => ("mul_mat_q5_1", 128, 64),
        GgmlDType::Q8_0 => ("mul_mat_q8_0", 128, 64),
        GgmlDType::Q2K => ("mul_mat_q2_K", 64, 128),
        GgmlDType::Q3K => ("mul_mat_q3_K", 128, 128),
        GgmlDType::Q4K => ("mul_mat_q4_K", 64, 128),
        GgmlDType::Q5K => ("mul_mat_q5_K", 64, 128),
        GgmlDType::Q6K => ("mul_mat_q6_K", 64, 64),
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f32>(x_rows * y_cols)? };
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (
            ceil_div(x_rows, mmq_y) as u32,
            ceil_div(y_cols, mmq_x) as u32,
            1,
        ),
        block_dim: (WARP_SIZE as u32, 4, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(/* vx */ &data.inner);
    builder.arg(/* vy */ &y_q8_1);
    builder.arg(/* dst */ &dst);
    barg!(
        builder,
        /* ncols_x */ x_cols as i32,
        /* nrows_x */ x_rows as i32,
        /* ncols_y */ y_cols as i32,
        /* nrows_y */ k_padded as i32,
        /* nrows_dst */ x_rows as i32
    );
    unsafe { builder.launch(cfg) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

#[allow(clippy::too_many_arguments)]
fn indexed_moe_forward_fused_q8_1_input(
    weight: &CudaView<u8>,
    w_shape: &crate::Shape, //[num_experts, n, k]
    w_dtype: GgmlDType,
    input: &CudaSlice<f32>,
    in_shape: &crate::Shape, //[batch, topk or 1, k]
    ids: &CudaView<u32>,
    idx_shape: &crate::Shape, //[batch, topk]
    dev: &CudaDevice,
) -> Result<(CudaStorage, crate::Shape)> {
    let (_, n, k) = w_shape.dims3()?;
    let batch = in_shape.dims()[0];
    let input_dim1 = in_shape.dims()[1];

    let topk = idx_shape.dims()[1];
    assert!(batch == idx_shape.dims()[0], "batch dim not match!");

    // Quantize input into q8_1.
    let total_rows = batch * input_dim1;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    // Get Q8_1 metadata.
    let q8_1_block_size = GgmlDType::Q8_1.block_size();
    let q8_1_type_size = GgmlDType::Q8_1.type_size();

    // Calculate the size of the output buffer in bytes.
    let num_blocks_per_row = k_padded / q8_1_block_size;
    let dst_row_size_bytes = num_blocks_per_row * q8_1_type_size;
    let y_size_in_bytes = total_rows * dst_row_size_bytes;
    let mut input_quant = unsafe { dev.alloc::<u8>(y_size_in_bytes)? };

    let input_view = input.slice(0..);
    quantize_q8_1(&input_view, &mut input_quant, k, total_rows, dev)?;

    // output buffer
    let outsize = batch * topk * n;
    let out = unsafe { dev.alloc::<f32>(outsize)? };

    let kernel_name = match w_dtype {
        GgmlDType::Q2K => "indexed_moe_forward_q2k_q8_1",
        GgmlDType::Q3K => "indexed_moe_forward_q3k_q8_1",
        GgmlDType::Q4K => "indexed_moe_forward_q4k_q8_1",
        GgmlDType::Q5K => "indexed_moe_forward_q5k_q8_1",
        GgmlDType::Q6K => "indexed_moe_forward_q6k_q8_1",
        GgmlDType::Q8_0 => "indexed_moe_forward_q8_0_q8_1",
        _ => crate::bail!("unsupported dtype for indexed_moe_forward {w_dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;
    let (nblocks, nwarps) = (n as u32, 4);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (nblocks, batch as u32, topk as u32),
        block_dim: (WARP_SIZE as u32, nwarps, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(weight);
    builder.arg(&input_quant);
    builder.arg(ids);
    builder.arg(&out);

    barg!(
        builder,
        n as i32,
        k as i32,
        batch as i32,
        topk as i32,
        k_padded as i32,
        input_dim1 as i32
    );
    unsafe { builder.launch(cfg) }.w()?;

    let mut out_shape = in_shape.dims().to_vec();
    out_shape.pop();
    out_shape.push(n);
    out_shape[1] = topk;
    Ok((
        CudaStorage::wrap_cuda_slice(out, dev.clone()),
        out_shape.into(),
    ))
}

impl QCudaStorage {
    pub fn indexed_moe_forward(
        &self,
        self_shape: &crate::Shape, //[num_experts, n, k]
        input: &CudaStorage,       //[batch, topk or 1, k]
        input_l: &crate::Layout,
        ids: &CudaStorage, //[batch, topk]
        ids_l: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        if matches!(
            self.dtype(),
            GgmlDType::Q8_0
                | GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
        ) {
            let input_storage = input.as_cuda_slice::<f32>()?;
            let ids_storage = ids.as_cuda_slice::<u32>()?;
            indexed_moe_forward_fused_q8_1_input(
                &self.data.inner.slice(0..),
                self_shape, //[num_experts, n, k]
                self.dtype(),
                input_storage,
                input_l.shape(), //[batch, topk or 1, k]
                &ids_storage.slice(0..),
                ids_l.shape(), //[batch, topk]
                &self.device,
            )
        } else {
            crate::bail!(
                "The given quantized dtype {:?} is not supported for indexed_moe_forward!",
                self.dtype()
            );
        }
    }

    pub fn zeros(device: &CudaDevice, el_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size_in_bytes = ceil_div(el_count, dtype.block_size()) * dtype.type_size();
        let padded_size_in_bytes =
            ceil_div(el_count + MATRIX_ROW_PADDING, dtype.block_size()) * dtype.type_size();
        let inner = device.alloc_zeros::<u8>(padded_size_in_bytes)?;
        Ok(QCudaStorage {
            data: PaddedCudaSlice {
                inner,
                len: size_in_bytes,
            },
            device: device.clone(),
            dtype,
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<CudaStorage> {
        fn deq<T: GgmlType>(buffer: &[u8], n: usize, dst: &mut [f32]) {
            let slice = unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const T, n) };
            let vec = slice.to_vec();
            T::to_float(&vec, dst)
        }

        let fast_kernel = matches!(
            self.dtype,
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
        );
        if fast_kernel {
            return dequantize_f32(&self.data, self.dtype, elem_count, self.device());
        }
        // Run the dequantization on cpu.

        let buffer = self
            .device
            .clone_dtoh(&self.data.inner.slice(..self.data.len))?;
        let mut out = vec![0.0; elem_count];
        let block_len = elem_count / self.dtype.block_size();
        match self.dtype {
            GgmlDType::F32 => deq::<f32>(&buffer, block_len, &mut out),
            GgmlDType::F16 => deq::<half::f16>(&buffer, block_len, &mut out),
            GgmlDType::BF16 => deq::<half::bf16>(&buffer, block_len, &mut out),
            GgmlDType::Q4_0 => deq::<crate::quantized::BlockQ4_0>(&buffer, block_len, &mut out),
            GgmlDType::Q4_1 => deq::<crate::quantized::BlockQ4_1>(&buffer, block_len, &mut out),
            GgmlDType::Q5_0 => deq::<crate::quantized::BlockQ5_0>(&buffer, block_len, &mut out),
            GgmlDType::Q5_1 => deq::<crate::quantized::BlockQ5_1>(&buffer, block_len, &mut out),
            GgmlDType::Q8_0 => deq::<crate::quantized::BlockQ8_0>(&buffer, block_len, &mut out),
            GgmlDType::Q8_1 => deq::<crate::quantized::BlockQ8_1>(&buffer, block_len, &mut out),
            GgmlDType::Q2K => deq::<crate::quantized::BlockQ2K>(&buffer, block_len, &mut out),
            GgmlDType::Q3K => deq::<crate::quantized::BlockQ3K>(&buffer, block_len, &mut out),
            GgmlDType::Q4K => deq::<crate::quantized::BlockQ4K>(&buffer, block_len, &mut out),
            GgmlDType::Q5K => deq::<crate::quantized::BlockQ5K>(&buffer, block_len, &mut out),
            GgmlDType::Q6K => deq::<crate::quantized::BlockQ6K>(&buffer, block_len, &mut out),
            GgmlDType::Q8K => deq::<crate::quantized::BlockQ8K>(&buffer, block_len, &mut out),
        }

        self.device
            .storage_from_cpu_storage(&crate::CpuStorage::F32(out))
    }

    pub fn dequantize_f16(&self, elem_count: usize) -> Result<CudaStorage> {
        dequantize_f16(&self.data, self.dtype, elem_count, self.device())
    }

    /// Like `quantize(&src)` but respects the source tensor's layout —
    /// specifically the `start_offset` and element count implied by the
    /// layout's shape. The plain `quantize` reads from raw storage offset
    /// 0 with `src.data.len()` elements, which is wrong when `src` is a
    /// view into a larger storage (`narrow`, `slice`, etc.). Callers that
    /// have a `Tensor` should prefer this variant.
    pub fn quantize_with_layout(
        &mut self,
        src: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<()> {
        let offset = layout.start_offset();
        let elem_count = layout.shape().elem_count();

        // Q8_0 / Q4_0 share a fast path: GPU-native quantize kernels that
        // read the source layout directly into a PaddedCudaSlice, without
        // the dtoh→quantize→htod roundtrip the CPU fallback does. Both are
        // used by the KV cache on every decode step, so avoiding the host
        // stall is the point of the fast path.
        if self.dtype == GgmlDType::Q8_0 || self.dtype == GgmlDType::Q4_0 {
            let is_q4 = self.dtype == GgmlDType::Q4_0;
            let block_size = self.dtype.block_size();
            let type_size = self.dtype.type_size();
            if !elem_count.is_multiple_of(block_size) {
                crate::bail!(
                    "quantize_{}: element count {elem_count} is not divisible by block size {block_size}",
                    if is_q4 { "q4_0" } else { "q8_0" },
                );
            }
            let unpadded_bytes = (elem_count / block_size) * type_size;
            let padded_bytes =
                unpadded_bytes + MATRIX_ROW_PADDING * type_size / block_size;
            let mut inner = if self.data.inner.len() == padded_bytes {
                std::mem::replace(
                    &mut self.data.inner,
                    self.device.alloc_zeros::<u8>(0)?,
                )
            } else {
                self.device.alloc_zeros::<u8>(padded_bytes)?
            };

            match (&src.slice, is_q4) {
                (crate::cuda_backend::CudaStorageSlice::F32(data), false) => {
                    let view = data.slice(offset..offset + elem_count);
                    quantize_q8_0_f32(&view, &mut inner, elem_count, 1, &self.device)?;
                }
                (crate::cuda_backend::CudaStorageSlice::F16(data), false) => {
                    let view = data.slice(offset..offset + elem_count);
                    quantize_q8_0_f16(&view, &mut inner, elem_count, 1, &self.device)?;
                }
                (crate::cuda_backend::CudaStorageSlice::F32(data), true) => {
                    let view = data.slice(offset..offset + elem_count);
                    quantize_q4_0_f32(&view, &mut inner, elem_count, 1, &self.device)?;
                }
                (crate::cuda_backend::CudaStorageSlice::F16(data), true) => {
                    let view = data.slice(offset..offset + elem_count);
                    quantize_q4_0_f16(&view, &mut inner, elem_count, 1, &self.device)?;
                }
                _ => crate::bail!(
                    "quantize_{}: unsupported source dtype (expected f32 or f16)",
                    if is_q4 { "q4_0" } else { "q8_0" },
                ),
            }

            self.data = PaddedCudaSlice {
                inner,
                len: unpadded_bytes,
            };
            return Ok(());
        }

        // Other GGML dtypes fall through to the legacy CPU-roundtrip path.
        // Slight quirk: the CPU path below reads `data.len()` and therefore
        // ignores the layout. For the KV-cache use case we only care about
        // Q8_0 / Q4_0 on the fast path above, so this is fine for now.
        self.quantize(src)
    }

    pub fn quantize(&mut self, src: &CudaStorage) -> Result<()> {
        // GPU-native fast path: Q8_0 can be produced directly on-device by a
        // small CUDA kernel, skipping the dtoh→quantize→htod roundtrip that
        // would otherwise stall every write. Other formats still fall back
        // to the CPU path below.
        if self.dtype == GgmlDType::Q8_0 {
            let elem_count = match &src.slice {
                crate::cuda_backend::CudaStorageSlice::F32(data) => data.len(),
                crate::cuda_backend::CudaStorageSlice::F16(data) => data.len(),
                _ => crate::bail!("quantize_q8_0: unsupported source dtype (expected f32 or f16)"),
            };
            let block_size = self.dtype.block_size();
            let type_size = self.dtype.type_size();
            if !elem_count.is_multiple_of(block_size) {
                crate::bail!(
                    "quantize_q8_0: element count {elem_count} is not divisible by block size {block_size}",
                );
            }
            // PaddedCudaSlice::len holds the unpadded byte count (what dequant expects).
            // The backing `inner` is oversized by one padding row per quantization
            // block so trailing partial blocks don't read OOB.
            let unpadded_bytes = (elem_count / block_size) * type_size;
            let padded_bytes =
                unpadded_bytes + MATRIX_ROW_PADDING * type_size / block_size;
            // Reuse the existing allocation if it is exactly the right size:
            // this is the per-step fast path (e.g., a KV cache re-quantizing
            // one new token every decode step) where a fresh cudaMalloc
            // would cost ~tens of µs and scales with device memory pressure.
            // Exact-size match keeps the padding region's zero-init intact
            // without needing a device-side memset.
            let mut inner = if self.data.inner.len() == padded_bytes {
                std::mem::replace(
                    &mut self.data.inner,
                    self.device.alloc_zeros::<u8>(0)?,
                )
            } else {
                self.device.alloc_zeros::<u8>(padded_bytes)?
            };

            // Treat the entire tensor as one "row" of `elem_count` elements.
            // The kernel pads the row up to kx_padded = ceil(k, 512) internally.
            match &src.slice {
                crate::cuda_backend::CudaStorageSlice::F32(data) => {
                    quantize_q8_0_f32(&data.slice(..), &mut inner, elem_count, 1, &self.device)?;
                }
                crate::cuda_backend::CudaStorageSlice::F16(data) => {
                    quantize_q8_0_f16(&data.slice(..), &mut inner, elem_count, 1, &self.device)?;
                }
                _ => unreachable!(),
            }

            self.data = PaddedCudaSlice {
                inner,
                len: unpadded_bytes,
            };
            return Ok(());
        }

        // Fallback: run the quantization on cpu.
        let src = match &src.slice {
            crate::cuda_backend::CudaStorageSlice::F32(data) => self.device.clone_dtoh(data)?,
            _ => crate::bail!("only f32 can be quantized"),
        };
        let src_len = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;
        qcpu_storage.quantize(&src)?;
        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .memcpy_htod(&*data, &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedCudaSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn quantize_imatrix(
        &mut self,
        src: &CudaStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Run the quantization on cpu.
        let src = match &src.slice {
            crate::cuda_backend::CudaStorageSlice::F32(data) => self.device.clone_dtoh(data)?,
            _ => crate::bail!("only f32 can be quantized"),
        };
        let src_len = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;
        qcpu_storage.quantize_imatrix(&src, imatrix_weights, n_per_row)?;
        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .memcpy_htod(&*data, &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedCudaSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        src: &crate::CpuStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Run the quantization on cpu.
        let src_len = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row);
        } else {
            unreachable!()
        }

        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .memcpy_htod(&*data, &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedCudaSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn quantize_onto(&mut self, src: &crate::CpuStorage) -> Result<()> {
        // Run the quantization on cpu.
        let src_len = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float(src.as_slice::<f32>()?);
        } else {
            unreachable!()
        }

        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .memcpy_htod(&*data, &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedCudaSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.data.len
    }

    pub fn fwd(
        &self,
        self_shape: &crate::Shape,
        storage: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        let max_bm = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            1
        } else {
            8
        };
        let use_vec_kernel = match layout.shape().dims() {
            [b, m, _k] => b * m <= max_bm,
            [b, _k] => *b <= max_bm,
            _ => false,
        };
        if use_vec_kernel {
            self.dequantize_matmul_vec(self_shape, storage, layout)
        } else {
            self.dequantize_matmul(self_shape, storage, layout)
        }
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        let mut out = vec![0u8; self.data.len];
        self.device
            .memcpy_dtoh(&self.data.inner.slice(..self.data.len), &mut out)?;
        Ok(out)
    }

    /// Device-side view of the unpadded Q8 bytes. Used by callers that
    /// want to memcpy the freshly-quantized data into a larger persistent
    /// buffer (e.g., a Q8 KV cache) without round-tripping through host.
    pub fn data_slice_for_copy(
        &self,
        byte_len: usize,
    ) -> Result<cudarc::driver::CudaView<u8>> {
        if byte_len > self.data.len {
            crate::bail!(
                "data_slice_for_copy: byte_len {byte_len} exceeds unpadded length {}",
                self.data.len
            );
        }
        Ok(self.data.inner.slice(..byte_len))
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        use cudarc::driver::DevicePtr;
        Ok(self.data.inner.device_ptr(self.data.inner.stream()).0 as *const u8)
    }
}

impl QCudaStorage {
    fn dequantize_matmul_vec(
        &self,
        self_shape: &crate::Shape,
        rhs: &CudaStorage,
        rhs_l: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        let (nrows, ncols) = self_shape.dims2()?;
        let rhs = rhs.as_cuda_slice::<f32>()?;
        let rhs = match rhs_l.contiguous_offsets() {
            Some((o1, o2)) => rhs.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "dmmv" }.bt())?,
        };
        let (b_size, k) = match rhs_l.shape().dims() {
            [b, m, k] => (b * m, *k),
            [b, k] => (*b, *k),
            _ => crate::bail!("unexpected rhs shape in dmmv {:?}", rhs_l.shape()),
        };
        if ncols != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", rhs_l.shape())
        }

        let out = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            dequantize_mul_mat_vec(&self.data, &rhs, self.dtype, ncols, nrows, self.device())?
        } else {
            mul_mat_vec_via_q8_1(
                &self.data,
                &rhs,
                self.dtype,
                ncols,
                nrows,
                b_size,
                self.device(),
            )?
        };
        let mut out_shape = rhs_l.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(nrows);
        Ok((out, out_shape.into()))
    }

    fn dequantize_matmul(
        &self,
        self_shape: &crate::Shape,
        storage: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        use crate::backend::BackendStorage;
        let (n, k) = self_shape.dims2()?;
        let (b, m, k2) = match layout.shape().dims() {
            &[b, m, k2] => (b, m, k2),
            &[m, k2] => (1, m, k2),
            s => crate::bail!("unexpected shape for input {s:?}"),
        };
        if k2 != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", layout.shape())
        }

        let out = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            let data_f32 = self.dequantize(n * k)?;
            let rhs_l = crate::Layout::new((k, n).into(), vec![1, k], 0).broadcast_as((b, k, n))?;
            storage.matmul(&data_f32, (b, m, n, k), layout, &rhs_l)?
        } else {
            let storage = storage.as_cuda_slice::<f32>()?;
            let storage = match layout.contiguous_offsets() {
                Some((o1, o2)) => storage.slice(o1..o2),
                None => Err(crate::Error::RequiresContiguous {
                    op: "quantized-matmul",
                }
                .bt())?,
            };
            mul_mat_via_q8_1(
                &self.data,
                &storage,
                self.dtype,
                /* x_rows */ n,
                /* x_cols */ k,
                /* y_rows */ k,
                /* y_cols */ b * m,
                self.device(),
            )?
        };
        let mut out_shape = layout.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(n);
        Ok((out, out_shape.into()))
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &CudaDevice,
    data: &[T],
) -> Result<super::QStorage> {
    let data = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, core::mem::size_of_val(data))
    };
    let dtype = T::DTYPE;
    let padded_len = data.len() + MATRIX_ROW_PADDING * dtype.type_size() / dtype.block_size();
    // Use alloc_zeros to ensure padding bytes are initialized to zero.
    let mut inner = device.alloc_zeros::<u8>(padded_len)?;
    device.memcpy_htod(data, &mut inner.slice_mut(..data.len()))?;
    Ok(QStorage::Cuda(QCudaStorage {
        data: PaddedCudaSlice {
            inner,
            len: data.len(),
        },
        device: device.clone(),
        dtype,
    }))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn cuda_quantize_q8_1() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let el = 256;
        let el_padded = pad(el, MATRIX_ROW_PADDING);
        let y_size_in_bytes =
            el_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
        let mut y_q8_1 = unsafe { dev.alloc::<u8>(y_size_in_bytes)? };
        let vs: Vec<f32> = (0..el).map(|v| v as f32).collect();
        let y = dev.clone_htod(&vs)?;
        quantize_q8_1(&y.as_view(), &mut y_q8_1, el, 1, &dev)?;
        Ok(())
    }

    #[test]
    fn cuda_mmv_q8_1() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let ncols = 256;
        let vs: Vec<f32> = (0..ncols).map(|v| v as f32).collect();
        let y = dev.clone_htod(&vs)?;
        let mut xs = QCudaStorage::zeros(&dev, ncols, GgmlDType::Q4_0)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
        let cuda_storage = mul_mat_vec_via_q8_1(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* ncols */ ncols,
            /* nrows */ 1,
            /* b_size */ 1,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let vs = dev.clone_dtoh(&vs.as_view())?;
        assert_eq!(vs.len(), 1);
        // for n = 255, n.(n+1).(2n+1) / 6 = 5559680
        // Q8 means 1/256 precision.
        assert_eq!(vs[0], 5561664.5);

        let cuda_storage = dequantize_mul_mat_vec(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* ncols */ ncols,
            /* nrows */ 1,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let vs = dev.clone_dtoh(&vs.as_view())?;
        assert_eq!(vs.len(), 1);
        assert_eq!(vs[0], 5561851.0);
        Ok(())
    }

    #[test]
    fn cuda_mm_q8_1() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let ncols = 256;
        let vs: Vec<f32> = (0..ncols * 4).map(|v| v as f32 / 4.).collect();
        let y = dev.clone_htod(&vs)?;
        let mut xs = QCudaStorage::zeros(&dev, ncols * 4, GgmlDType::Q4_0)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
        let cuda_storage = mul_mat_via_q8_1(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* x_rows */ 4,
            /* x_cols */ ncols,
            /* y_rows */ ncols,
            /* y_cols */ 4,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let vs = dev.clone_dtoh(&vs.as_view())?;

        /*
           x = torch.tensor([float(v) for v in range(1024)]).reshape(4, 256)
           x @ x.t() / 16
        tensor([[  347480.0000,   869720.0000,  1391960.0000,  1914200.0000],
                [  869720.0000,  2440536.0000,  4011352.0000,  5582166.5000],
                [ 1391960.0000,  4011352.0000,  6630742.0000,  9250132.0000],
                [ 1914200.0000,  5582166.5000,  9250132.0000, 12918099.0000]])
                */
        assert_eq!(vs.len(), 16);
        assert_eq!(vs[0], 347604.0);
        assert_eq!(vs[1], 888153.06);
        assert_eq!(vs[4], 869780.7);
        assert_eq!(vs[5], 2483145.0);
        assert_eq!(vs[11], 9407368.0);
        assert_eq!(vs[14], 9470856.0);
        assert_eq!(vs[15], 13138824.0);
        Ok(())
    }

    // The following test used to fail under compute-sanitizer until #2526.
    #[test]
    fn cuda_mm_q8_1_pad() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let (x_rows, ncols, y_cols) = (4, 16, 2048);
        let vs: Vec<f32> = (0..ncols * y_cols).map(|v| v as f32 / 256.).collect();
        let y = dev.clone_htod(&vs)?;
        let mut xs = QCudaStorage::zeros(&dev, ncols * x_rows, GgmlDType::Q4_0)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
        let cuda_storage = mul_mat_via_q8_1(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* x_rows */ x_rows,
            /* x_cols */ ncols,
            /* y_rows */ ncols,
            /* y_cols */ y_cols,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let _vs = dev.clone_dtoh(&vs.as_view())?;
        Ok(())
    }
}
