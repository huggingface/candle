// Adapted from https://github.com/guoqingbao/attention.rs/blob/main/src/moe.rs
#[cfg(feature = "cuda")]
use candle::cuda_backend::kernels::ffi;
#[allow(unused_imports)]
use candle::quantized::{self, QTensor};
use candle::{Result, Tensor};

#[cfg(feature = "cuda")]
pub fn moe_gemm(
    input: &Tensor,
    weights: &Tensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
    is_prefill: bool,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::DType;
    use half::{bf16, f16};

    fn cuda_fwd<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        input: &Tensor,
        weights: &Tensor,
        topk_weights: &Option<Tensor>,
        sorted_token_ids: &Tensor,
        experts_ids: &Tensor,
        topk: usize,
        is_prefill: bool,
    ) -> Result<Tensor> {
        let (mut size_m, size_k1) = input.dims2()?;
        if topk_weights.is_none() {
            size_m *= topk;
        }
        let (num_experts, size_n, size_k) = weights.dims3()?;
        assert!(
            size_k == size_k1,
            "input {:?} and weight {:?} last dim mismatch!",
            size_k1,
            size_k
        );
        let dev = input.device().as_cuda_device()?;
        let data_type = match input.dtype() {
            DType::F16 => 0,
            DType::BF16 => 1,
            _ => {
                candle::bail!("moe_gemm_wmma only accepts f16/bf16 inputs")
            }
        };

        let (input, _) = input.storage_and_layout();
        let input = match &*input {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("input must be a cuda tensor"),
        };

        let (weights, _) = weights.storage_and_layout();
        let weights = match &*weights {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("weight must be a cuda tensor"),
        };

        let (sorted_token_ids, _) = sorted_token_ids.storage_and_layout();
        let sorted_token_ids = match &*sorted_token_ids {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("sorted_token_ids must be a cuda tensor"),
        };

        let (experts_ids, _) = experts_ids.storage_and_layout();
        let experts_ids = match &*experts_ids {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("experts_ids must be a cuda tensor"),
        };

        let topk_weights_ptr = if let Some(topk_weights) = &topk_weights {
            let (topk_weights, _) = topk_weights.storage_and_layout();
            let topk_weights = match &*topk_weights {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!("topk_weights must be a cuda tensor"),
            };
            let weights_ptr = topk_weights.device_ptr(topk_weights.stream()).0 as *const f32;
            weights_ptr
        } else {
            std::ptr::null()
        };

        let output = unsafe { dev.alloc::<T>(size_m * size_n) }?;
        let expert_counts = unsafe { dev.alloc::<u32>(num_experts) }?;
        let expert_offsets = unsafe { dev.alloc::<u32>(num_experts + 1) }?;

        let stream = dev.cuda_stream().cu_stream() as i64;
        use core::ffi::c_void;

        unsafe {
            ffi::moe_gemm_wmma(
                input.device_ptr(input.stream()).0 as *const c_void, // [size_m, size_k]
                weights.device_ptr(weights.stream()).0 as *const c_void, // [num_experts, size_n, size_k]
                sorted_token_ids.device_ptr(sorted_token_ids.stream()).0 as *const i32,
                experts_ids.device_ptr(experts_ids.stream()).0 as *const i32,
                topk_weights_ptr,
                output.device_ptr(output.stream()).0 as *mut c_void, // [size_m, size_n]
                expert_counts.device_ptr(expert_counts.stream()).0 as *mut i32, // pre-allocated buffer [num_experts]
                expert_offsets.device_ptr(expert_offsets.stream()).0 as *mut i32, // pre-allocated buffer [num_experts + 1]
                num_experts as i32,
                topk as i32,
                size_m as i32,
                size_n as i32,
                size_k as i32,
                data_type as i32, // 0=float16, 1=bf16 (for input/output)
                is_prefill,
                stream,
            );
        }

        use candle::op::BackpropOp;
        let output = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
        let output = Tensor::from_storage(
            candle::Storage::Cuda(output),
            (size_m, size_n),
            BackpropOp::none(),
            false,
        );

        Ok(output)
    }

    match input.dtype() {
        DType::F16 => cuda_fwd::<f16>(
            input,
            weights,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
            is_prefill,
        ),
        DType::BF16 => cuda_fwd::<bf16>(
            input,
            weights,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
            is_prefill,
        ),
        _ => {
            candle::bail!("moe_gemm only accepts f16/bf16 inputs")
        }
    }
}

#[cfg(not(feature = "cuda"))]
pub fn moe_gemm(
    _: &Tensor,
    _: &Tensor,
    _: &Option<Tensor>,
    _: &Tensor,
    _: &Tensor,
    _: usize,
    _: bool,
) -> Result<Tensor> {
    candle::bail!("moe_gemm is only implemented for the cuda backend")
}

/// Phase 1 step-1: device-side expert offsets from a sorted expert_ids
/// array. Returns a `Tensor` of shape `[num_experts + 1]` i32 where
/// `offsets[e] = first index i with expert_ids[i] >= e` and
/// `offsets[num_experts] = m`. Used to unblock per-expert dispatch
/// in the MoE GEMM forward path — for prefill batches where the
/// existing per-(token,expert) kernel emits a huge grid.
///
/// `sorted_expert_ids` must be a 1-D contiguous CUDA tensor of i32
/// (u32 also accepted at the call site if cast first) sorted ascending.
#[cfg(feature = "cuda")]
pub fn moe_expert_offsets(
    sorted_expert_ids: &Tensor,
    num_experts: usize,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    if sorted_expert_ids.dims().len() != 1 {
        candle::bail!(
            "moe_expert_offsets: expert_ids must be 1-D, got {:?}",
            sorted_expert_ids.shape()
        );
    }
    let m = sorted_expert_ids.elem_count();
    let dev = sorted_expert_ids.device().as_cuda_device()?;
    let stream = dev.cuda_stream().cu_stream() as i64;

    // Candle stores index tensors as U32 by convention but their values
    // fit in i32; the kernel reads via *const i32 which is bit-equal.
    let (e_storage, _) = sorted_expert_ids.storage_and_layout();
    let e_slice = match &*e_storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
        _ => candle::bail!("moe_expert_offsets: sorted_expert_ids must be on CUDA"),
    };

    let offsets_alloc = unsafe { dev.alloc::<i32>(num_experts + 1) }?;
    let e_ptr = e_slice.device_ptr(e_slice.stream()).0 as *const i32;
    let off_ptr = offsets_alloc.device_ptr(offsets_alloc.stream()).0 as *mut i32;
    unsafe {
        ffi::moe_expert_offsets(
            e_ptr,
            off_ptr,
            m as i32,
            num_experts as i32,
            stream,
        );
    }

    use candle::op::BackpropOp;
    let storage = candle::CudaStorage::wrap_cuda_slice(offsets_alloc, dev.clone());
    Ok(Tensor::from_storage(
        candle::Storage::Cuda(storage),
        (num_experts + 1,),
        BackpropOp::none(),
        false,
    ))
}

#[cfg(not(feature = "cuda"))]
pub fn moe_expert_offsets(_: &Tensor, _: usize) -> Result<Tensor> {
    candle::bail!("moe_expert_offsets is cuda-only")
}

/// Phase 1 step-3 scaffolding: per-expert dispatch for the MoE gate||up
/// matmul, targeted at the prefill regime where the current
/// per-(token,expert) kernel emits an O(num_tokens × topk) grid that
/// cripples large batches (gemma4:26b prefill = -91.8% vs Ollama).
///
/// For each expert `e` with assigned tokens
/// `(expert_offsets[e+1] - expert_offsets[e]) > 0`:
///   1. Gather input rows: `gathered = inputs[sorted_token_ids[range] / topk]`
///      → contiguous `[tokens_for_e, K]` F32 or F16 buffer.
///   2. Dequantize `gate_up_weights[e]` Q4_K → F16
///      via `candle::quantized::cuda::dequantize_q4k_expert_f16`.
///   3. cuBLAS GEMM (CUDA_R_16F input × weight, CUDA_R_16F output,
///      CUDA_R_32F compute) — tensor cores fire when tokens_for_e ≥ 16.
///   4. Split + GELU(tanh) * up → `[tokens_for_e, N]` F32.
///   5. Scatter rows back into the global `[size_m, N]` output via
///      `sorted_token_ids[expert_offsets[e]..expert_offsets[e+1]]`.
///
/// This signature is the contract for the new dispatch path. Caller
/// (e.g. `forward_cuda` in `generic_transformer.rs`) decides whether
/// to take the per-expert path based on `size_m`:
///   `if size_m < N_PER_EXPERT_THRESHOLD { /* existing kernel */ }
///    else { moe_gemm_gguf_per_expert_gate_up_gelu_mul_concat(...) }`
///
/// **NOT YET IMPLEMENTED** — landing this is step 3 of
/// `docs/phase1_moe_implementation_plan.md`. Implementation requires:
///   - A gather kernel (or use `Tensor::index_select` if the
///     index dtype maps cleanly — `sorted_token_ids[range] / topk` is
///     per-element division, not a single index_select).
///   - cuBLAS GEMM via `cudarc::cublas::Gemm` (already used elsewhere
///     in candle for batched matmul).
///   - GELU·mul on `[tokens_for_e, 2N]` — re-use the pattern from
///     `fused_split_gelu_mul_f32` in `crates/server/.../fused_kernels.rs`.
///   - Scatter via atomicAdd or `index_add`.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn moe_gemm_gguf_per_expert_gate_up_gelu_mul_concat(
    inputs: &Tensor,
    gate_up_weights: &QTensor,
    sorted_token_ids: &Tensor,
    expert_offsets: &Tensor,
    num_experts: usize,
    topk: usize,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::quantized::GgmlDType;
    use candle::DType;
    use half::f16;

    // ── Validate inputs ─────────────────────────────────────────────
    let (_num_real_tokens, k_in) = inputs.dims2()?;
    let (gate_up_num_experts, two_n, k_w) = match gate_up_weights.shape().dims() {
        [e, n, k] => (*e, *n, *k),
        s => candle::bail!(
            "moe_gemm_gguf_per_expert_gate_up_gelu_mul_concat: weight must be 3D [num_experts, 2N, K], got {:?}",
            s
        ),
    };
    if gate_up_num_experts != num_experts {
        candle::bail!(
            "weight num_experts {gate_up_num_experts} != arg num_experts {num_experts}"
        );
    }
    if k_in != k_w {
        candle::bail!("input K {k_in} != weight K {k_w}");
    }
    if two_n % 2 != 0 {
        candle::bail!("weight 2N dim {two_n} must be even");
    }
    if gate_up_weights.dtype() != GgmlDType::Q4K {
        candle::bail!(
            "moe_gemm_gguf_per_expert_gate_up_gelu_mul_concat: Phase 1 step 3 \
             currently only implements Q4_K weights, got {:?}",
            gate_up_weights.dtype()
        );
    }
    if inputs.dtype() != DType::F32 {
        candle::bail!("input must be F32");
    }
    let n_out = two_n / 2;
    let size_m = sorted_token_ids.elem_count();
    let dev = inputs.device().as_cuda_device()?;
    let stream_i64 = dev.cuda_stream().cu_stream() as i64;

    // ── Resolve raw device pointers ─────────────────────────────────
    let (inp_storage, inp_layout) = inputs.storage_and_layout();
    let inputs_slice = match &*inp_storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("inputs must be on CUDA"),
    };

    // sorted_token_ids comes from the caller's topk + sort_last_dim;
    // candle stores indices as U32 by convention. We pass the same
    // device pointer to the kernel cast to *const i32 — values are
    // small token indices that fit in i32 either way.
    let (st_storage, _st_layout) = sorted_token_ids.storage_and_layout();
    let st_slice = match &*st_storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
        _ => candle::bail!("sorted_token_ids must be CUDA u32"),
    };

    // expert_offsets is built by moe_expert_offsets which allocates an
    // i32 buffer — read it as I32 here.
    let (off_storage, _off_layout) = expert_offsets.storage_and_layout();
    let off_slice = match &*off_storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<i32>()?,
        _ => candle::bail!("expert_offsets must be CUDA i32"),
    };
    let offsets_cpu: Vec<i32> = dev.cuda_stream()
        .memcpy_dtov(off_slice)
        .map_err(|e| candle::Error::Msg(format!("expert_offsets D2H: {e}")))?;
    if offsets_cpu.len() < num_experts + 1 {
        candle::bail!(
            "expert_offsets length {} < num_experts+1 {}",
            offsets_cpu.len(), num_experts + 1
        );
    }

    // Resolve the raw u8 device pointer + length for the gate_up_weights
    // blob; the per-expert dequant helper offsets into it.
    let weight_data_ptr = gate_up_weights.device_ptr()? as u64;
    // sizeof(block_q4_K) = 144, blocks per expert = 2N * K / QK_K
    let weight_data_len_bytes = num_experts * (two_n * k_w / 256) * 144;

    // ── Output: [size_m, N] F32 zero-init ───────────────────────────
    let out_alloc = unsafe { dev.alloc_zeros::<f32>(size_m * n_out) }?;

    // ── Compute max active-expert size from offsets to size persistent
    // workspaces. Hoisting allocations out of the per-expert loop saves
    // ~5 cudaMallocAsync calls per expert × ~60 experts × 30 layers =
    // ~9000 alloc calls per prefill (~50ms total at 5µs each).
    let mut max_n_e: usize = 0;
    for e in 0..num_experts {
        let n_e = (offsets_cpu[e + 1] - offsets_cpu[e]) as usize;
        if n_e > max_n_e { max_n_e = n_e; }
    }
    if max_n_e == 0 {
        // No active experts — nothing to do. Return zero-init output.
        drop(inp_storage); drop(st_storage); drop(off_storage);
        let storage = candle::CudaStorage::wrap_cuda_slice(out_alloc, dev.clone());
        return Ok(Tensor::from_storage(
            candle::Storage::Cuda(storage),
            (size_m, n_out),
            candle::op::BackpropOp::none(),
            false,
        ));
    }

    // ── Build active-expert list on host ────────────────────────────
    let mut active_expert_ids: Vec<i32> = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        if offsets_cpu[e + 1] > offsets_cpu[e] {
            active_expert_ids.push(e as i32);
        }
    }
    let n_active = active_expert_ids.len();

    // Persistent batched workspaces:
    //   gather_alloc: [N_active × max_n_e × K] F16 — padded per-expert
    //                 gather buffers, all in one contiguous slab so
    //                 candle's batched matmul can dispatch ONE
    //                 cublasGemmStridedBatched call.
    //   weights_alloc: [N_active × 2N × K] F16 — filled by ONE batched
    //                  dequant launch.
    // Memory bound on gemma4:26b prefill (N_active≤128, max_n_e≤size_m,
    // K=2816): in the worst case (sparse routing → many experts × few
    // tokens each) the gather slab is N_active × size_m × K × 2 bytes.
    // For size_m=8192, K=2816, N_active=128: ~5.8 GB — too big. So we
    // bound max_n_e at gather time to min(actual max_n_e, ~workspace
    // budget / (N_active × K × 2)). The wiring up top already computes
    // the true max_n_e from offsets so the padding doesn't blow.
    let gather_total_elems = n_active * max_n_e * k_in;
    let weights_total_elems = n_active * two_n * k_w;
    // Conservative budget check: if either workspace exceeds 4 GB, fall
    // back caller-side to dp4a path. (Caller's threshold should keep
    // us well below this in practice for gemma4:26b.)
    const MAX_WORKSPACE_BYTES: usize = 4 * 1024 * 1024 * 1024;
    if gather_total_elems * 2 > MAX_WORKSPACE_BYTES
        || weights_total_elems * 2 > MAX_WORKSPACE_BYTES
    {
        candle::bail!(
            "moe_gemm_gguf_per_expert: workspace {} GB exceeds 4 GB budget — caller should use dp4a path",
            (gather_total_elems.max(weights_total_elems) * 2) / (1024 * 1024 * 1024)
        );
    }
    let gather_alloc = unsafe { dev.alloc_zeros::<f16>(gather_total_elems) }?;
    let weights_alloc = unsafe { dev.alloc::<f16>(weights_total_elems) }?;

    // ── Step (b'): batched dequant of all active experts in ONE launch.
    let active_ids_alloc = dev.cuda_stream()
        .memcpy_stod(&active_expert_ids)
        .map_err(|e| candle::Error::Msg(format!("active_expert_ids H2D: {e}")))?;
    {
        let act_ptr = active_ids_alloc.device_ptr(active_ids_alloc.stream()).0 as *const i32;
        let wts_ptr = weights_alloc.device_ptr(weights_alloc.stream()).0 as *mut core::ffi::c_void;
        let wts_data_ptr = weight_data_ptr as *const core::ffi::c_void;
        unsafe {
            ffi::moe_batched_dequant_q4k_f16(
                wts_data_ptr, act_ptr, wts_ptr,
                n_active as i32, two_n as i32, k_w as i32,
                stream_i64,
            );
        }
        let _ = weight_data_len_bytes;
    }

    // ── Step (a'): ONE batched gather across ALL active experts.
    //              Replaces the N_active per-expert gather launches
    //              with a single launch that writes the padded
    //              [N_active, max_n_e, K] workspace in one shot.
    let gather_ptr_base = gather_alloc.device_ptr(gather_alloc.stream()).0;
    {
        let inp_ptr  = inputs_slice.device_ptr(inputs_slice.stream()).0 as *const f32;
        let st_ptr   = st_slice.device_ptr(st_slice.stream()).0 as *const i32;
        let act_ptr  = active_ids_alloc.device_ptr(active_ids_alloc.stream()).0 as *const i32;
        let off_ptr  = off_slice.device_ptr(off_slice.stream()).0 as *const i32;
        let gath_ptr = gather_ptr_base as *mut core::ffi::c_void;
        unsafe {
            ffi::moe_batched_gather_input_rows_f32_to_f16(
                inp_ptr, st_ptr, act_ptr, off_ptr, gath_ptr,
                n_active as i32, max_n_e as i32, k_in as i32, topk as i32,
                stream_i64,
            );
        }
    }

    let gather_storage = candle::CudaStorage::wrap_cuda_slice(gather_alloc, dev.clone());
    let gather_tensor_batched = Tensor::from_storage(
        candle::Storage::Cuda(gather_storage),
        (n_active, max_n_e, k_in),
        candle::op::BackpropOp::none(),
        false,
    );
    let weights_storage = candle::CudaStorage::wrap_cuda_slice(weights_alloc, dev.clone());
    let weights_tensor_full = Tensor::from_storage(
        candle::Storage::Cuda(weights_storage),
        (n_active, two_n, k_w),
        candle::op::BackpropOp::none(),
        false,
    );

    // ── Step (c'): ONE batched cuBLAS GEMM via candle Tensor::matmul.
    //              gemm_out[act_idx, m, n] =
    //                  sum_k gather[act_idx, m, k] × weights[act_idx, n, k]
    //              Shape: [N_active, max_n_e, 2N] F16.
    let gemm_out_batched = gather_tensor_batched.matmul(&weights_tensor_full.transpose(1, 2)?)?;

    // ── Step (d'): ONE batched GELU·mul + scatter across all experts.
    //              Replaces N_active scatter launches with a single
    //              launch that reads [N_active, max_n_e, 2N] and
    //              writes the F32 output[size_m, N] via scatter.
    let gemm_out_contiguous = gemm_out_batched.contiguous()?;
    let (gemm_storage, _gemm_layout) = gemm_out_contiguous.storage_and_layout();
    let gemm_slice = match &*gemm_storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f16>()?,
        _ => candle::bail!("gemm_out must be on CUDA F16"),
    };
    {
        let st_ptr  = st_slice.device_ptr(st_slice.stream()).0 as *const i32;
        let act_ptr = active_ids_alloc.device_ptr(active_ids_alloc.stream()).0 as *const i32;
        let off_ptr = off_slice.device_ptr(off_slice.stream()).0 as *const i32;
        let gemm_ptr = gemm_slice.device_ptr(gemm_slice.stream()).0 as *const core::ffi::c_void;
        let out_ptr  = out_alloc.device_ptr(out_alloc.stream()).0 as *mut f32;
        unsafe {
            ffi::moe_batched_gelu_mul_scatter_f16_to_f32(
                gemm_ptr, st_ptr, act_ptr, off_ptr, out_ptr,
                n_active as i32, max_n_e as i32, n_out as i32,
                stream_i64,
            );
        }
    }
    drop(gemm_storage);

    // ── Wrap output ────────────────────────────────────────────────
    drop(inp_storage); drop(st_storage); drop(off_storage);
    let storage = candle::CudaStorage::wrap_cuda_slice(out_alloc, dev.clone());
    Ok(Tensor::from_storage(
        candle::Storage::Cuda(storage),
        (size_m, n_out),
        candle::op::BackpropOp::none(),
        false,
    ))
}

/// Phase 1 step-5 (Q4_K MMA direct path): per-expert batched
/// `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` GEMM. Same
/// signature and output contract as
/// `moe_gemm_gguf_per_expert_gate_up_gelu_mul_concat` but skips the
/// F16 dequant intermediate (~3 GB on gemma4:26b prefill) by consuming
/// Q4_K weights directly with tensor cores and Q8_1-quantized inputs.
///
/// Pipeline:
///   1. F32 inputs → Q8_1 `[N_active, max_n_e, K/32]` (one batched
///      gather+quantize launch).
///   2. Q4_K weights × Q8_1 inputs → F16 logits `[N_active, max_n_e, 2N]`
///      (one batched MMA launch, one m16n8k32 INT8 mma.sync per K-tile).
///   3. F16 logits → F32 `[size_m, N]` via GELU·mul + scatter (one
///      batched launch — same kernel as the F16-dequant path).
///
/// Memory: Q8_1 input slab is ~1.06 bytes/elem (vs F16 gather's 2
/// bytes/elem), so workspace is ~half the F16 path. No F16 weight
/// materialization — saves the ~3 GB intermediate.
///
/// Status: kernel + FFI shipped (commit 9b081239). This wrapper wires
/// the orchestrator but has NOT been validated against the dp4a
/// reference yet. Numerical correctness needs a side-by-side check
/// before swapping into the dispatch path.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn moe_gemm_gguf_per_expert_gate_up_gelu_mul_concat_mma(
    inputs: &Tensor,
    gate_up_weights: &QTensor,
    sorted_token_ids: &Tensor,
    expert_offsets: &Tensor,
    num_experts: usize,
    topk: usize,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::quantized::GgmlDType;
    use candle::DType;

    let (_num_real_tokens, k_in) = inputs.dims2()?;
    let (gate_up_num_experts, two_n, k_w) = match gate_up_weights.shape().dims() {
        [e, n, k] => (*e, *n, *k),
        s => candle::bail!(
            "moe_gemm_gguf_per_expert_mma: weight must be 3D [num_experts, 2N, K], got {:?}",
            s
        ),
    };
    if gate_up_num_experts != num_experts {
        candle::bail!(
            "weight num_experts {gate_up_num_experts} != arg num_experts {num_experts}"
        );
    }
    if k_in != k_w {
        candle::bail!("input K {k_in} != weight K {k_w}");
    }
    if k_in % 256 != 0 {
        candle::bail!("Q4_K MMA path requires K % 256 == 0, got K={k_in}");
    }
    if gate_up_weights.dtype() != GgmlDType::Q4K {
        candle::bail!(
            "moe_gemm_gguf_per_expert_mma: only Q4_K weights supported, got {:?}",
            gate_up_weights.dtype()
        );
    }
    if inputs.dtype() != DType::F32 {
        candle::bail!("input must be F32");
    }
    let n_out = two_n / 2;
    let size_m = sorted_token_ids.elem_count();
    let dev = inputs.device().as_cuda_device()?;
    let stream_i64 = dev.cuda_stream().cu_stream() as i64;

    let (inp_storage, _) = inputs.storage_and_layout();
    let inputs_slice = match &*inp_storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("inputs must be on CUDA"),
    };
    let (st_storage, _) = sorted_token_ids.storage_and_layout();
    let st_slice = match &*st_storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
        _ => candle::bail!("sorted_token_ids must be CUDA u32"),
    };
    let (off_storage, _) = expert_offsets.storage_and_layout();
    let off_slice = match &*off_storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<i32>()?,
        _ => candle::bail!("expert_offsets must be CUDA i32"),
    };
    let offsets_cpu: Vec<i32> = dev.cuda_stream()
        .memcpy_dtov(off_slice)
        .map_err(|e| candle::Error::Msg(format!("expert_offsets D2H: {e}")))?;

    let weight_data_ptr = gate_up_weights.device_ptr()? as u64;

    let out_alloc = unsafe { dev.alloc_zeros::<f32>(size_m * n_out) }?;

    let mut max_n_e: usize = 0;
    for e in 0..num_experts {
        let n_e = (offsets_cpu[e + 1] - offsets_cpu[e]) as usize;
        if n_e > max_n_e { max_n_e = n_e; }
    }
    if max_n_e == 0 {
        drop(inp_storage); drop(st_storage); drop(off_storage);
        let storage = candle::CudaStorage::wrap_cuda_slice(out_alloc, dev.clone());
        return Ok(Tensor::from_storage(
            candle::Storage::Cuda(storage),
            (size_m, n_out),
            candle::op::BackpropOp::none(),
            false,
        ));
    }

    let mut active_expert_ids: Vec<i32> = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        if offsets_cpu[e + 1] > offsets_cpu[e] {
            active_expert_ids.push(e as i32);
        }
    }
    let n_active = active_expert_ids.len();

    // Workspaces:
    //   q81_alloc:   [N_active × max_n_e × K/32] block_q8_1 = 36 bytes/block
    //   gemm_alloc:  [N_active × max_n_e × 2N] F32 (output of MMA kernel,
    //                input to scatter+gelu). F32 (not F16) to avoid the
    //                rounding step that compounds drift over the 30+
    //                cascading layers of a real model — see gemma4:26b
    //                NaN-logits regression with F16 intermediate.
    let num_super_blocks = k_in / 32;
    let q81_total_bytes  = n_active * max_n_e * num_super_blocks * 36;
    let gemm_total_elems = n_active * max_n_e * two_n;
    const MAX_WORKSPACE_BYTES: usize = 4 * 1024 * 1024 * 1024;
    if q81_total_bytes > MAX_WORKSPACE_BYTES || gemm_total_elems * 4 > MAX_WORKSPACE_BYTES {
        candle::bail!(
            "moe_gemm_gguf_per_expert_mma: workspace > 4 GB — caller should use dp4a path"
        );
    }
    let q81_alloc  = dev.alloc_zeros::<u8>(q81_total_bytes)?;
    let gemm_alloc = dev.alloc_zeros::<f32>(gemm_total_elems)?;

    let active_ids_alloc = dev.cuda_stream()
        .memcpy_stod(&active_expert_ids)
        .map_err(|e| candle::Error::Msg(format!("active_expert_ids H2D: {e}")))?;

    // Step 1: F32 → Q8_1 batched gather+quantize.
    {
        let inp_ptr  = inputs_slice.device_ptr(inputs_slice.stream()).0 as *const f32;
        let st_ptr   = st_slice.device_ptr(st_slice.stream()).0 as *const i32;
        let act_ptr  = active_ids_alloc.device_ptr(active_ids_alloc.stream()).0 as *const i32;
        let off_ptr  = off_slice.device_ptr(off_slice.stream()).0 as *const i32;
        let q81_ptr  = q81_alloc.device_ptr(q81_alloc.stream()).0 as *mut core::ffi::c_void;
        unsafe {
            ffi::moe_batched_gather_input_rows_f32_to_q81(
                inp_ptr, st_ptr, act_ptr, off_ptr, q81_ptr,
                n_active as i32, max_n_e as i32, k_in as i32, topk as i32,
                stream_i64,
            );
        }
    }

    // Step 2: Q4_K × Q8_1 batched MMA.
    {
        let act_ptr   = active_ids_alloc.device_ptr(active_ids_alloc.stream()).0 as *const i32;
        let off_ptr   = off_slice.device_ptr(off_slice.stream()).0 as *const i32;
        let w_ptr     = weight_data_ptr as *const core::ffi::c_void;
        let q81_ptr   = q81_alloc.device_ptr(q81_alloc.stream()).0 as *const core::ffi::c_void;
        let gemm_ptr  = gemm_alloc.device_ptr(gemm_alloc.stream()).0 as *mut core::ffi::c_void;
        unsafe {
            ffi::moe_q4k_mma_batched_gate_up(
                w_ptr, q81_ptr, act_ptr, off_ptr, gemm_ptr,
                num_experts as i32, n_active as i32, max_n_e as i32,
                two_n as i32, k_in as i32,
                stream_i64,
            );
        }
    }

    // Step 3: F32 logits → F32 output via GELU·mul + scatter.
    {
        let st_ptr   = st_slice.device_ptr(st_slice.stream()).0 as *const i32;
        let act_ptr  = active_ids_alloc.device_ptr(active_ids_alloc.stream()).0 as *const i32;
        let off_ptr  = off_slice.device_ptr(off_slice.stream()).0 as *const i32;
        let gemm_ptr = gemm_alloc.device_ptr(gemm_alloc.stream()).0 as *const f32;
        let out_ptr  = out_alloc.device_ptr(out_alloc.stream()).0 as *mut f32;
        unsafe {
            ffi::moe_batched_gelu_mul_scatter_f32_to_f32(
                gemm_ptr, st_ptr, act_ptr, off_ptr, out_ptr,
                n_active as i32, max_n_e as i32, n_out as i32,
                stream_i64,
            );
        }
    }

    drop(inp_storage); drop(st_storage); drop(off_storage);
    let storage = candle::CudaStorage::wrap_cuda_slice(out_alloc, dev.clone());
    Ok(Tensor::from_storage(
        candle::Storage::Cuda(storage),
        (size_m, n_out),
        candle::op::BackpropOp::none(),
        false,
    ))
}

#[cfg(not(feature = "cuda"))]
pub fn moe_gemm_gguf_per_expert_gate_up_gelu_mul_concat_mma(
    _: &Tensor, _: &QTensor, _: &Tensor, _: &Tensor, _: usize, _: usize,
) -> Result<Tensor> {
    candle::bail!("moe_gemm_gguf_per_expert_gate_up_gelu_mul_concat_mma is cuda-only")
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn moe_gemm_gguf(
    input: &Tensor,
    weights: &QTensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
    is_prefill: bool,
    dtype: candle::DType,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::quantized::GgmlDType;
    use candle::DType;
    use half::{bf16, f16};

    #[allow(clippy::too_many_arguments)]
    fn cuda_fwd(
        input: &Tensor,
        weights: &QTensor,
        topk_weights: &Option<Tensor>,
        sorted_token_ids: &Tensor,
        experts_ids: &Tensor,
        topk: usize,
        is_prefill: bool,
        dtype: DType,
    ) -> Result<Tensor> {
        let (mut size_m, size_k) = input.dims2()?;
        if topk_weights.is_none() {
            size_m *= topk;
        }
        let (num_experts, size_n, size_k1) = weights.shape().dims3()?;
        assert!(
            size_k == size_k1,
            "input {:?} and weight {:?} last dim mismatch!",
            size_k,
            size_k1,
        );
        let dev = input.device().as_cuda_device()?;

        // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5
        let gguf_dtype = match weights.dtype() {
            GgmlDType::Q8_0 => 0,
            GgmlDType::Q4K => 1,
            GgmlDType::Q2K => 2,
            GgmlDType::Q3K => 3,
            GgmlDType::Q5K => 4,
            GgmlDType::Q6K => 5,
            _ => {
                candle::bail!(
                    "moe_gemm_gguf `ISQ` only accept q2k, q3k, q4k, q5k, q6k or q8_0 weights!"
                )
            }
        };

        let weight_ptr = weights.device_ptr()?;

        let topk_weights_ptr = if let Some(topk_weights) = &topk_weights {
            let (topk_weights, _) = topk_weights.storage_and_layout();
            let topk_weights = match &*topk_weights {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!("topk_weights must be a cuda tensor"),
            };
            let w_ptr = topk_weights.device_ptr(topk_weights.stream()).0 as *const f32;
            w_ptr
        } else {
            std::ptr::null()
        };

        let (sorted_token_ids, _) = sorted_token_ids.storage_and_layout();
        let sorted_token_ids = match &*sorted_token_ids {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("sorted_token_ids must be a cuda tensor"),
        };
        let (experts_ids, _) = experts_ids.storage_and_layout();
        let experts_ids = match &*experts_ids {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("experts_ids must be a cuda tensor"),
        };

        let output = unsafe { dev.alloc::<f32>(size_m * size_n) }?;
        let stream = dev.cuda_stream().cu_stream() as i64;
        use candle::op::BackpropOp;
        use core::ffi::c_void;

        assert!(size_k % 8 == 0, "size_k must divisible by 8");
        unsafe {
            if is_prefill {
                let input = input.to_dtype(dtype)?;
                let (input, _) = input.storage_and_layout();
                let (input_ptr, input_dtype) = match &*input {
                    candle::Storage::Cuda(c) => {
                        if dtype == DType::F16 {
                            let c = c.as_cuda_slice::<f16>()?;
                            (c.device_ptr(c.stream()).0 as *const c_void, 0)
                        } else {
                            let c = c.as_cuda_slice::<bf16>()?;
                            (c.device_ptr(c.stream()).0 as *const c_void, 1)
                        }
                    }
                    _ => candle::bail!("input must be a cuda tensor"),
                };
                ffi::moe_gemm_gguf_prefill(
                    input_ptr,  // [size_m or size_m/topk, size_k]
                    weight_ptr, // [num_experts, size_n, size_k]
                    sorted_token_ids.device_ptr(sorted_token_ids.stream()).0 as *const i32,
                    experts_ids.device_ptr(experts_ids.stream()).0 as *const i32,
                    topk_weights_ptr,
                    output.device_ptr(output.stream()).0 as *mut c_void, // [size_m, size_n]
                    num_experts as i32,
                    topk as i32,
                    size_m as i32,
                    size_n as i32,
                    size_k as i32,
                    input_dtype,
                    gguf_dtype as i32, // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5 (for weight)
                    stream,
                );
            } else {
                let (input, _) = input.storage_and_layout();
                let input = match &*input {
                    candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                    _ => candle::bail!("input must be a cuda tensor"),
                };

                ffi::moe_gemm_gguf(
                    input.device_ptr(input.stream()).0 as *const f32, // [size_m or size_m/topk, size_k]
                    weight_ptr as *const c_void, // [num_experts, size_n, size_k]
                    sorted_token_ids.device_ptr(sorted_token_ids.stream()).0 as *const i32,
                    experts_ids.device_ptr(experts_ids.stream()).0 as *const i32,
                    topk_weights_ptr,
                    output.device_ptr(output.stream()).0 as *mut c_void, // [size_m, size_n]
                    num_experts as i32,
                    topk as i32,
                    size_m as i32,
                    size_n as i32,
                    size_k as i32,
                    gguf_dtype as i32, // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5 (for weight)
                    stream,
                );
            }
        }

        let output = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
        let output = Tensor::from_storage(
            candle::Storage::Cuda(output),
            (size_m, size_n),
            BackpropOp::none(),
            false,
        );

        Ok(output)
    }

    match input.dtype() {
        DType::F32 => cuda_fwd(
            input,
            weights,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
            is_prefill,
            dtype,
        ),
        _ => {
            candle::bail!("moe_gemm_gguf only accepts f32 inputs")
        }
    }
}

#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn moe_gemm_gguf(
    _: &Tensor,
    _: &QTensor,
    _: &Option<Tensor>,
    _: &Tensor,
    _: &Tensor,
    _: usize,
    _: bool,
    _: candle::DType,
) -> Result<Tensor> {
    candle::bail!("moe_gemm_gguf is only implemented for the cuda backend")
}

/// Fused gate+up MoE GEMM with SiLU activation and elementwise multiply.
///
/// Replaces the sequence
///     gate = moe_gemm_gguf(input, gate_w)
///     up   = moe_gemm_gguf(input, up_w)
///     out  = silu(gate) * up
/// with a single CUDA launch that:
///   • shares the q8_1-quantized input across both dot products,
///   • holds gate and up weight rows in adjacent shared-memory slots so
///     the per-iteration block load from L1/L2 serves both partial sums,
///   • fuses the activation+multiply into the warp-reduce write so no
///     separate launches and no [M, N] intermediates are needed.
///
/// Used for MoE FFN inner matmul (gate × up) where the activation is SiLU
/// (i.e. SwiGLU). For other activations or for the gate-only matmul
/// without an up partner, fall back to the unfused `moe_gemm_gguf`.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn moe_gemm_gguf_gate_up_silu_mul(
    input: &Tensor,
    gate_weights: &QTensor,
    up_weights: &QTensor,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::quantized::GgmlDType;
    use candle::DType;

    let (size_m, size_k) = input.dims2()?;
    // Gate+up don't apply topk_weights — input arrives at [M / topk, K]
    // (each underlying token replicated topk times via sorted_token_ids).
    let size_m = size_m * topk;
    // Accept both 3D MoE weights [num_experts, N, K] and 2D dense weights
    // [N, K] (treated as num_experts=1 for the dense fusion variant).
    let (num_experts_g, size_n_g, size_k_g) = match gate_weights.shape().dims() {
        [n, k]    => (1, *n, *k),
        [e, n, k] => (*e, *n, *k),
        s => candle::bail!("gate_weights must be 2D or 3D, got {:?}", s),
    };
    let (num_experts_u, size_n_u, size_k_u) = match up_weights.shape().dims() {
        [n, k]    => (1, *n, *k),
        [e, n, k] => (*e, *n, *k),
        s => candle::bail!("up_weights must be 2D or 3D, got {:?}", s),
    };
    if num_experts_g != num_experts_u || size_n_g != size_n_u || size_k_g != size_k_u {
        candle::bail!(
            "moe_gemm_gguf_gate_up_silu_mul: gate {:?} and up {:?} shapes mismatch",
            gate_weights.shape().dims(), up_weights.shape().dims()
        );
    }
    if size_k != size_k_g {
        candle::bail!(
            "moe_gemm_gguf_gate_up_silu_mul: input K={} != weight K={}",
            size_k, size_k_g
        );
    }
    if gate_weights.dtype() != up_weights.dtype() {
        candle::bail!(
            "moe_gemm_gguf_gate_up_silu_mul: gate dtype {:?} != up dtype {:?}",
            gate_weights.dtype(), up_weights.dtype()
        );
    }
    let gguf_dtype = match gate_weights.dtype() {
        GgmlDType::Q8_0 => 0,
        GgmlDType::Q4K  => 1,
        GgmlDType::Q2K  => 2,
        GgmlDType::Q3K  => 3,
        GgmlDType::Q5K  => 4,
        GgmlDType::Q6K  => 5,
        _ => candle::bail!(
            "moe_gemm_gguf_gate_up_silu_mul only accepts q2k/q3k/q4k/q5k/q6k/q8_0 weights"
        ),
    };

    if input.dtype() != DType::F32 {
        candle::bail!("moe_gemm_gguf_gate_up_silu_mul only accepts f32 inputs");
    }
    let dev = input.device().as_cuda_device()?;

    let gate_ptr = gate_weights.device_ptr()?;
    let up_ptr   = up_weights.device_ptr()?;
    let (sorted_token_ids, _) = sorted_token_ids.storage_and_layout();
    let sorted_token_ids = match &*sorted_token_ids {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
        _ => candle::bail!("sorted_token_ids must be a cuda tensor"),
    };
    let (experts_ids, _) = experts_ids.storage_and_layout();
    let experts_ids = match &*experts_ids {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
        _ => candle::bail!("experts_ids must be a cuda tensor"),
    };
    let (input_storage, _) = input.storage_and_layout();
    let input_slice = match &*input_storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("input must be a cuda tensor"),
    };

    let output = unsafe { dev.alloc::<f32>(size_m * size_n_g) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    use candle::op::BackpropOp;
    use core::ffi::c_void;

    if size_k % 8 != 0 {
        candle::bail!("size_k must be divisible by 8");
    }
    unsafe {
        ffi::moe_gemm_gguf_gate_up_silu_mul(
            input_slice.device_ptr(input_slice.stream()).0 as *const f32,
            gate_ptr as *const c_void,
            up_ptr as *const c_void,
            sorted_token_ids.device_ptr(sorted_token_ids.stream()).0 as *const i32,
            experts_ids.device_ptr(experts_ids.stream()).0 as *const i32,
            output.device_ptr(output.stream()).0 as *mut c_void,
            num_experts_g as i32,
            topk as i32,
            size_m as i32,
            size_n_g as i32,
            size_k as i32,
            gguf_dtype as i32,
            stream,
        );
    }

    let output = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
    let output = Tensor::from_storage(
        candle::Storage::Cuda(output),
        (size_m, size_n_g),
        BackpropOp::none(),
        false,
    );
    Ok(output)
}

#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn moe_gemm_gguf_gate_up_silu_mul(
    _: &Tensor,
    _: &QTensor,
    _: &QTensor,
    _: &Tensor,
    _: &Tensor,
    _: usize,
) -> Result<Tensor> {
    candle::bail!("moe_gemm_gguf_gate_up_silu_mul is only implemented for the cuda backend")
}

/// Fused gate||up MoE GEMM with GELU(tanh-approx) + elementwise multiply
/// for the **concatenated** weight layout: `gate_up_weights` has shape
/// `[num_experts, 2*N, K]` where `gate = rows [0..N]` and
/// `up = rows [N..2N]`. Used by the gemma4-MoE FFN where the GGUF
/// stores both as `ffn_gate_up_exps.weight`.
///
/// Replaces the unfused gemma4 sequence
///   `gu = moe_gemm_gguf(input, gate_up_exps)` (output [M*topk, 2N])
///   followed by `split + gelu_tanh + mul` elementwise
/// with one matmul that emits `[M*topk, N]` activated output directly,
/// saving the [M*topk, 2N] intermediate write and 2-3 elementwise
/// launches per layer per token.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn moe_gemm_gguf_gate_up_gelu_mul_concat(
    input: &Tensor,
    gate_up_weights: &QTensor,   // [num_experts, 2*N, K]
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::quantized::GgmlDType;
    use candle::DType;

    let (size_m_in, size_k) = input.dims2()?;
    // Each input row is replicated topk times via sorted_token_ids.
    let size_m = size_m_in * topk;
    // Accept both 3D [num_experts, 2N, K] and 2D [2N, K] (dense, treated
    // as num_experts=1) — same trick as moe_gemm_gguf_gate_up_silu_mul.
    let (num_experts, two_n, size_k_g) = match gate_up_weights.shape().dims() {
        [n, k]    => (1, *n, *k),
        [e, n, k] => (*e, *n, *k),
        s => candle::bail!("gate_up_weights must be 2D or 3D, got {:?}", s),
    };
    if two_n % 2 != 0 {
        candle::bail!(
            "moe_gemm_gguf_gate_up_gelu_mul_concat: 2*N dim must be even, got {}",
            two_n
        );
    }
    let size_n = two_n / 2;
    if size_k != size_k_g {
        candle::bail!(
            "moe_gemm_gguf_gate_up_gelu_mul_concat: input K={} != weight K={}",
            size_k, size_k_g
        );
    }
    let gguf_dtype = match gate_up_weights.dtype() {
        GgmlDType::Q8_0 => 0,
        GgmlDType::Q4K  => 1,
        GgmlDType::Q2K  => 2,
        GgmlDType::Q3K  => 3,
        GgmlDType::Q5K  => 4,
        GgmlDType::Q6K  => 5,
        d => candle::bail!(
            "moe_gemm_gguf_gate_up_gelu_mul_concat: unsupported weight dtype {:?}", d
        ),
    };
    if input.dtype() != DType::F32 {
        candle::bail!("moe_gemm_gguf_gate_up_gelu_mul_concat: input must be F32");
    }
    let dev = input.device().as_cuda_device()?;
    let weight_ptr = gate_up_weights.device_ptr()?;

    let (sorted_token_ids, _) = sorted_token_ids.storage_and_layout();
    let sorted_token_ids = match &*sorted_token_ids {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
        _ => candle::bail!("sorted_token_ids must be cuda"),
    };
    let (experts_ids, _) = experts_ids.storage_and_layout();
    let experts_ids = match &*experts_ids {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
        _ => candle::bail!("experts_ids must be cuda"),
    };
    let (input_storage, _) = input.storage_and_layout();
    let input_slice = match &*input_storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("input must be cuda"),
    };

    let output = unsafe { dev.alloc::<f32>(size_m * size_n) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    use candle::op::BackpropOp;
    use core::ffi::c_void;
    if size_k % 8 != 0 {
        candle::bail!("size_k must be divisible by 8");
    }
    unsafe {
        ffi::moe_gemm_gguf_gate_up_gelu_mul_concat(
            input_slice.device_ptr(input_slice.stream()).0 as *const f32,
            weight_ptr as *const c_void,
            sorted_token_ids.device_ptr(sorted_token_ids.stream()).0 as *const i32,
            experts_ids.device_ptr(experts_ids.stream()).0 as *const i32,
            output.device_ptr(output.stream()).0 as *mut c_void,
            num_experts as i32,
            topk as i32,
            size_m as i32,
            size_n as i32,
            size_k as i32,
            gguf_dtype as i32,
            stream,
        );
    }
    let output = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
    let output = Tensor::from_storage(
        candle::Storage::Cuda(output),
        (size_m, size_n),
        BackpropOp::none(),
        false,
    );
    Ok(output)
}

#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn moe_gemm_gguf_gate_up_gelu_mul_concat(
    _: &Tensor, _: &QTensor, _: &Tensor, _: &Tensor, _: usize,
) -> Result<Tensor> {
    candle::bail!("moe_gemm_gguf_gate_up_gelu_mul_concat is only implemented for the cuda backend")
}

/// Dense (non-MoE) variant of gate||up + GELU(tanh) + mul fusion for
/// the **concatenated** weight layout `[2*N, K]`. Uses the MoE
/// concat kernel with num_experts=1 + dummy expert/token IDs (same
/// trick as `dense_gate_up_silu_mul`) — reads gate from rows [0..N]
/// and up from rows [N..2N] of the single-expert slab and writes
/// `gelu_tanh(gate) * up` to a fresh [M, N] output.
///
/// Replaces the standard 3-launch dense FFN GELU sequence
///   `up_states = ffn_up.forward(x)` (output [M, 2N], internal quantize)
///   `out = gate.gelu() * up` (split + activation + mul)
/// with one quantize + one fused matmul. Single-token decode (M=1) only.
#[cfg(feature = "cuda")]
pub fn dense_gate_up_gelu_mul_concat(
    input: &Tensor,             // [M, K] F32
    gate_up_w: &QTensor,        // [2*N, K] quantized concat
) -> Result<Tensor> {
    let dev = input.device();
    let m = input.dim(0)?;
    if m != 1 {
        candle::bail!("dense_gate_up_gelu_mul_concat: only M=1 supported, got M={}", m);
    }
    // Use the MoE kernel with topk=1 + dummy [0] indices. The 2D
    // [2N, K] weight is interpreted as num_experts=1 by the wrapper.
    let zeros = Tensor::from_slice(&[0u32], (m,), dev)?;
    moe_gemm_gguf_gate_up_gelu_mul_concat(input, gate_up_w, &zeros, &zeros, 1)
}

#[cfg(not(feature = "cuda"))]
pub fn dense_gate_up_gelu_mul_concat(
    _: &Tensor, _: &QTensor,
) -> Result<Tensor> {
    candle::bail!("dense_gate_up_gelu_mul_concat is only implemented for the cuda backend")
}

/// Dense (non-MoE) variant of gate+up+silu+mul fusion. Expects 2D
/// `gate_w` and `up_w` of shape `[N, K]`. Replaces the standard
/// 3-launch `silu(gate.forward(x)) * up.forward(x)` with one fused
/// kernel — saves F32 intermediate writes plus silu and mul launches.
///
/// Single-token decode (M=1) on CUDA. Allocates dummy
/// `sorted_token_ids = [0]` and `expert_ids = [0]` tensors since the
/// underlying MoE kernel expects that interface.
#[cfg(feature = "cuda")]
pub fn dense_gate_up_silu_mul(
    input: &Tensor,           // [M, K] F32
    gate_w: &QTensor,         // [N, K] quantized
    up_w: &QTensor,           // [N, K] quantized
) -> Result<Tensor> {
    let dev = input.device();
    let m = input.dim(0)?;
    if m != 1 {
        candle::bail!("dense_gate_up_silu_mul: only M=1 supported, got M={}", m);
    }
    // Build dummy [0] u32 indices on the same device.
    let zeros = Tensor::from_slice(&[0u32], (m,), dev)?;
    moe_gemm_gguf_gate_up_silu_mul(
        input,
        gate_w,
        up_w,
        &zeros,
        &zeros,
        1, // topk = 1
    )
}

#[cfg(not(feature = "cuda"))]
pub fn dense_gate_up_silu_mul(
    _: &Tensor, _: &QTensor, _: &QTensor,
) -> Result<Tensor> {
    candle::bail!("dense_gate_up_silu_mul is only implemented for the cuda backend")
}

/// True dense gate+up+silu*mul kernel (no MoE expert-routing overhead).
/// Replaces gate.forward + up.forward + fused_silu_mul (3 launches) with
/// 1 input-quantize + 1 fused matmul. Targets dense FFN paths in
/// GenericHetero (qwen3 base, qwen2 etc.) where load-time gate+up
/// concat already saves the matmul launch but still pays for the
/// [2*intermediate] intermediate buffer write.
#[cfg(feature = "cuda")]
pub fn dense_gate_up_silu_mul_v2(
    input: &Tensor,        // [M, K] F32 (M=1)
    gate_w: &QTensor,      // [N, K] quantized
    up_w: &QTensor,        // [N, K] quantized
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::quantized::GgmlDType;
    use candle::DType;

    if input.dtype() != DType::F32 {
        candle::bail!("dense_gate_up_silu_mul_v2: input must be F32");
    }
    let (size_m, size_k) = input.dims2()?;
    if size_m != 1 {
        candle::bail!("dense_gate_up_silu_mul_v2: only M=1 supported, got M={}", size_m);
    }
    let (size_n, size_k_g) = gate_w.shape().dims2()?;
    let (size_n_u, size_k_u) = up_w.shape().dims2()?;
    if size_n != size_n_u || size_k_g != size_k_u {
        candle::bail!("dense_gate_up_silu_mul_v2: gate {:?} and up {:?} shapes mismatch",
            gate_w.shape().dims(), up_w.shape().dims());
    }
    if size_k != size_k_g {
        candle::bail!("dense_gate_up_silu_mul_v2: input K={} != weight K={}", size_k, size_k_g);
    }
    if gate_w.dtype() != up_w.dtype() {
        candle::bail!("dense_gate_up_silu_mul_v2: gate {:?} != up {:?} dtype",
            gate_w.dtype(), up_w.dtype());
    }
    let gguf_dtype = match gate_w.dtype() {
        GgmlDType::Q8_0 => 0,
        GgmlDType::Q4K  => 1,
        GgmlDType::Q2K  => 2,
        GgmlDType::Q3K  => 3,
        GgmlDType::Q5K  => 4,
        GgmlDType::Q6K  => 5,
        d => candle::bail!("dense_gate_up_silu_mul_v2: unsupported weight dtype {:?}", d),
    };

    let dev = input.device().as_cuda_device()?;
    let (input_storage, _) = input.storage_and_layout();
    let input_slice = match &*input_storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("input must be cuda"),
    };

    let gate_ptr = gate_w.device_ptr()?;
    let up_ptr = up_w.device_ptr()?;
    let output = unsafe { dev.alloc::<f32>(size_n) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;

    if size_k % 8 != 0 {
        candle::bail!("size_k must be divisible by 8");
    }
    unsafe {
        ffi::dense_gate_up_silu_mul_v2(
            input_slice.device_ptr(input_slice.stream()).0 as *const f32,
            gate_ptr as *const core::ffi::c_void,
            up_ptr as *const core::ffi::c_void,
            output.device_ptr(output.stream()).0 as *mut f32,
            size_n as i32,
            size_k as i32,
            gguf_dtype as i32,
            stream,
        );
    }

    use candle::op::BackpropOp;
    let storage = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
    Ok(Tensor::from_storage(
        candle::Storage::Cuda(storage),
        (size_m, size_n),
        BackpropOp::none(),
        false,
    ))
}

#[cfg(not(feature = "cuda"))]
pub fn dense_gate_up_silu_mul_v2(
    _: &Tensor, _: &QTensor, _: &QTensor,
) -> Result<Tensor> {
    candle::bail!("dense_gate_up_silu_mul_v2 only implemented for cuda")
}

/// Fused MoE down-projection + topk reduction. Each (token, expert)
/// row's weighted partial result is accumulated into a [M_real, N]
/// output via atomicAdd. Replaces the unfused
///   ys = moe_gemm_gguf(input, down_w, Some(topk_weights))     // [M*topk, N]
///   ys = ys.reshape((M_real, topk, N))?.sum(D::Minus2)?       // [M_real, N]
/// sequence with a single launch.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn moe_gemm_gguf_down_reduce(
    input: &Tensor,
    weights: &QTensor,
    sorted_token_ids: &Tensor,
    expert_ids: &Tensor,
    topk_weights: &Tensor,
    topk: usize,
    n_real_tokens: usize,
    residual: Option<&Tensor>,   // [n_real_tokens, n] in F16/BF16 — when
                                  // provided, output buffer is initialized
                                  // with this cast to F32 before the
                                  // atomicAdd reduction (fuses the post-MLP
                                  // residual add into the kernel).
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::quantized::GgmlDType;
    use candle::DType;

    let (size_m, size_k) = input.dims2()?;
    if size_m != n_real_tokens * topk {
        candle::bail!(
            "moe_gemm_gguf_down_reduce: input M={} != n_real_tokens={} × topk={}",
            size_m, n_real_tokens, topk
        );
    }
    let (num_experts, size_n, size_k1) = weights.shape().dims3()?;
    if size_k != size_k1 {
        candle::bail!(
            "moe_gemm_gguf_down_reduce: input K={} != weight K={}",
            size_k, size_k1
        );
    }

    let gguf_dtype = match weights.dtype() {
        GgmlDType::Q8_0 => 0,
        GgmlDType::Q4K  => 1,
        GgmlDType::Q2K  => 2,
        GgmlDType::Q3K  => 3,
        GgmlDType::Q5K  => 4,
        GgmlDType::Q6K  => 5,
        d => candle::bail!("moe_gemm_gguf_down_reduce: unsupported weight dtype {:?}", d),
    };
    if input.dtype() != DType::F32 {
        candle::bail!("moe_gemm_gguf_down_reduce: input must be F32");
    }

    let dev = input.device().as_cuda_device()?;
    let weight_ptr = weights.device_ptr()?;

    let (input_storage, _) = input.storage_and_layout();
    let input_slice = match &*input_storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("input must be cuda"),
    };
    let (sorted_storage, _) = sorted_token_ids.storage_and_layout();
    let sorted_slice = match &*sorted_storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
        _ => candle::bail!("sorted_token_ids must be cuda"),
    };
    let (experts_storage, _) = expert_ids.storage_and_layout();
    let experts_slice = match &*experts_storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
        _ => candle::bail!("expert_ids must be cuda"),
    };
    let (tw_storage, _) = topk_weights.storage_and_layout();
    let tw_slice = match &*tw_storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("topk_weights must be cuda"),
    };

    // Output buffer. If a residual is supplied, init it from the
    // residual cast to F32 in one launch — folds the post-MLP residual
    // add into the kernel. Otherwise zero-init.
    use half::{bf16, f16};
    use core::ffi::c_void;
    let out_count = n_real_tokens * size_n;
    let stream = dev.cuda_stream().cu_stream() as i64;

    let out_alloc = if let Some(res) = residual {
        let res_c = res.contiguous()?;
        let elems: usize = res_c.shape().elem_count();
        if elems != out_count {
            candle::bail!(
                "moe_gemm_gguf_down_reduce: residual elem_count={} != n_real_tokens*N={}",
                elems, out_count
            );
        }
        // For F32 residual we can cudaMemcpy directly to the output
        // buffer (no cast). For F16/BF16 we use the cast kernel.
        match res.dtype() {
            DType::F32 => {
                let alloc = unsafe { dev.alloc::<f32>(out_count) }?;
                let (rs, _) = res_c.storage_and_layout();
                let src = match &*rs {
                    candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                    _ => candle::bail!("residual must be cuda"),
                };
                // memcpy F32 → F32. Use cast launcher with dtype=2 fallback?
                // Simpler: read+write via the host wrapper or use candle's
                // built-in copy.
                use candle::cuda_backend::cudarc::driver::sys::cuMemcpyDtoDAsync_v2;
                use candle::cuda_backend::cudarc::driver::DevicePtr;
                // Bind context before raw CUDA driver call (mirrors
                // llama.cpp's `cudaSetDevice`). Without this, when
                // per-thread streams are in use, the calling thread
                // may not have the right context bound and the memcpy
                // lands on the wrong stream — silent corruption.
                dev.cuda_stream().context().bind_to_thread()?;
                let dst_dev = alloc.device_ptr(alloc.stream()).0;
                let src_dev = src.device_ptr(src.stream()).0;
                let stream_ptr = dev.cuda_stream().cu_stream();
                unsafe {
                    let st = cuMemcpyDtoDAsync_v2(
                        dst_dev,
                        src_dev,
                        out_count * std::mem::size_of::<f32>(),
                        stream_ptr,
                    );
                    if st != candle::cuda_backend::cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                        candle::bail!("cuMemcpyDtoDAsync (residual init) failed: {:?}", st);
                    }
                }
                alloc
            }
            DType::F16 | DType::BF16 => {
                let alloc = unsafe { dev.alloc::<f32>(out_count) }?;
                let dst = alloc.device_ptr(alloc.stream()).0 as *mut f32;
                let (res_storage, _) = res_c.storage_and_layout();
                let (src_ptr, dt) = match res.dtype() {
                    DType::F16 => {
                        let s = match &*res_storage {
                            candle::Storage::Cuda(c) => c.as_cuda_slice::<f16>()?,
                            _ => candle::bail!("residual must be cuda"),
                        };
                        (s.device_ptr(s.stream()).0 as *const c_void, 0_i32)
                    }
                    DType::BF16 => {
                        let s = match &*res_storage {
                            candle::Storage::Cuda(c) => c.as_cuda_slice::<bf16>()?,
                            _ => candle::bail!("residual must be cuda"),
                        };
                        (s.device_ptr(s.stream()).0 as *const c_void, 1_i32)
                    }
                    _ => unreachable!(),
                };
                unsafe {
                    ffi::cast_init_f32_from_dtype(dst, src_ptr, out_count as i32, dt, stream);
                }
                alloc
            }
            d => candle::bail!(
                "moe_gemm_gguf_down_reduce: residual dtype {:?} unsupported", d
            ),
        }
    } else {
        unsafe { dev.alloc_zeros::<f32>(out_count) }?
    };
    let out_ptr = out_alloc.device_ptr(out_alloc.stream()).0 as *mut f32;

    unsafe {
        ffi::moe_gemm_gguf_down_reduce(
            input_slice.device_ptr(input_slice.stream()).0 as *const f32,
            weight_ptr as *const c_void,
            sorted_slice.device_ptr(sorted_slice.stream()).0 as *const i32,
            experts_slice.device_ptr(experts_slice.stream()).0 as *const i32,
            tw_slice.device_ptr(tw_slice.stream()).0 as *const f32,
            out_ptr,
            num_experts as i32,
            topk as i32,
            size_m as i32,
            size_n as i32,
            size_k as i32,
            gguf_dtype as i32,
            stream,
        );
    }

    use candle::op::BackpropOp;
    let storage = candle::CudaStorage::wrap_cuda_slice(out_alloc, dev.clone());
    Ok(Tensor::from_storage(
        candle::Storage::Cuda(storage),
        (n_real_tokens, size_n),
        BackpropOp::none(),
        false,
    ))
}

#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn moe_gemm_gguf_down_reduce(
    _: &Tensor, _: &QTensor, _: &Tensor, _: &Tensor, _: &Tensor, _: usize, _: usize,
) -> Result<Tensor> {
    candle::bail!("moe_gemm_gguf_down_reduce is only implemented for the cuda backend")
}

/// Fused MoE routing: takes router logits `[n_rows, n_experts]` and
/// returns `(topk_weights, topk_ids)` where each is `[n_rows, n_expert_used]`.
///
/// Replaces the candle op sequence
///   softmax_last_dim → arg_sort_last_dim → narrow → contiguous →
///   gather → sum_keepdim → broadcast_div
/// with a single CUDA launch. Currently supports n_experts ∈
/// {32, 64, 128, 256}; for other expert counts the caller must fall
/// back to the unfused path.
#[cfg(feature = "cuda")]
pub fn topk_softmax(
    logits: &Tensor,
    n_expert_used: usize,
    with_norm: bool,
) -> Result<(Tensor, Tensor)> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::DType;

    let dims = logits.dims();
    if dims.len() != 2 {
        candle::bail!("topk_softmax requires a 2D logits tensor, got {:?}", dims);
    }
    let n_rows = dims[0];
    let n_experts = dims[1];
    if !matches!(n_experts, 32 | 64 | 128 | 256) {
        candle::bail!(
            "topk_softmax only supports n_experts ∈ {{32, 64, 128, 256}}; got {}",
            n_experts
        );
    }
    if n_expert_used == 0 || n_expert_used > n_experts {
        candle::bail!(
            "topk_softmax: n_expert_used={} out of range (n_experts={})",
            n_expert_used, n_experts
        );
    }
    if logits.dtype() != DType::F32 {
        candle::bail!(
            "topk_softmax: expected f32 logits, got {:?}",
            logits.dtype()
        );
    }
    let dev = logits.device().as_cuda_device()?;
    let logits = logits.contiguous()?;
    let (logits_storage, _) = logits.storage_and_layout();
    let logits_slice = match &*logits_storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("topk_softmax: logits must be a cuda tensor"),
    };

    let weights = unsafe { dev.alloc::<f32>(n_rows * n_expert_used) }?;
    let ids = unsafe { dev.alloc::<u32>(n_rows * n_expert_used) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;

    unsafe {
        ffi::topk_softmax(
            logits_slice.device_ptr(logits_slice.stream()).0 as *const f32,
            weights.device_ptr(weights.stream()).0 as *mut f32,
            ids.device_ptr(ids.stream()).0 as *mut u32,
            n_rows as i32,
            n_experts as i32,
            n_expert_used as i32,
            if with_norm { 1 } else { 0 },
            stream,
        );
    }

    use candle::op::BackpropOp;
    let weights_storage = candle::CudaStorage::wrap_cuda_slice(weights, dev.clone());
    let weights = Tensor::from_storage(
        candle::Storage::Cuda(weights_storage),
        (n_rows, n_expert_used),
        BackpropOp::none(),
        false,
    );
    let ids_storage = candle::CudaStorage::wrap_cuda_slice(ids, dev.clone());
    let ids = Tensor::from_storage(
        candle::Storage::Cuda(ids_storage),
        (n_rows, n_expert_used),
        BackpropOp::none(),
        false,
    );
    Ok((weights, ids))
}

#[cfg(not(feature = "cuda"))]
pub fn topk_softmax(_: &Tensor, _: usize, _: bool) -> Result<(Tensor, Tensor)> {
    candle::bail!("topk_softmax is only implemented for the cuda backend")
}

/// Fused MoE gate matmul + softmax + top-k. Replaces the
/// `gate.forward(xs) -> logits` (cublas SGEMV) and `topk_softmax(logits)`
/// pair with a single launch. Caller passes the gate weight directly as a
/// dense F32 tensor; output is the same `(weights, ids)` pair.
///
/// Constraints: `xs` and `gate_w` must be F32 contiguous on CUDA, `xs`
/// shape `[n_rows, hidden]`, `gate_w` shape `[n_experts, hidden]`,
/// `n_experts ∈ {32, 64, 128, 256}`, and `hidden ≤ 8192`. Returns
/// `Ok(None)` if any constraint is violated, so the caller can fall back.
#[cfg(feature = "cuda")]
pub fn gate_topk_softmax(
    xs: &Tensor,
    gate_w: &Tensor,
    n_expert_used: usize,
    with_norm: bool,
) -> Result<Option<(Tensor, Tensor)>> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::DType;

    if xs.dtype() != DType::F32 || gate_w.dtype() != DType::F32 {
        return Ok(None);
    }
    if !xs.device().is_cuda() {
        return Ok(None);
    }
    let x_dims = xs.dims();
    let w_dims = gate_w.dims();
    if x_dims.len() != 2 || w_dims.len() != 2 {
        return Ok(None);
    }
    let (n_rows, hidden) = (x_dims[0], x_dims[1]);
    let (n_experts, w_hidden) = (w_dims[0], w_dims[1]);
    if hidden != w_hidden {
        return Ok(None);
    }
    if hidden > 8192 || (hidden & 3) != 0 {
        return Ok(None);
    }
    if !matches!(n_experts, 32 | 64 | 128 | 256) {
        return Ok(None);
    }
    if n_expert_used == 0 || n_expert_used > n_experts {
        return Ok(None);
    }

    let dev = xs.device().as_cuda_device()?;
    let xs = xs.contiguous()?;
    let gate_w = gate_w.contiguous()?;
    let (xs_storage, _) = xs.storage_and_layout();
    let (gw_storage, _) = gate_w.storage_and_layout();
    let xs_slice = match &*xs_storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => return Ok(None),
    };
    let gw_slice = match &*gw_storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => return Ok(None),
    };

    let weights = unsafe { dev.alloc::<f32>(n_rows * n_expert_used) }?;
    let ids = unsafe { dev.alloc::<u32>(n_rows * n_expert_used) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;

    unsafe {
        ffi::gate_topk_softmax(
            xs_slice.device_ptr(xs_slice.stream()).0 as *const core::ffi::c_void,
            gw_slice.device_ptr(gw_slice.stream()).0 as *const core::ffi::c_void,
            weights.device_ptr(weights.stream()).0 as *mut core::ffi::c_void,
            ids.device_ptr(ids.stream()).0 as *mut core::ffi::c_void,
            hidden as i32,
            n_rows as i32,
            n_experts as i32,
            n_expert_used as i32,
            if with_norm { 1 } else { 0 },
            stream,
        );
    }

    use candle::op::BackpropOp;
    let weights_storage = candle::CudaStorage::wrap_cuda_slice(weights, dev.clone());
    let weights = Tensor::from_storage(
        candle::Storage::Cuda(weights_storage),
        (n_rows, n_expert_used),
        BackpropOp::none(),
        false,
    );
    let ids_storage = candle::CudaStorage::wrap_cuda_slice(ids, dev.clone());
    let ids = Tensor::from_storage(
        candle::Storage::Cuda(ids_storage),
        (n_rows, n_expert_used),
        BackpropOp::none(),
        false,
    );
    Ok(Some((weights, ids)))
}

#[cfg(not(feature = "cuda"))]
pub fn gate_topk_softmax(_: &Tensor, _: &Tensor, _: usize, _: bool) -> Result<Option<(Tensor, Tensor)>> {
    Ok(None)
}

/// Multi-block F32 GEMV for the MoE router gate. Replaces cublas SGEMV
/// (overhead-bound at ~16us/call on 2048×128) with a hand-rolled
/// multi-block kernel. Returns `Ok(None)` if the shape isn't supported,
/// so the caller can fall back to `gate.forward(xs)`.
#[cfg(feature = "cuda")]
pub fn gate_gemv_f32(xs: &Tensor, gate_w: &Tensor) -> Result<Option<Tensor>> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::DType;

    if xs.dtype() != DType::F32 || gate_w.dtype() != DType::F32 {
        return Ok(None);
    }
    if !xs.device().is_cuda() {
        return Ok(None);
    }
    let x_dims = xs.dims();
    let w_dims = gate_w.dims();
    if x_dims.len() != 2 || w_dims.len() != 2 {
        return Ok(None);
    }
    let (n_rows, hidden) = (x_dims[0], x_dims[1]);
    let (n_experts, w_hidden) = (w_dims[0], w_dims[1]);
    if hidden != w_hidden || (hidden & 31) != 0 {
        return Ok(None);
    }

    let dev = xs.device().as_cuda_device()?;
    let xs = xs.contiguous()?;
    let gate_w = gate_w.contiguous()?;
    let (xs_storage, _) = xs.storage_and_layout();
    let (gw_storage, _) = gate_w.storage_and_layout();
    let xs_slice = match &*xs_storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => return Ok(None),
    };
    let gw_slice = match &*gw_storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => return Ok(None),
    };

    let logits = unsafe { dev.alloc::<f32>(n_rows * n_experts) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;

    unsafe {
        ffi::gate_gemv_f32(
            xs_slice.device_ptr(xs_slice.stream()).0 as *const core::ffi::c_void,
            gw_slice.device_ptr(gw_slice.stream()).0 as *const core::ffi::c_void,
            logits.device_ptr(logits.stream()).0 as *mut core::ffi::c_void,
            hidden as i32,
            n_rows as i32,
            n_experts as i32,
            stream,
        );
    }

    use candle::op::BackpropOp;
    let logits_storage = candle::CudaStorage::wrap_cuda_slice(logits, dev.clone());
    Ok(Some(Tensor::from_storage(
        candle::Storage::Cuda(logits_storage),
        (n_rows, n_experts),
        BackpropOp::none(),
        false,
    )))
}

#[cfg(not(feature = "cuda"))]
pub fn gate_gemv_f32(_: &Tensor, _: &Tensor) -> Result<Option<Tensor>> {
    Ok(None)
}

/// Scatter one F16 vector into the Q4 KV residual buffer at `slot`.
/// `src` and `dst` are raw cudarc CudaSlice<f16> handles via candle's
/// CudaDevice. Caller is responsible for matching n_kv * head_dim
/// elements between src and the slot column in dst.
#[cfg(feature = "cuda")]
pub unsafe fn kv_residual_scatter_f16_raw(
    src: *const core::ffi::c_void,
    dst: *mut core::ffi::c_void,
    n_kv: i32,
    head_dim: i32,
    slot: i32,
    stream: i64,
) {
    ffi::kv_residual_scatter_f16(src, dst, n_kv, head_dim, slot, stream)
}

/// Device-slot variant of `kv_residual_scatter_f16_raw`. Reads `slot`
/// from `slot_dev[0] % 32` on the GPU instead of taking a host int.
/// Required for CUDA graph capture of Q4 KV append — the host updates
/// `slot_dev` outside the captured region each token.
#[cfg(feature = "cuda")]
pub unsafe fn kv_residual_scatter_f16_dev_slot_raw(
    src: *const core::ffi::c_void,
    dst: *mut core::ffi::c_void,
    slot_dev: *const core::ffi::c_void,
    n_kv: i32,
    head_dim: i32,
    stream: i64,
) {
    ffi::kv_residual_scatter_f16_dev_slot(src, dst, slot_dev, n_kv, head_dim, stream)
}

/// Byte-copy scatter for the Q4_0 V append path, with device-side
/// position. Replaces `memcpy_dtod(dst[pos*token_bytes ..])` whose host
/// offset would freeze under CUDA graph capture.
#[cfg(feature = "cuda")]
pub unsafe fn q4_v_scatter_bytes_dev_pos_raw(
    src: *const core::ffi::c_void,
    dst: *mut core::ffi::c_void,
    pos_dev: *const core::ffi::c_void,
    token_bytes: i32,
    stream: i64,
) {
    debug_assert!(token_bytes % 4 == 0, "token_bytes must be multiple of 4");
    ffi::q4_v_scatter_bytes_dev_pos(src, dst, pos_dev, token_bytes, stream)
}

/// Conditional Q4_0 quantize+flush of the K residual under graph capture.
/// See `ffi::flush_k_residual_q4_dev_pos` for semantics.
#[cfg(feature = "cuda")]
pub unsafe fn flush_k_residual_q4_dev_pos_raw(
    residual: *const core::ffi::c_void,
    k_blocks: *mut core::ffi::c_void,
    pos_dev: *const core::ffi::c_void,
    n_kv: i32,
    head_dim: i32,
    max_seq_blocks: i32,
    stream: i64,
) {
    ffi::flush_k_residual_q4_dev_pos(
        residual, k_blocks, pos_dev, n_kv, head_dim, max_seq_blocks, stream,
    )
}

/// Fused RMS norm + quantized matmul (single-token decode).
/// Returns the F32 matmul output [out_rows] = (rms_norm(x, w_norm)) @ w_mm^T.
/// `w_mm` is the QTensor for the matmul weight; supported quant types
/// are Q8_0/Q4K/Q2K/Q3K/Q5K/Q6K (same as candle's mmvq path).
#[cfg(feature = "cuda")]
pub fn rms_qmatmul(
    x: &Tensor,             // [..., hidden] F32; only [1, hidden] / [hidden] supported
    w_norm: &Tensor,        // [hidden] F32
    w_mm: &QTensor,         // [out_rows, hidden] quantized
    rms_eps: f32,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::quantized::GgmlDType;
    use candle::DType;
    use core::ffi::c_void;

    if x.dtype() != DType::F32 || w_norm.dtype() != DType::F32 {
        candle::bail!(
            "rms_qmatmul: x and w_norm must be F32; got {:?} and {:?}",
            x.dtype(), w_norm.dtype()
        );
    }
    let hidden = x.dim(candle::D::Minus1)?;
    if w_norm.shape().elem_count() != hidden {
        candle::bail!(
            "rms_qmatmul: w_norm len {} != hidden {}",
            w_norm.shape().elem_count(), hidden
        );
    }
    let (out_rows, k) = w_mm.shape().dims2()?;
    if k != hidden {
        candle::bail!("rms_qmatmul: w_mm K={} != hidden={}", k, hidden);
    }
    if hidden % 32 != 0 {
        candle::bail!("rms_qmatmul: hidden must be divisible by 32 (got {})", hidden);
    }

    let quant_type = match w_mm.dtype() {
        GgmlDType::Q8_0 => 0,
        GgmlDType::Q4K  => 1,
        GgmlDType::Q2K  => 2,
        GgmlDType::Q3K  => 3,
        GgmlDType::Q5K  => 4,
        GgmlDType::Q6K  => 5,
        d => candle::bail!("rms_qmatmul: unsupported weight dtype {:?}", d),
    };

    let dev = x.device().as_cuda_device()?;
    let x_c = x.contiguous()?;
    let wn_c = w_norm.contiguous()?;

    let (xs, _) = x_c.storage_and_layout();
    let xs_slice = match &*xs {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("x must be cuda"),
    };
    let (ws, _) = wn_c.storage_and_layout();
    let ws_slice = match &*ws {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("w_norm must be cuda"),
    };
    let w_mm_ptr = w_mm.device_ptr()?;

    let out_alloc = unsafe { dev.alloc::<f32>(out_rows) }?;
    let out_ptr = out_alloc.device_ptr(out_alloc.stream()).0 as *mut f32;
    let stream = dev.cuda_stream().cu_stream() as i64;

    unsafe {
        ffi::rms_qmatmul(
            xs_slice.device_ptr(xs_slice.stream()).0 as *const f32,
            ws_slice.device_ptr(ws_slice.stream()).0 as *const f32,
            w_mm_ptr as *const c_void,
            out_ptr,
            hidden as i32,
            out_rows as i32,
            rms_eps,
            quant_type as i32,
            stream,
        );
    }

    use candle::op::BackpropOp;
    let storage = candle::CudaStorage::wrap_cuda_slice(out_alloc, dev.clone());
    Ok(Tensor::from_storage(
        candle::Storage::Cuda(storage),
        (1, 1, out_rows),
        BackpropOp::none(),
        false,
    ))
}

#[cfg(not(feature = "cuda"))]
pub fn rms_qmatmul(_: &Tensor, _: &Tensor, _: &QTensor, _: f32) -> Result<Tensor> {
    candle::bail!("rms_qmatmul is only implemented for the cuda backend")
}

/// Two-launch fused (rms_norm + quantize_q8_1) → MVQ. Saves one
/// launch vs the rms_norm + wqkv.forward sequence (which internally
/// does quantize + matmul = 2 launches; total 3 → 2). Avoids the
/// per-block redundancy of the all-fused rms_qmatmul.
#[cfg(feature = "cuda")]
pub fn rms_norm_then_qmatmul(
    x: &Tensor,             // [..., hidden] F32
    w_norm: &Tensor,        // [hidden] F32
    w_mm: &QTensor,         // [out_rows, hidden] quantized
    rms_eps: f32,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::quantized::cuda::mvq_via_pre_quantized_q8_1;
    use candle::DType;
    use core::ffi::c_void;

    if x.dtype() != DType::F32 || w_norm.dtype() != DType::F32 {
        candle::bail!("rms_norm_then_qmatmul: x and w_norm must be F32");
    }
    let hidden = x.dim(candle::D::Minus1)?;
    if w_norm.shape().elem_count() != hidden {
        candle::bail!("rms_norm_then_qmatmul: w_norm len mismatch");
    }
    let (out_rows, k) = w_mm.shape().dims2()?;
    if k != hidden {
        candle::bail!("rms_norm_then_qmatmul: w_mm K={} != hidden={}", k, hidden);
    }
    if hidden % 32 != 0 {
        candle::bail!("rms_norm_then_qmatmul: hidden must be divisible by 32");
    }
    if hidden > 16384 {
        candle::bail!("rms_norm_then_qmatmul: hidden {} exceeds single-block limit 16384", hidden);
    }

    let dev = x.device().as_cuda_device()?;
    let x_c = x.contiguous()?;
    let wn_c = w_norm.contiguous()?;

    // y_q8_1 buffer: padded to MATRIX_ROW_PADDING (512). Size = padded/32 * 36.
    let kx_padded = ((hidden + 511) / 512) * 512;
    let y_q8_1_bytes = (kx_padded / 32) * 36;
    // Important: zero-init so the trailing padding region (hidden..kx_padded)
    // produces exact-zero contributions in the matmul (q values + ds=0).
    let y_q8_1 = dev.alloc_zeros::<u8>(y_q8_1_bytes)?;

    let stream = dev.cuda_stream().cu_stream() as i64;
    let x_off = {
        let (_s, l) = x_c.storage_and_layout();
        l.start_offset()
    };
    let (xs_st, _) = x_c.storage_and_layout();
    let xs_slice = match &*xs_st {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("x must be cuda"),
    };
    let (ws_st, _) = wn_c.storage_and_layout();
    let ws_slice = match &*ws_st {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("w_norm must be cuda"),
    };
    let x_base = xs_slice.device_ptr(xs_slice.stream()).0 as usize;
    let x_ptr = (x_base + x_off * 4) as *const f32;
    let w_ptr = ws_slice.device_ptr(ws_slice.stream()).0 as *const f32;
    let y_ptr = y_q8_1.device_ptr(y_q8_1.stream()).0 as *mut c_void;

    // Step 1: fused rms_norm + quantize_q8_1.
    unsafe {
        ffi::rms_quantize_q8_1(x_ptr, w_ptr, y_ptr, hidden as i32, rms_eps, stream);
    }

    // Step 2: standard MVQ on the pre-quantized buffer.
    let qstor = match &w_mm.storage() {
        candle::quantized::QStorage::Cuda(s) => s,
        _ => candle::bail!("w_mm must be on CUDA"),
    };
    let out_storage = mvq_via_pre_quantized_q8_1(qstor, &y_q8_1, hidden, out_rows, 1)?;

    use candle::op::BackpropOp;
    Ok(Tensor::from_storage(
        candle::Storage::Cuda(out_storage),
        (1, 1, out_rows),
        BackpropOp::none(),
        false,
    ))
}

#[cfg(not(feature = "cuda"))]
pub fn rms_norm_then_qmatmul(_: &Tensor, _: &Tensor, _: &QTensor, _: f32) -> Result<Tensor> {
    candle::bail!("rms_norm_then_qmatmul is only implemented for the cuda backend")
}

/// Test wrapper for the IMMA primitive — only used by `test_mma`.
#[cfg(feature = "cuda")]
pub unsafe fn mma_test_m16n8k32(
    a: *const core::ffi::c_void,
    b: *const core::ffi::c_void,
    d: *mut core::ffi::c_void,
    stream: i64,
) {
    ffi::mma_test_m16n8k32(a, b, d, stream)
}

/// Tensor-core Q4_K × Q8_1 MMVQ for batch=1 decode. Replaces the
/// dp4a inner loop with `mma.sync.aligned.m16n8k32` — processes 16
/// weight rows in parallel per 32-K chunk. `ncols_x` must be a
/// multiple of 256.
#[cfg(feature = "cuda")]
pub unsafe fn q4k_mmvq_imma(
    vx: *const core::ffi::c_void,
    vy: *const core::ffi::c_void,
    dst: *mut core::ffi::c_void,
    ncols_x: i32,
    nrows_x: i32,
    stream: i64,
) {
    ffi::q4k_mmvq_imma(vx, vy, dst, ncols_x, nrows_x, stream)
}

/// Tensor-core MoE Q4_K gate||up matmul + GELU(tanh) + mul. Drop-in
/// IMMA replacement for the dp4a `moe_gemm_gguf_gate_up_gelu_mul_concat`
/// (for Q4_K weights with K%256=0). Falls back at the dispatch site if
/// the weight dtype is not Q4_K or K isn't aligned.
#[cfg(feature = "cuda")]
pub fn moe_q4k_imma_gate_up_gelu_mul_concat(
    input: &Tensor,                   // F32 [M, K]
    gate_up_weights: &QTensor,        // Q4K [num_experts, 2*N, K]
    sorted_token_ids: &Tensor,        // U32 [size_m]
    experts_ids: &Tensor,             // U32 [size_m]
    topk: usize,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::quantized::GgmlDType;
    use candle::DType;
    use candle::op::BackpropOp;
    use core::ffi::c_void;

    if gate_up_weights.dtype() != GgmlDType::Q4K {
        candle::bail!("moe_q4k_imma_gate_up_gelu_mul_concat: requires Q4_K weights");
    }
    if input.dtype() != DType::F32 {
        candle::bail!("moe_q4k_imma_gate_up_gelu_mul_concat: input must be F32");
    }
    let (size_m_in, size_k) = input.dims2()?;
    if size_k % 256 != 0 {
        candle::bail!(
            "moe_q4k_imma_gate_up_gelu_mul_concat: K={} not a multiple of 256", size_k
        );
    }
    let size_m = size_m_in * topk;
    let (num_experts, two_n, size_k_g) = match gate_up_weights.shape().dims() {
        [n, k]    => (1usize, *n, *k),
        [e, n, k] => (*e, *n, *k),
        s => candle::bail!("gate_up_weights must be 2D or 3D, got {:?}", s),
    };
    if two_n % 2 != 0 {
        candle::bail!(
            "moe_q4k_imma_gate_up_gelu_mul_concat: 2*N must be even, got {}", two_n
        );
    }
    let size_n = two_n / 2;
    if size_k != size_k_g {
        candle::bail!(
            "moe_q4k_imma_gate_up_gelu_mul_concat: input K={} != weight K={}",
            size_k, size_k_g
        );
    }

    let dev = input.device().as_cuda_device()?;
    let weight_ptr = gate_up_weights.device_ptr()?;

    let (input_storage, _) = input.storage_and_layout();
    let input_slice = match &*input_storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("input must be cuda"),
    };
    let (sorted_storage, _) = sorted_token_ids.storage_and_layout();
    let sorted_slice = match &*sorted_storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
        _ => candle::bail!("sorted_token_ids must be cuda"),
    };
    let (experts_storage, _) = experts_ids.storage_and_layout();
    let experts_slice = match &*experts_storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
        _ => candle::bail!("experts_ids must be cuda"),
    };

    let output = unsafe { dev.alloc::<f32>(size_m * size_n) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;

    unsafe {
        ffi::moe_q4k_imma_gate_up_gelu_mul_concat(
            input_slice.device_ptr(input_slice.stream()).0 as *const c_void,
            weight_ptr as *const c_void,
            sorted_slice.device_ptr(sorted_slice.stream()).0 as *const i32,
            experts_slice.device_ptr(experts_slice.stream()).0 as *const i32,
            output.device_ptr(output.stream()).0 as *mut c_void,
            num_experts as i32,
            topk as i32,
            size_m as i32,
            size_n as i32,
            size_k as i32,
            stream,
        );
    }

    let storage = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
    Ok(Tensor::from_storage(
        candle::Storage::Cuda(storage),
        (size_m, size_n),
        BackpropOp::none(),
        false,
    ))
}

#[cfg(not(feature = "cuda"))]
pub fn moe_q4k_imma_gate_up_gelu_mul_concat(
    _: &Tensor, _: &QTensor, _: &Tensor, _: &Tensor, _: usize,
) -> Result<Tensor> {
    candle::bail!("moe_q4k_imma_gate_up_gelu_mul_concat is cuda-only")
}

/// Fused quantized matmul + residual add (single-token decode).
/// Returns `(W_q @ x) + residual` in one launch.
/// `x`: F32 [..., hidden] (only [1, 1, hidden] / [hidden] supported)
/// `residual`: F32 [out_rows] — pre-initialized for the atomicAdd.
#[cfg(feature = "cuda")]
pub fn qmatmul_add(
    x: &Tensor,
    w_q: &QTensor,
    residual: &Tensor,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::quantized::GgmlDType;
    use candle::DType;
    use core::ffi::c_void;

    if x.dtype() != DType::F32 || residual.dtype() != DType::F32 {
        candle::bail!(
            "qmatmul_add: x and residual must be F32; got {:?} / {:?}",
            x.dtype(), residual.dtype()
        );
    }
    let hidden = x.dim(candle::D::Minus1)?;
    let (out_rows, k) = w_q.shape().dims2()?;
    if k != hidden {
        candle::bail!("qmatmul_add: w_q K={} != hidden={}", k, hidden);
    }
    if residual.shape().elem_count() != out_rows {
        candle::bail!(
            "qmatmul_add: residual elem_count={} != out_rows={}",
            residual.shape().elem_count(), out_rows
        );
    }
    if hidden % 32 != 0 {
        candle::bail!("qmatmul_add: hidden must be divisible by 32 (got {})", hidden);
    }

    let quant_type = match w_q.dtype() {
        GgmlDType::Q8_0 => 0,
        GgmlDType::Q4K  => 1,
        GgmlDType::Q2K  => 2,
        GgmlDType::Q3K  => 3,
        GgmlDType::Q5K  => 4,
        GgmlDType::Q6K  => 5,
        d => candle::bail!("qmatmul_add: unsupported weight dtype {:?}", d),
    };

    let dev = x.device().as_cuda_device()?;
    let x_c = x.contiguous()?;
    let res_c = residual.contiguous()?;

    // Pre-init output with residual (cuMemcpyDtoDAsync of F32 → F32).
    // Apply the residual tensor's start_offset to the source pointer —
    // candle's `as_cuda_slice().device_ptr()` returns the storage base,
    // so we'd otherwise read the wrong bytes when residual is a view.
    let out_alloc = unsafe { dev.alloc::<f32>(out_rows) }?;
    use candle::cuda_backend::cudarc::driver::sys::cuMemcpyDtoDAsync_v2;
    // Bind context before raw CUDA driver calls (mirrors
    // llama.cpp's `cudaSetDevice` discipline). Required when
    // per-thread streams are the candle default.
    dev.cuda_stream().context().bind_to_thread()?;
    let stream_ptr = dev.cuda_stream().cu_stream();
    let res_offset = {
        let (_storage, layout) = res_c.storage_and_layout();
        layout.start_offset()
    };
    let (rs, _) = res_c.storage_and_layout();
    let res_slice = match &*rs {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("residual must be cuda"),
    };
    let dst_dev = out_alloc.device_ptr(out_alloc.stream()).0;
    let src_base = res_slice.device_ptr(res_slice.stream()).0 as usize;
    let src_dev = (src_base + res_offset * std::mem::size_of::<f32>()) as u64;
    unsafe {
        // DEBUG: when LLMSERVER_MOE_WO_RES_NORES=1, init dst to zeros
        // (skip residual). If output matches wo.forward (without
        // residual add), the matmul kernel is fine and the bug is in
        // residual init. If still wrong, the matmul kernel itself is
        // the bug.
        if std::env::var("LLMSERVER_MOE_WO_RES_NORES").map_or(false, |v| v == "1") {
            use candle::cuda_backend::cudarc::driver::sys::cuMemsetD32Async;
            let st = cuMemsetD32Async(
                dst_dev,
                0,
                out_rows,
                stream_ptr,
            );
            if st != candle::cuda_backend::cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                candle::bail!("qmatmul_add: cuMemsetD32Async failed: {:?}", st);
            }
        } else {
            let st = cuMemcpyDtoDAsync_v2(
                dst_dev,
                src_dev,
                out_rows * std::mem::size_of::<f32>(),
                stream_ptr,
            );
            if st != candle::cuda_backend::cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                candle::bail!("qmatmul_add: cuMemcpyDtoDAsync failed: {:?}", st);
            }
        }
    }

    let x_offset = {
        let (_storage, layout) = x_c.storage_and_layout();
        layout.start_offset()
    };
    let (xs, _) = x_c.storage_and_layout();
    let xs_slice = match &*xs {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("x must be cuda"),
    };
    let w_ptr = w_q.device_ptr()?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    let out_ptr = out_alloc.device_ptr(out_alloc.stream()).0 as *mut f32;
    let x_base = xs_slice.device_ptr(xs_slice.stream()).0 as usize;
    let x_ptr = (x_base + x_offset * std::mem::size_of::<f32>()) as *const f32;

    // Allocate q8_1 scratch via candle (instead of cudaMallocAsync inside
    // the kernel). This puts the scratch through cudarc's tracked pool
    // and unifies pool management with the rest of candle's allocations.
    let kx_padded = ((hidden + 511) / 512) * 512;
    let q8_blocks = kx_padded / 32; // QK8_1 = 32
    // block_q8_1 size = 32 int8 + 1 half2 (4 bytes) = 36 bytes. Round up to
    // u32 elements: 9 u32 per block.
    let scratch_u32 = q8_blocks * 9;
    let y_q8_1_scratch = unsafe { dev.alloc::<u32>(scratch_u32) }?;
    let scratch_ptr = y_q8_1_scratch.device_ptr(y_q8_1_scratch.stream()).0 as *mut c_void;

    unsafe {
        ffi::qmatmul_add(
            x_ptr,
            w_ptr as *const c_void,
            scratch_ptr,
            out_ptr,
            hidden as i32,
            out_rows as i32,
            quant_type as i32,
            stream,
        );
    }
    // Keep scratch alive past kernel launch (drop after).
    drop(y_q8_1_scratch);
    // DEBUG: sync after to make timing-related issues observable.
    if std::env::var("LLMSERVER_MOE_WO_RES_SYNC").map_or(false, |v| v == "1") {
        let _ = dev.cuda_stream().synchronize();
    }

    use candle::op::BackpropOp;
    let storage = candle::CudaStorage::wrap_cuda_slice(out_alloc, dev.clone());
    let t = Tensor::from_storage(
        candle::Storage::Cuda(storage),
        (1, 1, out_rows),
        BackpropOp::none(),
        false,
    );
    if std::env::var("LLMSERVER_MOE_WO_RES_FORCE_COPY").map_or(false, |v| v == "1") {
        // Copy through standard candle path so the resulting CudaSlice has
        // proper read/write event tracking (which our raw-pointer write
        // bypasses).
        return t.copy();
    }
    Ok(t)
}

#[cfg(not(feature = "cuda"))]
pub fn qmatmul_add(_: &Tensor, _: &QTensor, _: &Tensor) -> Result<Tensor> {
    candle::bail!("qmatmul_add is only implemented for the cuda backend")
}

/// Fused (a + b) + RMS norm: returns (xs = a + b, normed = gamma*xs/rms(xs))
/// in a single launch. Saves one kernel launch vs separate broadcast_add +
/// rms_norm.
///
/// All tensors must be F32 on the same CUDA device. Length determined by
/// `a.shape().elem_count()` (must equal b and gamma).
#[cfg(feature = "cuda")]
pub fn add_rms_norm(
    a: &Tensor,
    b: &Tensor,
    gamma: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::DType;

    if a.dtype() != DType::F32 || b.dtype() != DType::F32 || gamma.dtype() != DType::F32 {
        candle::bail!(
            "add_rms_norm: all inputs must be F32 (got {:?} {:?} {:?})",
            a.dtype(), b.dtype(), gamma.dtype()
        );
    }
    let hidden = a.shape().elem_count();
    if b.shape().elem_count() != hidden || gamma.shape().elem_count() != hidden {
        candle::bail!(
            "add_rms_norm: shape mismatch (a={}, b={}, gamma={})",
            hidden, b.shape().elem_count(), gamma.shape().elem_count()
        );
    }
    if hidden > 16384 {
        candle::bail!("add_rms_norm: hidden {} exceeds single-block limit 16384", hidden);
    }

    let dev = a.device().as_cuda_device()?;
    let a_c = a.contiguous()?;
    let b_c = b.contiguous()?;
    let g_c = gamma.contiguous()?;

    let stream_ptr = dev.cuda_stream().cu_stream() as i64;
    let xs_alloc = unsafe { dev.alloc::<f32>(hidden) }?;
    let nm_alloc = unsafe { dev.alloc::<f32>(hidden) }?;

    let a_off = a_c.layout().start_offset();
    let b_off = b_c.layout().start_offset();
    let g_off = g_c.layout().start_offset();
    let (a_st, _) = a_c.storage_and_layout();
    let (b_st, _) = b_c.storage_and_layout();
    let (g_st, _) = g_c.storage_and_layout();
    let a_slice = match &*a_st {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("a must be cuda"),
    };
    let b_slice = match &*b_st {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("b must be cuda"),
    };
    let g_slice = match &*g_st {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("gamma must be cuda"),
    };
    let a_base = a_slice.device_ptr(a_slice.stream()).0 as usize;
    let b_base = b_slice.device_ptr(b_slice.stream()).0 as usize;
    let g_base = g_slice.device_ptr(g_slice.stream()).0 as usize;
    let a_ptr = (a_base + a_off * 4) as *const f32;
    let b_ptr = (b_base + b_off * 4) as *const f32;
    let g_ptr = (g_base + g_off * 4) as *const f32;
    let xs_ptr = xs_alloc.device_ptr(xs_alloc.stream()).0 as *mut f32;
    let nm_ptr = nm_alloc.device_ptr(nm_alloc.stream()).0 as *mut f32;

    unsafe {
        ffi::add_rms_norm(
            a_ptr, b_ptr, g_ptr, xs_ptr, nm_ptr,
            hidden as i32, eps, stream_ptr,
        );
    }

    use candle::op::BackpropOp;
    let xs_storage = candle::CudaStorage::wrap_cuda_slice(xs_alloc, dev.clone());
    let nm_storage = candle::CudaStorage::wrap_cuda_slice(nm_alloc, dev.clone());
    let xs_t = Tensor::from_storage(
        candle::Storage::Cuda(xs_storage),
        a.shape().clone(),
        BackpropOp::none(),
        false,
    );
    let nm_t = Tensor::from_storage(
        candle::Storage::Cuda(nm_storage),
        a.shape().clone(),
        BackpropOp::none(),
        false,
    );
    Ok((xs_t, nm_t))
}

#[cfg(not(feature = "cuda"))]
pub fn add_rms_norm(_: &Tensor, _: &Tensor, _: &Tensor, _: f32) -> Result<(Tensor, Tensor)> {
    candle::bail!("add_rms_norm is only implemented for the cuda backend")
}

/// Fused post-QKV attention prep for single-token decode.
///
/// Inputs:
///   `qkv`: F32 packed [n_q*hd + 2*n_kv*hd] (output of fused QKV matmul,
///          with seq=1 squeezed)
///   `q_norm_w`, `k_norm_w`: F32 [hd] RMS-norm weights
///   `rope_cos`, `rope_sin`: T (matching `out_dtype`) [hd/2] RoPE tables
///                           for the current position
/// Outputs (return tuple):
///   `q_out`: T [n_q, hd]
///   `k_out`: T [n_kv, hd]
///   `v_out`: T [n_kv, hd]
///
/// `out_dtype` must be F16 or BF16. `rope_style` is 0 (neox) or 1
/// (interleaved). All inputs must be on the same CUDA device.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn attn_post_qkv_decode(
    qkv: &Tensor,
    q_norm_w: &Tensor,
    k_norm_w: &Tensor,
    rope_cos: &Tensor,        // [max_seq, hd/2] full table at offset 0
    rope_sin: &Tensor,        // [max_seq, hd/2]
    n_q: usize,
    n_kv: usize,
    hd: usize,
    rope_pos: usize,
    rms_eps: f32,
    q_scale: f32,            // multiplied into Q only — caller passes
                              // 1/sqrt(head_dim) to fold the attention
                              // scale (eliminates a downstream affine).
    out_dtype: candle::DType,
    rope_style: i32,
) -> Result<(Tensor, Tensor, Tensor)> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::DType;
    use core::ffi::c_void;
    use half::{bf16, f16};

    if qkv.dtype() != DType::F32 {
        candle::bail!("attn_post_qkv_decode: qkv must be F32, got {:?}", qkv.dtype());
    }
    if q_norm_w.dtype() != DType::F32 || k_norm_w.dtype() != DType::F32 {
        candle::bail!("attn_post_qkv_decode: norm weights must be F32");
    }
    if rope_cos.dtype() != out_dtype || rope_sin.dtype() != out_dtype {
        candle::bail!(
            "attn_post_qkv_decode: rope_cos/sin dtype {:?}/{:?} must match out_dtype {:?}",
            rope_cos.dtype(), rope_sin.dtype(), out_dtype
        );
    }

    let dev = qkv.device().as_cuda_device()?;
    let qkv_c = qkv.contiguous()?;
    let qn_c  = q_norm_w.contiguous()?;
    let kn_c  = k_norm_w.contiguous()?;
    // rope_cos / rope_sin are passed as the FULL [max_seq, hd/2] table
    // — the kernel reads the right row using the rope_pos parameter.
    // No offset gymnastics needed.
    let rc_c = rope_cos.contiguous()?;
    let rs_c = rope_sin.contiguous()?;

    let dtype_int: i32 = match out_dtype {
        DType::F16  => 0,
        DType::BF16 => 1,
        DType::F32  => 2,
        d => candle::bail!("attn_post_qkv_decode: unsupported out dtype {:?}", d),
    };

    let total = n_q + 2 * n_kv;
    let total_q_elems = n_q * hd;
    let total_kv_elems = n_kv * hd;

    let _ = total;

    let stream_i64 = dev.cuda_stream().cu_stream() as i64;

    // Helper to read the f32 input tensor's pointer.
    let (qkv_storage, _) = qkv_c.storage_and_layout();
    let qkv_slice = match &*qkv_storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("qkv must be cuda"),
    };
    let qkv_ptr = qkv_slice.device_ptr(qkv_slice.stream()).0 as *const f32;

    let (qn_storage, _) = qn_c.storage_and_layout();
    let qn_slice = match &*qn_storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("q_norm_w must be cuda"),
    };
    let qn_ptr = qn_slice.device_ptr(qn_slice.stream()).0 as *const f32;

    let (kn_storage, _) = kn_c.storage_and_layout();
    let kn_slice = match &*kn_storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("k_norm_w must be cuda"),
    };
    let kn_ptr = kn_slice.device_ptr(kn_slice.stream()).0 as *const f32;

    let (rc_storage, _) = rc_c.storage_and_layout();
    let (rs_storage, _) = rs_c.storage_and_layout();
    let (rc_ptr, rs_ptr) = match out_dtype {
        DType::F16 => {
            let rc = match &*rc_storage {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f16>()?,
                _ => candle::bail!("rope_cos must be cuda"),
            };
            let rs = match &*rs_storage {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f16>()?,
                _ => candle::bail!("rope_sin must be cuda"),
            };
            (rc.device_ptr(rc.stream()).0 as *const c_void,
             rs.device_ptr(rs.stream()).0 as *const c_void)
        }
        DType::BF16 => {
            let rc = match &*rc_storage {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<bf16>()?,
                _ => candle::bail!("rope_cos must be cuda"),
            };
            let rs = match &*rs_storage {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<bf16>()?,
                _ => candle::bail!("rope_sin must be cuda"),
            };
            (rc.device_ptr(rc.stream()).0 as *const c_void,
             rs.device_ptr(rs.stream()).0 as *const c_void)
        }
        DType::F32 => {
            let rc = match &*rc_storage {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!("rope_cos must be cuda"),
            };
            let rs = match &*rs_storage {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!("rope_sin must be cuda"),
            };
            (rc.device_ptr(rc.stream()).0 as *const c_void,
             rs.device_ptr(rs.stream()).0 as *const c_void)
        }
        _ => unreachable!(),
    };

    use candle::op::BackpropOp;
    match out_dtype {
        DType::F16 => {
            let q_out = unsafe { dev.alloc::<f16>(total_q_elems) }?;
            let k_out = unsafe { dev.alloc::<f16>(total_kv_elems) }?;
            let v_out = unsafe { dev.alloc::<f16>(total_kv_elems) }?;
            let q_ptr = q_out.device_ptr(q_out.stream()).0 as *mut c_void;
            let k_ptr = k_out.device_ptr(k_out.stream()).0 as *mut c_void;
            let v_ptr = v_out.device_ptr(v_out.stream()).0 as *mut c_void;
            unsafe {
                ffi::attn_post_qkv_decode(
                    qkv_ptr, qn_ptr, kn_ptr, rc_ptr, rs_ptr,
                    q_ptr, k_ptr, v_ptr,
                    n_q as i32, n_kv as i32, hd as i32,
                    rope_pos as i32,
                    rms_eps, q_scale, dtype_int, rope_style, stream_i64,
                );
            }
            let q_storage = candle::CudaStorage::wrap_cuda_slice(q_out, dev.clone());
            let k_storage = candle::CudaStorage::wrap_cuda_slice(k_out, dev.clone());
            let v_storage = candle::CudaStorage::wrap_cuda_slice(v_out, dev.clone());
            let q_t = Tensor::from_storage(candle::Storage::Cuda(q_storage), (n_q, hd), BackpropOp::none(), false);
            let k_t = Tensor::from_storage(candle::Storage::Cuda(k_storage), (n_kv, hd), BackpropOp::none(), false);
            let v_t = Tensor::from_storage(candle::Storage::Cuda(v_storage), (n_kv, hd), BackpropOp::none(), false);
            Ok((q_t, k_t, v_t))
        }
        DType::BF16 => {
            let q_out = unsafe { dev.alloc::<bf16>(total_q_elems) }?;
            let k_out = unsafe { dev.alloc::<bf16>(total_kv_elems) }?;
            let v_out = unsafe { dev.alloc::<bf16>(total_kv_elems) }?;
            let q_ptr = q_out.device_ptr(q_out.stream()).0 as *mut c_void;
            let k_ptr = k_out.device_ptr(k_out.stream()).0 as *mut c_void;
            let v_ptr = v_out.device_ptr(v_out.stream()).0 as *mut c_void;
            unsafe {
                ffi::attn_post_qkv_decode(
                    qkv_ptr, qn_ptr, kn_ptr, rc_ptr, rs_ptr,
                    q_ptr, k_ptr, v_ptr,
                    n_q as i32, n_kv as i32, hd as i32,
                    rope_pos as i32,
                    rms_eps, q_scale, dtype_int, rope_style, stream_i64,
                );
            }
            let q_storage = candle::CudaStorage::wrap_cuda_slice(q_out, dev.clone());
            let k_storage = candle::CudaStorage::wrap_cuda_slice(k_out, dev.clone());
            let v_storage = candle::CudaStorage::wrap_cuda_slice(v_out, dev.clone());
            let q_t = Tensor::from_storage(candle::Storage::Cuda(q_storage), (n_q, hd), BackpropOp::none(), false);
            let k_t = Tensor::from_storage(candle::Storage::Cuda(k_storage), (n_kv, hd), BackpropOp::none(), false);
            let v_t = Tensor::from_storage(candle::Storage::Cuda(v_storage), (n_kv, hd), BackpropOp::none(), false);
            Ok((q_t, k_t, v_t))
        }
        DType::F32 => {
            let q_out = unsafe { dev.alloc::<f32>(total_q_elems) }?;
            let k_out = unsafe { dev.alloc::<f32>(total_kv_elems) }?;
            let v_out = unsafe { dev.alloc::<f32>(total_kv_elems) }?;
            let q_ptr = q_out.device_ptr(q_out.stream()).0 as *mut c_void;
            let k_ptr = k_out.device_ptr(k_out.stream()).0 as *mut c_void;
            let v_ptr = v_out.device_ptr(v_out.stream()).0 as *mut c_void;
            unsafe {
                ffi::attn_post_qkv_decode(
                    qkv_ptr, qn_ptr, kn_ptr, rc_ptr, rs_ptr,
                    q_ptr, k_ptr, v_ptr,
                    n_q as i32, n_kv as i32, hd as i32,
                    rope_pos as i32,
                    rms_eps, q_scale, dtype_int, rope_style, stream_i64,
                );
            }
            let q_storage = candle::CudaStorage::wrap_cuda_slice(q_out, dev.clone());
            let k_storage = candle::CudaStorage::wrap_cuda_slice(k_out, dev.clone());
            let v_storage = candle::CudaStorage::wrap_cuda_slice(v_out, dev.clone());
            let q_t = Tensor::from_storage(candle::Storage::Cuda(q_storage), (n_q, hd), BackpropOp::none(), false);
            let k_t = Tensor::from_storage(candle::Storage::Cuda(k_storage), (n_kv, hd), BackpropOp::none(), false);
            let v_t = Tensor::from_storage(candle::Storage::Cuda(v_storage), (n_kv, hd), BackpropOp::none(), false);
            Ok((q_t, k_t, v_t))
        }
        _ => unreachable!(),
    }
}

/// Same as `attn_post_qkv_decode` but Q is returned as F32 (K, V in
/// `out_dtype`). Saves a downstream `to_dtype(F32)` cast launch on Q
/// for the Q4 KV decode path.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn attn_post_qkv_decode_qf32(
    qkv: &Tensor,
    q_norm_w: &Tensor,
    k_norm_w: &Tensor,
    rope_cos: &Tensor,
    rope_sin: &Tensor,
    n_q: usize,
    n_kv: usize,
    hd: usize,
    rope_pos: usize,
    rms_eps: f32,
    q_scale: f32,
    out_dtype: candle::DType,
    rope_style: i32,
) -> Result<(Tensor, Tensor, Tensor)> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::DType;
    use core::ffi::c_void;
    use half::{bf16, f16};

    if qkv.dtype() != DType::F32 {
        candle::bail!("attn_post_qkv_decode_qf32: qkv must be F32");
    }
    if q_norm_w.dtype() != DType::F32 || k_norm_w.dtype() != DType::F32 {
        candle::bail!("attn_post_qkv_decode_qf32: norm weights must be F32");
    }
    if rope_cos.dtype() != out_dtype || rope_sin.dtype() != out_dtype {
        candle::bail!("attn_post_qkv_decode_qf32: rope dtype mismatch");
    }
    let dev = qkv.device().as_cuda_device()?;
    let qkv_c = qkv.contiguous()?;
    let qn_c  = q_norm_w.contiguous()?;
    let kn_c  = k_norm_w.contiguous()?;
    let rc_c = rope_cos.contiguous()?;
    let rs_c = rope_sin.contiguous()?;

    let dtype_int: i32 = match out_dtype {
        DType::F16  => 0,
        DType::BF16 => 1,
        DType::F32  => 2,
        d => candle::bail!("attn_post_qkv_decode_qf32: unsupported {:?}", d),
    };

    let total_q_elems = n_q * hd;
    let total_kv_elems = n_kv * hd;
    let stream_i64 = dev.cuda_stream().cu_stream() as i64;

    let (qkv_storage, _) = qkv_c.storage_and_layout();
    let qkv_slice = match &*qkv_storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("qkv must be cuda"),
    };
    let qkv_ptr = qkv_slice.device_ptr(qkv_slice.stream()).0 as *const f32;
    let (qn_storage, _) = qn_c.storage_and_layout();
    let qn_slice = match &*qn_storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("q_norm cuda"),
    };
    let qn_ptr = qn_slice.device_ptr(qn_slice.stream()).0 as *const f32;
    let (kn_storage, _) = kn_c.storage_and_layout();
    let kn_slice = match &*kn_storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("k_norm cuda"),
    };
    let kn_ptr = kn_slice.device_ptr(kn_slice.stream()).0 as *const f32;
    let (rc_storage, _) = rc_c.storage_and_layout();
    let (rs_storage, _) = rs_c.storage_and_layout();
    let (rc_ptr, rs_ptr) = match out_dtype {
        DType::F16 => {
            let rc = match &*rc_storage {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f16>()?,
                _ => candle::bail!("rope_cos cuda"),
            };
            let rs = match &*rs_storage {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f16>()?,
                _ => candle::bail!("rope_sin cuda"),
            };
            (rc.device_ptr(rc.stream()).0 as *const c_void,
             rs.device_ptr(rs.stream()).0 as *const c_void)
        }
        DType::BF16 => {
            let rc = match &*rc_storage {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<bf16>()?,
                _ => candle::bail!("rope_cos cuda"),
            };
            let rs = match &*rs_storage {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<bf16>()?,
                _ => candle::bail!("rope_sin cuda"),
            };
            (rc.device_ptr(rc.stream()).0 as *const c_void,
             rs.device_ptr(rs.stream()).0 as *const c_void)
        }
        DType::F32 => {
            let rc = match &*rc_storage {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!("rope_cos cuda"),
            };
            let rs = match &*rs_storage {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!("rope_sin cuda"),
            };
            (rc.device_ptr(rc.stream()).0 as *const c_void,
             rs.device_ptr(rs.stream()).0 as *const c_void)
        }
        _ => unreachable!(),
    };

    use candle::op::BackpropOp;
    let q_out = unsafe { dev.alloc::<f32>(total_q_elems) }?;
    let v_out = unsafe { dev.alloc::<f32>(total_kv_elems) }?;
    let q_ptr = q_out.device_ptr(q_out.stream()).0 as *mut f32;
    let v_ptr = v_out.device_ptr(v_out.stream()).0 as *mut f32;
    let result = match out_dtype {
        DType::F16 => {
            let k_out = unsafe { dev.alloc::<f16>(total_kv_elems) }?;
            let k_ptr = k_out.device_ptr(k_out.stream()).0 as *mut c_void;
            unsafe {
                ffi::attn_post_qkv_decode_qf32(
                    qkv_ptr, qn_ptr, kn_ptr, rc_ptr, rs_ptr,
                    q_ptr, k_ptr, v_ptr,
                    n_q as i32, n_kv as i32, hd as i32,
                    rope_pos as i32, rms_eps, q_scale, dtype_int, rope_style, stream_i64,
                );
            }
            let q_storage = candle::CudaStorage::wrap_cuda_slice(q_out, dev.clone());
            let k_storage = candle::CudaStorage::wrap_cuda_slice(k_out, dev.clone());
            let v_storage = candle::CudaStorage::wrap_cuda_slice(v_out, dev.clone());
            (
                Tensor::from_storage(candle::Storage::Cuda(q_storage), (n_q, hd), BackpropOp::none(), false),
                Tensor::from_storage(candle::Storage::Cuda(k_storage), (n_kv, hd), BackpropOp::none(), false),
                Tensor::from_storage(candle::Storage::Cuda(v_storage), (n_kv, hd), BackpropOp::none(), false),
            )
        }
        DType::BF16 => {
            let k_out = unsafe { dev.alloc::<bf16>(total_kv_elems) }?;
            let k_ptr = k_out.device_ptr(k_out.stream()).0 as *mut c_void;
            unsafe {
                ffi::attn_post_qkv_decode_qf32(
                    qkv_ptr, qn_ptr, kn_ptr, rc_ptr, rs_ptr,
                    q_ptr, k_ptr, v_ptr,
                    n_q as i32, n_kv as i32, hd as i32,
                    rope_pos as i32, rms_eps, q_scale, dtype_int, rope_style, stream_i64,
                );
            }
            let q_storage = candle::CudaStorage::wrap_cuda_slice(q_out, dev.clone());
            let k_storage = candle::CudaStorage::wrap_cuda_slice(k_out, dev.clone());
            let v_storage = candle::CudaStorage::wrap_cuda_slice(v_out, dev.clone());
            (
                Tensor::from_storage(candle::Storage::Cuda(q_storage), (n_q, hd), BackpropOp::none(), false),
                Tensor::from_storage(candle::Storage::Cuda(k_storage), (n_kv, hd), BackpropOp::none(), false),
                Tensor::from_storage(candle::Storage::Cuda(v_storage), (n_kv, hd), BackpropOp::none(), false),
            )
        }
        DType::F32 => {
            let k_out = unsafe { dev.alloc::<f32>(total_kv_elems) }?;
            let k_ptr = k_out.device_ptr(k_out.stream()).0 as *mut c_void;
            unsafe {
                ffi::attn_post_qkv_decode_qf32(
                    qkv_ptr, qn_ptr, kn_ptr, rc_ptr, rs_ptr,
                    q_ptr, k_ptr, v_ptr,
                    n_q as i32, n_kv as i32, hd as i32,
                    rope_pos as i32, rms_eps, q_scale, dtype_int, rope_style, stream_i64,
                );
            }
            let q_storage = candle::CudaStorage::wrap_cuda_slice(q_out, dev.clone());
            let k_storage = candle::CudaStorage::wrap_cuda_slice(k_out, dev.clone());
            let v_storage = candle::CudaStorage::wrap_cuda_slice(v_out, dev.clone());
            (
                Tensor::from_storage(candle::Storage::Cuda(q_storage), (n_q, hd), BackpropOp::none(), false),
                Tensor::from_storage(candle::Storage::Cuda(k_storage), (n_kv, hd), BackpropOp::none(), false),
                Tensor::from_storage(candle::Storage::Cuda(v_storage), (n_kv, hd), BackpropOp::none(), false),
            )
        }
        _ => unreachable!(),
    };
    Ok(result)
}

#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn attn_post_qkv_decode_qf32(
    _: &Tensor, _: &Tensor, _: &Tensor, _: &Tensor, _: &Tensor,
    _: usize, _: usize, _: usize, _: usize,
    _: f32, _: f32, _: candle::DType, _: i32,
) -> Result<(Tensor, Tensor, Tensor)> {
    candle::bail!("attn_post_qkv_decode_qf32 cuda only")
}

/// No-norm variant of `attn_post_qkv_decode_qf32`: just RoPE + scale +
/// dtype cast, skipping the q_norm/k_norm step. For models like
/// qwen2/llama/mistral that don't apply RMS norm to Q/K. Q & V are
/// emitted in F32; K stays in `out_dtype`.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn attn_post_qkv_decode_qf32_no_norm(
    qkv: &Tensor,
    rope_cos: &Tensor,
    rope_sin: &Tensor,
    n_q: usize,
    n_kv: usize,
    hd: usize,
    rope_pos: usize,
    q_scale: f32,
    out_dtype: candle::DType,
    rope_style: i32,
) -> Result<(Tensor, Tensor, Tensor)> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::DType;
    use core::ffi::c_void;
    use half::{bf16, f16};

    if qkv.dtype() != DType::F32 {
        candle::bail!("attn_post_qkv_decode_qf32_no_norm: qkv must be F32");
    }
    if rope_cos.dtype() != out_dtype || rope_sin.dtype() != out_dtype {
        candle::bail!("attn_post_qkv_decode_qf32_no_norm: rope dtype mismatch");
    }
    let dtype_int: i32 = match out_dtype {
        DType::F16  => 0,
        DType::BF16 => 1,
        DType::F32  => 2,
        d => candle::bail!("attn_post_qkv_decode_qf32_no_norm: unsupported {:?}", d),
    };
    let dev = qkv.device().as_cuda_device()?;
    let qkv_c = qkv.contiguous()?;
    let rc_c = rope_cos.contiguous()?;
    let rs_c = rope_sin.contiguous()?;

    let total_q_elems = n_q * hd;
    let total_kv_elems = n_kv * hd;
    let stream_i64 = dev.cuda_stream().cu_stream() as i64;

    let (qkv_storage, _) = qkv_c.storage_and_layout();
    let qkv_slice = match &*qkv_storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("qkv must be cuda"),
    };
    let qkv_ptr = qkv_slice.device_ptr(qkv_slice.stream()).0 as *const f32;
    let (rc_storage, _) = rc_c.storage_and_layout();
    let (rs_storage, _) = rs_c.storage_and_layout();
    let (rc_ptr, rs_ptr) = match out_dtype {
        DType::F16 => {
            let rc = match &*rc_storage {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f16>()?,
                _ => candle::bail!("rope_cos cuda"),
            };
            let rs = match &*rs_storage {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f16>()?,
                _ => candle::bail!("rope_sin cuda"),
            };
            (rc.device_ptr(rc.stream()).0 as *const c_void,
             rs.device_ptr(rs.stream()).0 as *const c_void)
        }
        DType::BF16 => {
            let rc = match &*rc_storage {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<bf16>()?,
                _ => candle::bail!("rope_cos cuda"),
            };
            let rs = match &*rs_storage {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<bf16>()?,
                _ => candle::bail!("rope_sin cuda"),
            };
            (rc.device_ptr(rc.stream()).0 as *const c_void,
             rs.device_ptr(rs.stream()).0 as *const c_void)
        }
        DType::F32 => {
            let rc = match &*rc_storage {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!("rope_cos cuda"),
            };
            let rs = match &*rs_storage {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!("rope_sin cuda"),
            };
            (rc.device_ptr(rc.stream()).0 as *const c_void,
             rs.device_ptr(rs.stream()).0 as *const c_void)
        }
        _ => unreachable!(),
    };

    use candle::op::BackpropOp;
    let q_out = unsafe { dev.alloc::<f32>(total_q_elems) }?;
    let v_out = unsafe { dev.alloc::<f32>(total_kv_elems) }?;
    let q_ptr = q_out.device_ptr(q_out.stream()).0 as *mut f32;
    let v_ptr = v_out.device_ptr(v_out.stream()).0 as *mut f32;
    let result = match out_dtype {
        DType::F16 => {
            let k_out = unsafe { dev.alloc::<f16>(total_kv_elems) }?;
            let k_ptr = k_out.device_ptr(k_out.stream()).0 as *mut c_void;
            unsafe {
                ffi::attn_post_qkv_decode_qf32_no_norm(
                    qkv_ptr, rc_ptr, rs_ptr,
                    q_ptr, k_ptr, v_ptr,
                    n_q as i32, n_kv as i32, hd as i32,
                    rope_pos as i32, q_scale, dtype_int, rope_style, stream_i64,
                );
            }
            let q_storage = candle::CudaStorage::wrap_cuda_slice(q_out, dev.clone());
            let k_storage = candle::CudaStorage::wrap_cuda_slice(k_out, dev.clone());
            let v_storage = candle::CudaStorage::wrap_cuda_slice(v_out, dev.clone());
            (
                Tensor::from_storage(candle::Storage::Cuda(q_storage), (n_q, hd), BackpropOp::none(), false),
                Tensor::from_storage(candle::Storage::Cuda(k_storage), (n_kv, hd), BackpropOp::none(), false),
                Tensor::from_storage(candle::Storage::Cuda(v_storage), (n_kv, hd), BackpropOp::none(), false),
            )
        }
        DType::BF16 => {
            let k_out = unsafe { dev.alloc::<bf16>(total_kv_elems) }?;
            let k_ptr = k_out.device_ptr(k_out.stream()).0 as *mut c_void;
            unsafe {
                ffi::attn_post_qkv_decode_qf32_no_norm(
                    qkv_ptr, rc_ptr, rs_ptr,
                    q_ptr, k_ptr, v_ptr,
                    n_q as i32, n_kv as i32, hd as i32,
                    rope_pos as i32, q_scale, dtype_int, rope_style, stream_i64,
                );
            }
            let q_storage = candle::CudaStorage::wrap_cuda_slice(q_out, dev.clone());
            let k_storage = candle::CudaStorage::wrap_cuda_slice(k_out, dev.clone());
            let v_storage = candle::CudaStorage::wrap_cuda_slice(v_out, dev.clone());
            (
                Tensor::from_storage(candle::Storage::Cuda(q_storage), (n_q, hd), BackpropOp::none(), false),
                Tensor::from_storage(candle::Storage::Cuda(k_storage), (n_kv, hd), BackpropOp::none(), false),
                Tensor::from_storage(candle::Storage::Cuda(v_storage), (n_kv, hd), BackpropOp::none(), false),
            )
        }
        DType::F32 => {
            let k_out = unsafe { dev.alloc::<f32>(total_kv_elems) }?;
            let k_ptr = k_out.device_ptr(k_out.stream()).0 as *mut c_void;
            unsafe {
                ffi::attn_post_qkv_decode_qf32_no_norm(
                    qkv_ptr, rc_ptr, rs_ptr,
                    q_ptr, k_ptr, v_ptr,
                    n_q as i32, n_kv as i32, hd as i32,
                    rope_pos as i32, q_scale, dtype_int, rope_style, stream_i64,
                );
            }
            let q_storage = candle::CudaStorage::wrap_cuda_slice(q_out, dev.clone());
            let k_storage = candle::CudaStorage::wrap_cuda_slice(k_out, dev.clone());
            let v_storage = candle::CudaStorage::wrap_cuda_slice(v_out, dev.clone());
            (
                Tensor::from_storage(candle::Storage::Cuda(q_storage), (n_q, hd), BackpropOp::none(), false),
                Tensor::from_storage(candle::Storage::Cuda(k_storage), (n_kv, hd), BackpropOp::none(), false),
                Tensor::from_storage(candle::Storage::Cuda(v_storage), (n_kv, hd), BackpropOp::none(), false),
            )
        }
        _ => unreachable!(),
    };
    Ok(result)
}

#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn attn_post_qkv_decode_qf32_no_norm(
    _: &Tensor, _: &Tensor, _: &Tensor,
    _: usize, _: usize, _: usize, _: usize,
    _: f32, _: candle::DType, _: i32,
) -> Result<(Tensor, Tensor, Tensor)> {
    candle::bail!("attn_post_qkv_decode_qf32_no_norm cuda only")
}

#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn attn_post_qkv_decode(
    _: &Tensor, _: &Tensor, _: &Tensor, _: &Tensor, _: &Tensor,
    _: usize, _: usize, _: usize, _: usize,
    _: f32, _: f32, _: candle::DType, _: i32,
) -> Result<(Tensor, Tensor, Tensor)> {
    candle::bail!("attn_post_qkv_decode is only implemented for the cuda backend")
}
