use core::ffi::c_void;
#[allow(dead_code)]
#[allow(improper_ctypes)]
extern "C" {
    // for unquntized models
    pub fn moe_gemm_wmma(
        input: *const c_void,         // device pointer [size_m, size_k]
        weights: *const c_void,       // device pointer [num_experts, size_n, size_k]
        sorted_token_ids: *const i32, // device pointer [size_m]
        expert_ids: *const i32,       // host array [size_m] (expert id per sorted token)
        topk_weights: *const f32,
        output: *mut c_void,      // device pointer [size_m, size_n]
        expert_counts: *mut i32,  // pre-allocated buffer [num_experts]
        expert_offsets: *mut i32, // pre-allocated buffer [num_experts + 1]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        dtype: i32, // 0=float16, 1=bf16 (for input/output)
        is_prefill: bool,
        stream: i64,
    );

    pub fn moe_gemm_gguf(
        input: *const f32,      // input [size_m, size_k]
        weights: *const c_void, // weights [num_experts, size_n, size_k]
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32, // device ptr or nullptr
        output: *mut c_void,      // float output [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        gguf_dtype: i32, // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5  (for weights)
        stream: i64,
    );

    /// Fused gate+up MoE GEMM with SiLU activation and elementwise
    /// multiply. Shares the input quantize and the per-block input load
    /// across both gate and up dot products, then writes a single
    /// `silu(gate) * up` value per output position. Replaces the
    /// two-call `moe_gemm_gguf(gate)` + `moe_gemm_gguf(up)` +
    /// elementwise-silu-and-mul sequence with a single launch.
    pub fn moe_gemm_gguf_gate_up_silu_mul(
        input: *const f32,
        gate_weights: *const c_void,
        up_weights: *const c_void,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        output: *mut c_void, // float output [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        gguf_dtype: i32,
        stream: i64,
    );

    /// Same as moe_gemm_gguf_gate_up_silu_mul but with GELU(tanh-approx)
    /// activation and a CONCATENATED gate||up weight tensor of shape
    /// `[num_experts, 2*size_n, size_k]` (gate = first N rows, up =
    /// next N rows). Used by the gemma4-MoE FFN where the GGUF stores
    /// gate and up fused as `ffn_gate_up_exps.weight`.
    pub fn moe_gemm_gguf_gate_up_gelu_mul_concat(
        input: *const f32,
        gate_up_weights: *const c_void,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        output: *mut c_void, // float output [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,        // expert ffn dim (half of stored 2N)
        size_k: i32,
        gguf_dtype: i32,
        stream: i64,
    );

    /// Fused MoE down-projection + topk reduction. Each (token, expert)
    /// row's weighted partial result is added into the [num_real_tokens,
    /// hidden] output via atomicAdd. Caller must pre-zero the output.
    /// Saves the explicit reshape+sum that the unfused down + topk
    /// reduction sequence requires.
    pub fn moe_gemm_gguf_down_reduce(
        input: *const f32,
        weights: *const c_void,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32,
        output: *mut f32,         // [num_real_tokens, size_n] pre-zeroed
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        gguf_dtype: i32,
        stream: i64,
    );

    /// Phase 1 building block: device-side expert offsets from a sorted
    /// `expert_ids` array. Emits `expert_offsets[num_experts + 1]` such
    /// that `expert_offsets[e] = first index where expert_ids[i] >= e`
    /// (and `expert_offsets[num_experts] = M`). Per-expert pair counts
    /// are `expert_offsets[e+1] - expert_offsets[e]`.
    ///
    /// O(num_experts × log M) work; one block per expert, single-thread
    /// binary search. Used to unblock per-expert dispatch in
    /// `moe_gemm_gguf_*` for prefill batches where the current
    /// per-(token,expert) kernel emits an O(num_tokens × topk) grid.
    pub fn moe_expert_offsets(
        sorted_expert_ids: *const i32,
        expert_offsets: *mut i32,        // [num_experts + 1]
        m: i32,
        num_experts: i32,
        stream: i64,
    );

    /// Phase 1 step-3: gather input rows for per-expert dispatch.
    /// Writes a contiguous [n_e, K] F16 buffer where row i is
    /// `inputs[sorted_token_ids[start+i] / topk]`. One CUDA block per
    /// output row, one thread per K element (strided).
    pub fn moe_gather_input_rows_f32_to_f16(
        inputs: *const f32,
        sorted_token_ids: *const i32,
        out_f16: *mut core::ffi::c_void,
        n_e: i32,
        start: i32,
        k: i32,
        topk: i32,
        stream: i64,
    );

    /// Phase 1 step-3: GELU·mul + scatter for per-expert dispatch.
    /// Input is the cuBLAS GEMM output `[n_e, 2N]` F16; for each row,
    /// applies `gelu_tanh(gate) * up` and scatters the resulting
    /// `[N]` row into the final F32 output at index
    /// `sorted_token_ids[start+i]`.
    pub fn moe_gelu_mul_scatter_f16_to_f32(
        in_f16: *const core::ffi::c_void,
        sorted_token_ids: *const i32,
        out_f32: *mut f32,
        n_e: i32,
        start: i32,
        n: i32,
        stream: i64,
    );

    /// Phase 1 step-4 batched gather: writes a padded
    /// `[N_active, max_n_e, K]` F16 workspace from F32 inputs across
    /// ALL active experts in ONE launch. Each (act_idx, row) reads
    /// `sorted_token_ids[expert_offsets[active_expert_ids[act_idx]] + row]`,
    /// divides by topk, and copies the input row in. Padding rows
    /// (row ≥ n_e[act_idx]) are skipped — caller must pre-zero out.
    pub fn moe_batched_gather_input_rows_f32_to_f16(
        inputs: *const f32,
        sorted_token_ids: *const i32,
        active_expert_ids: *const i32,
        expert_offsets: *const i32,
        out_f16: *mut core::ffi::c_void,
        n_active: i32,
        max_n_e: i32,
        k: i32,
        topk: i32,
        stream: i64,
    );

    /// Phase 1 step-5 (Q4_K MMA path): per-expert batched
    /// `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` GEMM for
    /// gate||up. Consumes Q4_K weights (no F16 dequant intermediate) and
    /// pre-quantized Q8_1 inputs. Output is the gate||up logits F16 slab
    /// `[N_active, max_n_e, 2N]`, ready for the existing batched
    /// `moe_batched_gelu_mul_scatter_f16_to_f32` kernel.
    ///
    /// Per block (one warp): computes a [16 weight rows × 8 input rows]
    /// output tile using one m16n8k32 INT8 MMA per K-tile (32 elems).
    /// Total compute per block is 16 × 8 × K = 128K dot products done
    /// via tensor cores instead of dp4a.
    pub fn moe_q4k_mma_batched_gate_up(
        gate_up_w: *const core::ffi::c_void,
        inputs_q81: *const core::ffi::c_void,
        active_expert_ids: *const i32,
        expert_offsets: *const i32,
        dst_f16: *mut core::ffi::c_void,
        num_experts: i32,
        n_active: i32,
        max_n_e: i32,
        two_n: i32,
        k: i32,
        stream: i64,
    );

    /// Chunked IMMA M=8: per-block (pair_start, expert) metadata pre-
    /// computed (host or device) so each chunk has a SINGLE expert.
    /// Reads num_chunks_dev[0] from device — caller can launch with
    /// max_num_chunks (worst case) and inactive blocks will skip.
    pub fn moe_q4k_imma_m8_chunks_gate_up(
        gate_up_w: *const core::ffi::c_void,
        inputs_q81: *const core::ffi::c_void,
        sorted_token_ids: *const i32,
        chunk_pair_start: *const i32,
        chunk_expert: *const i32,
        num_chunks_dev: *const i32,
        dst_f32: *mut f32,
        num_experts: i32,
        topk: i32,
        max_num_chunks: i32,
        size_m: i32,
        n: i32,
        k: i32,
        stream: i64,
    );

    /// Device-side chunk builder for the chunked IMMA M=8 path. Walks
    /// expert_ids and emits per-chunk metadata WITHOUT a D2H sync.
    /// Single-thread kernel — chunk count is small, linear scan beats
    /// parallel-prefix overhead.
    pub fn moe_q4k_imma_m8_build_chunks(
        expert_ids: *const i32,
        chunk_pair_start: *mut i32,
        chunk_expert: *mut i32,
        num_chunks_out: *mut i32,
        size_m: i32,
        stream: i64,
    );

    /// 4-warp IMMA M=8 (independent warps, different m_tile each).
    /// Block produces 64 weight rows × 8 pairs = 512 outputs.
    pub fn moe_q4k_imma_m8_m64_gate_up(
        gate_up_w: *const core::ffi::c_void,
        inputs_q81: *const core::ffi::c_void,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        dst_f32: *mut f32,
        num_experts: i32,
        topk: i32,
        size_m: i32,
        n: i32,
        k: i32,
        stream: i64,
    );

    /// DENSE Q4_K IMMA M=8 plain matmul (no activation, no fusion).
    /// Use case: attention QKV proj, out_proj, embedding output proj,
    /// or any Q4_K matmul on multi-row F32 input.
    pub fn dense_q4k_imma_m8_matmul(
        w: *const core::ffi::c_void,
        inputs_q81: *const core::ffi::c_void,
        dst_f32: *mut f32,
        size_m: i32,
        n: i32,
        k: i32,
        stream: i64,
    );

    /// DENSE Q4_K IMMA M=8 silu*mul. Separate gate and up Q4_K weights;
    /// each block does 16 rows × 8 tokens via mma m16n8k32. SiLU(gate)
    /// * up activation, F32 output [size_m, N].
    pub fn dense_q4k_imma_m8_silu(
        gate_w: *const core::ffi::c_void,
        up_w: *const core::ffi::c_void,
        inputs_q81: *const core::ffi::c_void,
        dst_f32: *mut f32,
        size_m: i32,
        n: i32,
        k: i32,
        stream: i64,
    );

    /// IMMA M=8 with 2 warps per block, each warp handling a different
    /// m_tile (16 weight rows). Block produces 32 weight rows × 8 pairs.
    /// Independent warps — no sync, no shared mem. Reduces grid-X by 2×.
    pub fn moe_q4k_imma_m8_2w_gate_up(
        gate_up_w: *const core::ffi::c_void,
        inputs_q81: *const core::ffi::c_void,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        dst_f32: *mut f32,
        num_experts: i32,
        topk: i32,
        size_m: i32,
        n: i32,
        k: i32,
        stream: i64,
    );

    /// IMMA M=8 with M_TILE=32: 2 mma ops per K-tile per warp, producing
    /// 32 weight rows × 8 pairs per block (256 outputs vs 128 in the
    /// base M=16 variant).
    pub fn moe_q4k_imma_m8_m32_gate_up(
        gate_up_w: *const core::ffi::c_void,
        inputs_q81: *const core::ffi::c_void,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        dst_f32: *mut f32,
        num_experts: i32,
        topk: i32,
        size_m: i32,
        n: i32,
        k: i32,
        stream: i64,
    );

    /// Per-pair-tile IMMA M=8 for the MoE down step. Same pattern as
    /// the gate||up IMMA M=8 but with atomicAdd output scaled by
    /// topk_weights[pair].
    pub fn moe_q4k_imma_m8_down(
        down_w: *const core::ffi::c_void,
        inputs_q81: *const core::ffi::c_void,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32,
        dst_f32: *mut f32,
        num_experts: i32,
        topk: i32,
        size_m: i32,
        hidden: i32,
        k: i32,
        stream: i64,
    );

    /// Multi-warp IMMA M=8: 4 warps per block, each warp handles a
    /// different 16-row weight slice but they share the SAME 8 sorted
    /// pairs. Block produces 64 weight rows × 8 pairs = 512 outputs.
    pub fn moe_q4k_imma_m8_mw_gate_up(
        gate_up_w: *const core::ffi::c_void,
        inputs_q81: *const core::ffi::c_void,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        dst_f32: *mut f32,
        num_experts: i32,
        topk: i32,
        size_m: i32,
        n: i32,
        k: i32,
        stream: i64,
    );

    /// Per-pair-tile IMMA M=8: extends the M=1 broadcast IMMA to M=8 by
    /// packing 8 consecutive sorted pairs into one block. When all 8
    /// pairs share an expert (typical, since pairs are sorted by expert),
    /// one mma.sync m16n8k32 covers all 8 output columns vs 8 separate
    /// mma ops in the broadcast variant. Boundary blocks where pairs
    /// span experts mask out the wrong-expert lanes per-output.
    pub fn moe_q4k_imma_m8_gate_up(
        gate_up_w: *const core::ffi::c_void,
        inputs_q81: *const core::ffi::c_void,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        dst_f32: *mut f32,
        num_experts: i32,
        topk: i32,
        size_m: i32,
        n: i32,
        k: i32,
        stream: i64,
    );

    /// K-split MMQ: 4 warps share the SAME 16×8 output tile but split
    /// the K-loop work 4 ways. Reduces per-warp sequential work 4× →
    /// more concurrent warps for the SM scheduler.
    pub fn moe_q4k_mmq_splitk_gate_up(
        gate_up_w: *const core::ffi::c_void,
        inputs_q81: *const core::ffi::c_void,
        active_expert_ids: *const i32,
        expert_offsets: *const i32,
        dst_f32: *mut core::ffi::c_void,
        num_experts: i32,
        n_active: i32,
        max_n_e: i32,
        two_n: i32,
        k: i32,
        stream: i64,
    );

    /// Multi-warp Q4_K MMQ kernel modeled on llama.cpp mmq.cu. 4 warps
    /// per block (128 threads) share the same 16 weight rows via
    /// shared-memory cooperative load. Each warp owns 8 input rows;
    /// block produces 16 × 32 output tile. Q4_K HBM weight reads are
    /// shared 4× vs the single-warp moe_q4k_mma_batched_gate_up.
    /// Output: F32 `[N_active, max_n_e, 2N]` (same as MMA non-fused),
    /// reuses the existing GELU·mul scatter.
    pub fn moe_q4k_mmq_gate_up(
        gate_up_w: *const core::ffi::c_void,
        inputs_q81: *const core::ffi::c_void,
        active_expert_ids: *const i32,
        expert_offsets: *const i32,
        dst_f32: *mut core::ffi::c_void,
        num_experts: i32,
        n_active: i32,
        max_n_e: i32,
        two_n: i32,
        k: i32,
        stream: i64,
    );

    /// Phase 1 step-5 FUSED variant: gate||up MMA + GELU·mul + scatter
    /// in a single launch. Eliminates the F32 `[N_active, max_n_e, 2N]`
    /// intermediate buffer that the unfused variant materialises in HBM,
    /// removing the round-trip cost that dominates prefill latency.
    /// Output is F32 `[size_m, N]` (final MoE branch output before the
    /// down projection).
    pub fn moe_q4k_mma_batched_gate_up_gelu_mul_scatter(
        gate_up_w: *const core::ffi::c_void,
        inputs_q81: *const core::ffi::c_void,
        sorted_token_ids: *const i32,
        active_expert_ids: *const i32,
        expert_offsets: *const i32,
        dst_f32: *mut f32,
        num_experts: i32,
        n_active: i32,
        max_n_e: i32,
        n_gate: i32,
        k: i32,
        stream: i64,
    );

    /// Phase 1 step-5 (Q4_K MMA path): batched F32 → Q8_1 gather+quantize.
    /// Same dispatch contract as the F32→F16 gather, but writes Q8_1 blocks
    /// (`[N_active, max_n_e, K/32]` block_q8_1). Q8_1 is the input format
    /// consumed by the `mma.sync.m16n8k32.s8.s8.s32` Q4_K MMA kernel.
    pub fn moe_batched_gather_input_rows_f32_to_q81(
        inputs: *const f32,
        sorted_token_ids: *const i32,
        active_expert_ids: *const i32,
        expert_offsets: *const i32,
        out_q81: *mut core::ffi::c_void,
        n_active: i32,
        max_n_e: i32,
        k: i32,
        topk: i32,
        stream: i64,
    );

    /// Phase 1 step-4 batched scatter: reads the padded GEMM output
    /// `[N_active, max_n_e, 2N]` F16, applies `gelu_tanh(gate)·up` per
    /// valid row, and scatters into `out[sorted_token_ids[...], :]`
    /// F32 — all active experts in ONE launch.
    pub fn moe_batched_gelu_mul_scatter_f16_to_f32(
        in_f16: *const core::ffi::c_void,
        sorted_token_ids: *const i32,
        active_expert_ids: *const i32,
        expert_offsets: *const i32,
        out_f32: *mut f32,
        n_active: i32,
        max_n_e: i32,
        n: i32,
        stream: i64,
    );

    /// F32 input variant of the GELU·mul + scatter — pairs with the
    /// Q4_K MMA kernel that writes F32 directly (no F16 rounding step,
    /// which compounds drift over 30+ cascading layers in real models).
    pub fn moe_batched_gelu_mul_scatter_f32_to_f32(
        in_f32: *const f32,
        sorted_token_ids: *const i32,
        active_expert_ids: *const i32,
        expert_offsets: *const i32,
        out_f32: *mut f32,
        n_active: i32,
        max_n_e: i32,
        n: i32,
        stream: i64,
    );

    /// Phase 1 step-4 batched dequant: dequantizes N_active experts'
    /// `[rows_per_expert, cols]` Q4_K weight slabs into a contiguous
    /// `[n_active, rows_per_expert, cols]` F16 workspace in ONE launch.
    /// Replaces N_active separate dequant launches and the per-call
    /// scratch-buffer copies the host-loop variant required.
    /// `active_expert_ids` is a host-supplied i32 array indicating which
    /// experts to materialise (e.g. the sparse list of experts with at
    /// least one token assigned).
    pub fn moe_batched_dequant_q4k_f16(
        all_weights: *const core::ffi::c_void,
        active_expert_ids: *const i32,
        out_f16: *mut core::ffi::c_void,
        n_active: i32,
        rows_per_expert: i32,
        cols: i32,
        stream: i64,
    );

    /// Fused quantized matmul + residual add. `dst` must be PRE-INITIALIZED
    /// with the residual (e.g. via cuMemcpyDtoDAsync of an F32 tensor).
    /// Kernel atomicAdds matmul partial sums into `dst`. Output =
    /// matmul(W, x) + residual. Saves the post-matmul broadcast_add.
    pub fn qmatmul_add(
        x: *const f32,              // [hidden]
        w_q: *const c_void,         // [out_rows, hidden] quantized
        y_q8_1_scratch: *mut c_void, // q8_1 scratch (caller-allocated)
        dst: *mut f32,              // [out_rows] pre-init with residual
        hidden: i32,
        out_rows: i32,
        quant_type: i32,
        stream: i64,
    );

    /// Fused RMS norm + Q*_K matmul for single-token decode. Replaces
    /// the rms_norm + quantize_q8_1 + mul_mat_vec sequence with one
    /// launch — saves both the explicit norm launch and the q8_1
    /// quantize launch (the latter is internal to candle's QMatMul,
    /// invisible at the API level but real overhead).
    pub fn rms_qmatmul(
        x: *const f32,              // [hidden]
        w_norm: *const f32,         // [hidden]
        w_mm: *const c_void,        // [out_rows, hidden] quantized
        out: *mut f32,              // [out_rows]
        hidden: i32,
        out_rows: i32,
        rms_eps: f32,
        quant_type: i32,
        stream: i64,
    );

    /// Cast and copy: dst[i] = (f32) src[i] for n elements. Used to
    /// initialize the moe_gemm_gguf_down_reduce output buffer with the
    /// residual values so the post-MLP residual add is folded into the
    /// final atomicAdd accumulation.
    pub fn cast_init_f32_from_dtype(
        dst: *mut f32,
        src: *const c_void,
        n: i32,
        dtype: i32,                 // 0=f16, 1=bf16
        stream: i64,
    );

    /// Fused MoE routing: softmax → top-k → optional renorm in a single
    /// CUDA launch. Replaces ~6 candle ops (softmax_last_dim,
    /// arg_sort_last_dim, narrow, contiguous, gather, sum_keepdim,
    /// broadcast_div) per MoE layer.
    pub fn topk_softmax(
        logits: *const f32,           // [n_rows, n_experts]
        weights: *mut f32,            // [n_rows, n_expert_used]
        ids: *mut u32,                // [n_rows, n_expert_used]
        n_rows: i32,
        n_experts: i32,
        n_expert_used: i32,
        with_norm: i32,
        stream: i64,
    );

    /// Fused post-QKV attention prep for single-token decode.
    /// Replaces q_norm + k_norm + 3× to_dtype + RoPE (cos/sin cast,
    /// q.contiguous, k.contiguous, rope_q, rope_k) — typically 5–9
    /// candle launches per layer — with a single CUDA launch.
    pub fn attn_post_qkv_decode(
        qkv: *const f32,            // [n_q*hd + 2*n_kv*hd]
        q_norm_w: *const f32,       // [hd]
        k_norm_w: *const f32,       // [hd]
        rope_cos: *const c_void,    // [max_seq, hd/2] — FULL table
        rope_sin: *const c_void,    // [max_seq, hd/2]
        q_out: *mut c_void,         // [n_q, hd]
        k_out: *mut c_void,         // [n_kv, hd]
        v_out: *mut c_void,         // [n_kv, hd]
        n_q: i32,
        n_kv: i32,
        hd: i32,
        rope_pos: i32,
        rms_eps: f32,
        q_scale: f32,               // multiply Q by this — folds 1/sqrt(d)
        dtype: i32,
        rope_style: i32,
        stream: i64,
    );

    /// Same as `attn_post_qkv_decode` but Q is written in F32 (K/V stay
    /// in `dtype`). Used by the Q4 KV decode path to skip the
    /// to_dtype(F32) cast on Q before the downstream score kernel.
    pub fn attn_post_qkv_decode_qf32(
        qkv: *const f32,
        q_norm_w: *const f32,
        k_norm_w: *const f32,
        rope_cos: *const c_void,
        rope_sin: *const c_void,
        q_out: *mut f32,
        k_out: *mut c_void,
        v_out: *mut f32,
        n_q: i32,
        n_kv: i32,
        hd: i32,
        rope_pos: i32,
        rms_eps: f32,
        q_scale: f32,
        dtype: i32,
        rope_style: i32,
        stream: i64,
    );

    /// No-norm variant of `attn_post_qkv_decode_qf32` for qwen2/llama/
    /// mistral (no q_norm/k_norm). RoPE + scale + cast only.
    pub fn attn_post_qkv_decode_qf32_no_norm(
        qkv: *const f32,
        rope_cos: *const c_void,
        rope_sin: *const c_void,
        q_out: *mut f32,
        k_out: *mut c_void,
        v_out: *mut f32,
        n_q: i32,
        n_kv: i32,
        hd: i32,
        rope_pos: i32,
        q_scale: f32,
        dtype: i32,
        rope_style: i32,
        stream: i64,
    );

    pub fn moe_gemm_gguf_prefill(
        input: *const c_void, // input [size_m, size_k]
        weights: *const u8,   // weights [num_experts, size_n, size_k]
        sorted_token_ids: *const i32,
        expert_ids: *const i32,   //must be host ptr
        topk_weights: *const f32, // device ptr or nullptr
        output: *mut c_void,      // float output [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        input_dtype: i32, // 0=f16, 1=bf16 (for inputs)
        gguf_dtype: i32,  //Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5  (for weights)
        stream: i64,
    );

    // ============== Dense GGUF MMVQ launchers (from mmvq_gguf.cu) ==============

    // BF16 output launchers
    pub fn launch_mmvq_gguf_q4_0_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q4_1_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_0_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_1_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q8_0_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q2_k_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q3_k_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q4_k_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_k_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q6_k_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );

    // F32 output launchers
    pub fn launch_mmvq_gguf_q4_0_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q4_1_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_0_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_1_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q8_0_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q2_k_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q3_k_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q4_k_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_k_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q6_k_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );

    pub fn launch_mmvq_gguf_q4_0_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q4_1_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_0_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_1_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q8_0_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q2_k_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q3_k_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q4_k_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_k_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q6_k_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );

    // Quantize launchers (activation → Q8_1)
    pub fn launch_mmvq_gguf_quantize_q8_1_bf16(
        x: *const c_void,
        vy: *mut c_void,
        kx: i32,
        kx_padded: i32,
        num_rows: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_quantize_q8_1_f16(
        x: *const c_void,
        vy: *mut c_void,
        kx: i32,
        kx_padded: i32,
        num_rows: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_quantize_q8_1_f32(
        x: *const c_void,
        vy: *mut c_void,
        kx: i32,
        kx_padded: i32,
        num_rows: i32,
        stream: *mut c_void,
    );

    // ============== Dense GGUF MMQ launchers (from mmq_gguf/) ==============

    // MMQ quantize launchers (f32 -> block_q8_1_mmq, 3 scale layouts)
    pub fn launch_mmq_quantize_q8_1_D4(
        x: *const c_void,
        ids: *const i32,
        vy: *mut c_void,
        type_x: i32,
        ne00: i64,
        s01: i64,
        s02: i64,
        s03: i64,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
        stream: *mut c_void,
    );
    pub fn launch_mmq_quantize_q8_1_DS4(
        x: *const c_void,
        ids: *const i32,
        vy: *mut c_void,
        type_x: i32,
        ne00: i64,
        s01: i64,
        s02: i64,
        s03: i64,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
        stream: *mut c_void,
    );
    pub fn launch_mmq_quantize_q8_1_D2S6(
        x: *const c_void,
        ids: *const i32,
        vy: *mut c_void,
        type_x: i32,
        ne00: i64,
        s01: i64,
        s02: i64,
        s03: i64,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
        stream: *mut c_void,
    );

    // MMQ matmul launchers (one per quant type)
    pub fn launch_mmq_gguf_q4_0(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q4_1(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q5_0(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q5_1(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q8_0(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q2_k(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q3_k(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q4_k(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q5_k(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q6_k(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );

    /// One-warp test kernel: 16x32 s8 × 32x8 s8 → 16x8 s32 via
    /// mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 (Ampere+).
    /// For development of the full IMMA-based MMVQ; not used in the
    /// model forward path.
    pub fn mma_test_m16n8k32(
        a: *const c_void,
        b: *const c_void,
        d: *mut c_void,
        stream: i64,
    );

    /// Tensor-core Q4_K × Q8_1 MMVQ for batch=1 decode. One warp per
    /// 16-row group, processes all super-blocks of those rows. Replaces
    /// the dp4a inner loop with mma.sync.aligned.m16n8k32 IMMA — 16
    /// weight rows in parallel per 32-K chunk.
    /// `ncols_x` must be a multiple of 256 (one Q4_K super-block).
    pub fn q4k_mmvq_imma(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stream: i64,
    );

    /// Tensor-core MoE Q4_K gate||up matmul + GELU(tanh) + mul. Drop-in
    /// replacement for the dp4a `moe_gemm_gguf_gate_up_gelu_mul_concat`
    /// (for Q4_K weights with K%256=0). Quantizes the F32 input internally.
    pub fn moe_q4k_imma_gate_up_gelu_mul_concat(
        inputs: *const c_void,
        gate_up_w: *const c_void,
        sorted_token_ids: *const i32,
        expert_ids:       *const i32,
        dst: *mut c_void,
        num_experts: i32,
        topk: i32,
        size_m: i32,
        n: i32,
        k: i32,
        stream: i64,
    );

    /// Fused MoE gate matmul + softmax + top-k. Combines the per-layer
    /// `logits = gate.forward(x)` (cublas SGEMV) and `topk_softmax(logits)`
    /// launches into one kernel. Loads `x` into shared memory once per
    /// token, then each warp lane computes a slice of the n_experts logits
    /// and the existing softmax+topk reduction proceeds in-place.
    /// `n_experts` ∈ {32, 64, 128, 256}; `hidden` ≤ 8192.
    pub fn gate_topk_softmax(
        x_in: *const c_void,
        gate_w: *const c_void,
        weights_out: *mut c_void,
        ids_out: *mut c_void,
        hidden: i32,
        n_rows: i32,
        n_experts: i32,
        n_expert_used: i32,
        with_norm: i32,
        stream: i64,
    );

    /// Multi-block F32 GEMV for the MoE router gate matmul. Replaces a
    /// cublas SGEMV (overhead-bound at ~16us/call on 2048×128) with a
    /// hand-rolled kernel that splits experts across many blocks for
    /// full SM saturation. `hidden` must be a multiple of 32.
    pub fn gate_gemv_f32(
        xs: *const c_void,
        gate_w: *const c_void,
        logits: *mut c_void,
        hidden: i32,
        n_rows: i32,
        n_experts: i32,
        stream: i64,
    );

    /// Scatter one (k or v) F16 vector into the Q4 KV residual buffer
    /// at slot `slot` ∈ [0, 32). Replaces a 3-launch dance
    /// (clone_dtod + slice_set + memcpy_dtod) with one scatter kernel.
    pub fn kv_residual_scatter_f16(
        src: *const c_void,
        dst: *mut c_void,
        n_kv: i32,
        head_dim: i32,
        slot: i32,
        stream: i64,
    );

    /// Same as `kv_residual_scatter_f16`, but reads `slot` from a device
    /// tensor (`slot_dev[0] & 31`) instead of taking it as a host int.
    /// Required for CUDA graph capture of Q4 KV append — the host updates
    /// `slot_dev` outside the captured region each token, while the
    /// captured kernel always reads from the same device pointer.
    pub fn kv_residual_scatter_f16_dev_slot(
        src: *const c_void,
        dst: *mut c_void,
        slot_dev: *const c_void,
        n_kv: i32,
        head_dim: i32,
        stream: i64,
    );

    /// Byte-copy scatter for the Q4_0 V append path. Copies `token_bytes`
    /// bytes from `src` into `dst[pos_dev[0] * token_bytes ..]`. Replaces
    /// host-side `memcpy_dtod` whose offset would freeze under CUDA graph
    /// capture. `token_bytes` must be a multiple of 4.
    pub fn q4_v_scatter_bytes_dev_pos(
        src: *const c_void,
        dst: *mut c_void,
        pos_dev: *const c_void,
        token_bytes: i32,
        stream: i64,
    );

    /// Conditional Q4_0 quantize-and-flush of the K residual to k_q4_blocks.
    /// Fires every token under graph capture; the kernel itself decides
    /// at run-time whether to write (only when `pos_dev[0] & 31 == 31`,
    /// i.e. this append closes a 32-token window). The host-int path's
    /// quantize-from-F16 + memcpy_dtod chain (which would freeze the
    /// block_idx at capture time) collapses into one launch.
    pub fn flush_k_residual_q4_dev_pos(
        residual: *const c_void,
        k_blocks: *mut c_void,
        pos_dev: *const c_void,
        n_kv: i32,
        head_dim: i32,
        max_seq_blocks: i32,
        stream: i64,
    );

    /// Dense (non-MoE) gate+up+silu*mul fused kernel.
    /// `gate_w` and `up_w` are quantized [N, K]; `input` is F32 [K];
    /// `output` is F32 [N] pre-allocated.
    /// Replaces 3 launches (ffn_up matmul + narrow + fused_silu_mul) with
    /// 1 quantize + 1 fused matmul.
    pub fn dense_gate_up_silu_mul_v2(
        input: *const f32,
        gate_w: *const c_void,
        up_w: *const c_void,
        output: *mut f32,
        size_n: i32,
        size_k: i32,
        gguf_dtype: i32,
        stream: i64,
    );

    /// Fused RMS norm + Q8_1 quantize for single-token decode.
    /// Replaces rms_norm + quantize_q8_1 (2 launches) with one. Caller
    /// then dispatches the matmul-vec kernel against the q8_1 output.
    /// `hidden` must be a multiple of 32. Single-block; hidden ≤ 16384.
    pub fn rms_quantize_q8_1(
        x: *const f32,          // [hidden]
        w_norm: *const f32,     // [hidden]
        y_q8_1: *mut c_void,    // [hidden/32 * sizeof(block_q8_1)]
        hidden: i32,
        rms_eps: f32,
        stream: i64,
    );

    /// Fused (a + b) + rms_norm for single-token decode.
    /// Writes both `xs_out = a + b` and `normed_out = gamma * xs / rms(xs)`
    /// in a single launch. Saves one launch vs separate broadcast_add +
    /// rms_norm. Single-block kernel; hidden ≤ 8192.
    pub fn add_rms_norm(
        a: *const f32,          // [hidden]
        b: *const f32,          // [hidden]
        gamma: *const f32,      // [hidden]
        xs_out: *mut f32,       // [hidden]
        normed_out: *mut f32,   // [hidden]
        hidden: i32,
        eps: f32,
        stream: i64,
    );
}
