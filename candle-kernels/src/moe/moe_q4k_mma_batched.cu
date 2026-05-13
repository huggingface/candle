/**
 * @file moe_q4k_mma_batched.cu
 *
 * Phase 1 step-5 (Task #71 closure): per-expert batched Q4_K MMA GEMM.
 *
 * GOAL: close the 12× gemma4:26b prefill gap to Ollama (2195 → 26938
 * prompt-tok/s) by replacing the dp4a per-(token,expert) inner loop
 * with `mma.sync.aligned.m16n8k32` tensor-core ops, dispatched in a
 * per-expert batched form.
 *
 * Why this is the right strategy (per session-3 measurement chain in
 * docs/phase1_moe_implementation_plan.md): the dequant-then-F16-GEMM
 * approach (5 wiring variants tried) is irreducibly bandwidth-bound by
 * the ~3 GB F16 weight materialization per call. Direct Q4_K MMQ
 * consumes the quantized weights with tensor cores (~16× peak FLOPs of
 * the dp4a path) without materializing an F16 intermediate.
 *
 * STATUS: stub. The kernel signature + launch math + the per-expert
 * dispatch contract are in place; the inner-loop body needs the
 * mma.sync inline asm to be wired (multi-day task on top of correct
 * Q4_K → INT8 lane unpacking).
 *
 * INTERFACE
 * ---------
 * Inputs:
 *   - gate_up_w  [num_experts, 2N, K] Q4_K (288 bytes / 256 elems)
 *   - inputs_y   [N_active, max_n_e, K] F16, padded gather output
 *                from `moe_batched_gather_input_rows_f32_to_f16`
 *   - active_expert_ids [N_active] i32
 *   - expert_offsets    [num_experts + 1] i32
 *   - sorted_token_ids  [size_m] i32  (only needed if the kernel
 *                                       also fuses the scatter; here
 *                                       we keep them separate)
 *
 * Output:
 *   - dst [N_active, max_n_e, 2N] F16
 *
 * Grid:
 *   gridDim.x = ceil(2N / N_TILE)        where N_TILE = 16 (gate||up packed)
 *   gridDim.y = ceil(max_n_e / M_TILE)    where M_TILE = 16
 *   gridDim.z = N_active
 *
 * Per-block (one warp, 32 threads, blockDim.x = 32):
 *   1. Compute (act_idx, m_tile, n_tile) from blockIdx.
 *   2. Look up expert = active_expert_ids[act_idx], n_e from offsets.
 *   3. For each K-tile of K_TILE = 32 quantized elements:
 *      a. Load Q4_K block bytes for this (n_tile, k_tile) for the expert.
 *      b. Dequant to INT8 in shared memory (32 elements per Q4_K nibble pair).
 *      c. Load Q8_1 input lane from inputs_y[(act_idx, m, k)].
 *      d. mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 to FP32 acc.
 *   4. Scale by the Q4_K (dall, dmin) factors.
 *   5. Apply mask: rows m ≥ n_e write 0 (padded).
 *   6. Write F16 output.
 *
 * IMPLEMENTATION NOTES (for the kernel-writer)
 * -------------------------------------------
 * - mma.sync uses register-resident operand fragments. For Q4_K with
 *   nibble lanes, you'll unpack to INT8 in registers per K-tile and
 *   apply the (sc, m) scale factors after the mma accumulation, not
 *   inside. This matches llama.cpp's mmq.cu m16n8k32 path.
 * - The dall/dmin scale broadcast is per-256-element super-block; with
 *   K_TILE=32 you process 8 K-tiles per super-block, all sharing the
 *   same scale. Cache it in a register.
 * - For correctness, validate against the existing dp4a kernel on a
 *   small test (4 experts, 32 tokens, K=256) before scaling up.
 *
 * SOURCE REFERENCES (in tmp/)
 * ----------------------------
 * - tmp/llama.cpp/ggml/src/ggml-cuda/mmq.cu — production Q4_K MMQ
 * - tmp/llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh — dp4a baseline
 * - tmp/llama.cpp/ggml/src/ggml-cuda/quantize.cu — Q8_1 quantize
 */

#include "gguf.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace moe_q4k_mma_batched_ns {

#define QK_K_MMA 256
#define MMA_M 16
#define MMA_N 8
#define MMA_K 32

typedef struct {
    __half2 dm;
    uint8_t scales[12];
    uint8_t qs[128];
} block_q4_K_mma;

static_assert(sizeof(block_q4_K_mma) == 4 + 12 + 128, "block_q4_K_mma size");

} // namespace

/**
 * Kernel entry. Not yet wired into a launcher (the mma.sync inner body
 * is the missing piece). The launcher and FFI binding are deferred
 * until the body compiles; the stub keeps the build green.
 */
extern "C" __global__ void moe_q4k_mma_batched_gate_up_kernel(
    const void * __restrict__ gate_up_w,         // [num_experts, 2N, K] Q4_K
    const void * __restrict__ inputs_y,          // [N_active, max_n_e, K] F16
    const int32_t * __restrict__ active_expert_ids,
    const int32_t * __restrict__ expert_offsets,
    void * __restrict__ dst,                     // [N_active, max_n_e, 2N] F16
    int num_experts,
    int max_n_e,
    int two_n,
    int K
) {
    // Compute block indices.
    const int act_idx = blockIdx.z;
    const int m_tile  = blockIdx.y;
    const int n_tile  = blockIdx.x;

    const int expert  = active_expert_ids[act_idx];
    if (expert < 0 || expert >= num_experts) return;
    const int start   = expert_offsets[expert];
    const int end     = expert_offsets[expert + 1];
    const int n_e     = end - start;

    const int m_base = m_tile * moe_q4k_mma_batched_ns::MMA_M;
    const int n_base = n_tile * moe_q4k_mma_batched_ns::MMA_N;

    if (m_base >= max_n_e || n_base >= two_n) return;

    using moe_q4k_mma_batched_ns::block_q4_K_mma;
    using moe_q4k_mma_batched_ns::QK_K_MMA;
    using moe_q4k_mma_batched_ns::MMA_M;
    using moe_q4k_mma_batched_ns::MMA_N;
    using moe_q4k_mma_batched_ns::MMA_K;

    const int lane = threadIdx.x;
    const int blocks_per_row = K / QK_K_MMA;    // Q4_K super-blocks per weight row
    const int two_n_loc      = two_n;            // gate||up packed: 2N rows total

    // Weight base pointer for this expert. Each expert has [2N, K/QK_K_MMA]
    // Q4_K super-blocks stored contiguously.
    const block_q4_K_mma * w_expert =
        (const block_q4_K_mma *) gate_up_w
        + (size_t)expert * two_n_loc * blocks_per_row;

    // Inputs are F16 [N_active, max_n_e, K]; this block's slab is the
    // expert's [max_n_e, K] slice at offset act_idx * max_n_e * K elems.
    const __half * y_slab = (const __half *) inputs_y
        + ((size_t)act_idx * max_n_e + m_base) * K;

    // MMA accumulator fragment: 4 i32 lanes per thread = [16, 8] tile.
    int acc[4] = {0, 0, 0, 0};

    // Loop K in MMA_K-sized chunks. For Q4_K (256 elems per block) and
    // MMA_K=32, each super-block covers 8 mma K-tiles. The Q4_K
    // (dall, dmin) scale is per-super-block, so we accumulate the
    // raw INT32 dot then scale at the end.
    //
    // NOTE: this loop body is the WORK-IN-PROGRESS half. The
    // INT8 lane assembly from Q4_K nibbles + the Q8_1 input lane
    // load + the inline mma.sync need to be wired exactly like
    // moe_q4k_imma_gate_up_gelu_mul_concat_kernel does it (lines
    // 130-185 of moe_q4k_imma_gate_up.cu) — adapted to read the
    // INPUT side from y_slab (gather workspace) instead of vy
    // (per-token Q8_1 cache).
    //
    // The dispatch + accumulator layout above is correct; the
    // remaining work is plumbing the INT8 fragments through the
    // existing mma.sync asm block.
    for (int kb = 0; kb < blocks_per_row; ++kb) {
        const block_q4_K_mma * wb = w_expert
            + (size_t)(n_base + lane / 4) * blocks_per_row + kb;
        (void)wb;
        // ...inner mma.sync.aligned.m16n8k32 calls per K-tile...
    }

    // Write F16 output. Each thread writes one element of the
    // [16, 8] tile based on its lane mapping.
    const int m_out = m_base + (lane >> 2);
    const int n_out = n_base + (lane & 3);
    if (m_out < n_e && m_out < max_n_e && n_out < two_n) {
        __half * dst_h = (__half *) dst
            + ((size_t)act_idx * max_n_e + m_out) * two_n_loc
            + n_out;
        // Once the mma asm fills acc[], this should be a scaled cast.
        *dst_h = __float2half((float)acc[0]);
    }

    (void)num_experts;
}
