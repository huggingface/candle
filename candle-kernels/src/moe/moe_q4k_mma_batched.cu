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

#define MMA_BATCHED_QK_K 256
#define MMA_BATCHED_M    16
#define MMA_BATCHED_N    8
#define MMA_BATCHED_K    32

typedef struct {
    __half2 dm;
    uint8_t scales[12];
    uint8_t qs[128];
} block_q4_K_mma;

typedef struct {
    __half2 ds;
    int8_t  qs[32];
} block_q8_1_mma;

static_assert(sizeof(block_q4_K_mma) == 4 + 12 + 128, "block_q4_K_mma size");
static_assert(sizeof(block_q8_1_mma) == 4 + 32, "block_q8_1_mma size");

__device__ __forceinline__ void get_scale_min_k4_dev(
    int j, const uint8_t * q, uint8_t & d, uint8_t & m
) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

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

    const int m_base = m_tile * MMA_BATCHED_M;
    const int n_base = n_tile * MMA_BATCHED_N;

    if (m_base >= max_n_e || n_base >= two_n) return;

    using moe_q4k_mma_batched_ns::block_q4_K_mma;
    using moe_q4k_mma_batched_ns::block_q8_1_mma;
    using moe_q4k_mma_batched_ns::get_scale_min_k4_dev;

    const int lane = threadIdx.x;
    const int blocks_per_row = K / MMA_BATCHED_QK_K;    // Q4_K super-blocks per row
    const int two_n_loc      = two_n;

    // Lane mapping for the m16n8k32 INT8 MMA fragments:
    //   A is [16, 32] INT8 — 4 INT32 regs per thread (each holds 4 INT8 elts).
    //   B is [8, 32] INT8 — 2 INT32 regs per thread.
    //   C/D is [16, 8] INT32 — 4 regs per thread.
    // Standard PTX lane→(row, col) decomposition: each warp is split into
    // groups of 4 threads, with g = lane/4 selecting the row pair and
    // tj = lane%4 selecting the column quartet.
    const int g  = lane >> 2;       // 0..7
    const int tj = lane & 3;        // 0..3

    // Two output rows per thread for A's m=16 fragment:
    //   row_a = m_base + g       (lower 8 rows)
    //   row_b = m_base + g + 8   (upper 8 rows)
    const int row_a = m_base + g;
    const int row_b = m_base + g + 8;
    const bool va = row_a < max_n_e;
    const bool vb = row_b < max_n_e;

    // Weight base pointer for this expert. Layout: [num_experts, 2N, blocks_per_row]
    // Q4_K super-blocks.
    const block_q4_K_mma * w_expert =
        (const block_q4_K_mma *) gate_up_w
        + (size_t)expert * (size_t)two_n_loc * blocks_per_row;

    // Inputs are pre-quantized Q8_1 in this design. The caller (per-expert
    // batched dispatch) is expected to gather then quantize F32 → Q8_1
    // into a [N_active, max_n_e, blocks_per_row * 8] block_q8_1 slab. The
    // gather kernel needs to be paired with a Q8_1 quantize step (todo).
    const block_q8_1_mma * y_slab = (const block_q8_1_mma *) inputs_y
        + ((size_t)act_idx * max_n_e + m_base) * blocks_per_row * 8;

    // Accumulator fragments for the OUTPUT columns this thread owns.
    // Each thread handles columns (4 * (n_base/MMA_N) + tj) and
    // (... + tj + 4). Stored as 4-wide INT32 for the s32.s8.s8.s32 MMA.
    float gate_acc_lo = 0.f, gate_acc_hi = 0.f;
    float up_acc_lo   = 0.f, up_acc_hi   = 0.f;

    // Loop over Q4_K super-blocks (every 8 K-tiles).
    for (int kb = 0; kb < blocks_per_row; ++kb) {
        // Weight pointers for the 4 output-row "buddies" this lane covers.
        // For batched per-expert, each block of 16 weight rows starts at
        // (n_tile * MMA_N), so the m16 dimension here is actually the
        // n_tile's 16-row group (we use the same MMA pattern as the
        // existing IMMA kernel, with N rows mapped to the m16 axis).
        const int w_row_a = n_base + g;       // weight row index
        const int w_row_b = n_base + g + 8;
        const bool vw_a = w_row_a < two_n_loc;
        const bool vw_b = w_row_b < two_n_loc;

        const block_q4_K_mma * wa = w_expert + (size_t)w_row_a * blocks_per_row + kb;
        const block_q4_K_mma * wb = w_expert + (size_t)w_row_b * blocks_per_row + kb;

        const float dall_a = vw_a ? __low2float (wa->dm) : 0.f;
        const float dmin_a = vw_a ? __high2float(wa->dm) : 0.f;
        const float dall_b = vw_b ? __low2float (wb->dm) : 0.f;
        const float dmin_b = vw_b ? __high2float(wb->dm) : 0.f;

        float acc_d_lo = 0.f, acc_m_lo = 0.f;
        float acc_d_hi = 0.f, acc_m_hi = 0.f;

        // 8 K-tiles per super-block. Same nibble-unpack and mma.sync
        // pattern as moe_q4k_imma_gate_up_gelu_mul_concat_kernel.
        // The MMA A side here is WEIGHTS (m16), B is INPUTS (n8).
        // Output [16 weight rows, 8 input rows] = [n_tile × 16, m_tile × 8].
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int il = s >> 1;
            const int ip = s & 1;
            const int qs_off = 32 * il + 8 * tj;

            uint32_t qa_lo = vw_a ? *(const uint32_t *)(wa->qs + qs_off + 0) : 0;
            uint32_t qa_hi = vw_a ? *(const uint32_t *)(wa->qs + qs_off + 4) : 0;
            uint32_t qb_lo = vw_b ? *(const uint32_t *)(wb->qs + qs_off + 0) : 0;
            uint32_t qb_hi = vw_b ? *(const uint32_t *)(wb->qs + qs_off + 4) : 0;

            int A0, A1, A2, A3;
            if (ip == 0) {
                A0 = (int)(qa_lo & 0x0F0F0F0F);
                A2 = (int)(qa_hi & 0x0F0F0F0F);
                A1 = (int)(qb_lo & 0x0F0F0F0F);
                A3 = (int)(qb_hi & 0x0F0F0F0F);
            } else {
                A0 = (int)((qa_lo >> 4) & 0x0F0F0F0F);
                A2 = (int)((qa_hi >> 4) & 0x0F0F0F0F);
                A1 = (int)((qb_lo >> 4) & 0x0F0F0F0F);
                A3 = (int)((qb_hi >> 4) & 0x0F0F0F0F);
            }

            // B: input Q8_1 for this K-tile. The lane mapping for n8
            // packs 8 rows × 32 cols into 32 thread × 2 INT32 regs.
            // For batched per-expert with m8 mapping to rows of inputs,
            // each thread loads 2 INT32s from the appropriate input row.
            const int yb_row = m_base + (lane & 7);  // 8 input rows
            const bool vy_row = yb_row < n_e;
            const block_q8_1_mma * yb = (vy_row && yb_row < max_n_e)
                ? y_slab + ((size_t)(lane & 7) * blocks_per_row + kb) * 8 + s
                : nullptr;
            int B0 = yb ? ((const int *)yb->qs)[2 * (lane >> 3) + 0] : 0;
            int B1 = yb ? ((const int *)yb->qs)[2 * (lane >> 3) + 1] : 0;

            int D0 = 0, D1 = 0, D2 = 0, D3 = 0;
            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+r"(D0), "+r"(D1), "+r"(D2), "+r"(D3)
                : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1));

            int dot2 = __dp4a(0x01010101, B1, __dp4a(0x01010101, B0, 0));
            dot2 += __shfl_xor_sync(0xffffffff, dot2, 1);
            dot2 += __shfl_xor_sync(0xffffffff, dot2, 2);

            const float d8 = yb ? __low2float(yb->ds) : 0.f;

            uint8_t sc_a, m_a, sc_b, m_b;
            get_scale_min_k4_dev(s, wa->scales, sc_a, m_a);
            get_scale_min_k4_dev(s, wb->scales, sc_b, m_b);

            acc_d_lo += d8 * (float)D0 * (float)sc_a;
            acc_m_lo += d8 * (float)dot2 * (float)m_a;
            acc_d_hi += d8 * (float)D2 * (float)sc_b;
            acc_m_hi += d8 * (float)dot2 * (float)m_b;
        }

        gate_acc_lo += dall_a * acc_d_lo - dmin_a * acc_m_lo;
        gate_acc_hi += dall_b * acc_d_hi - dmin_b * acc_m_hi;
    }

    // Output write. Each thread writes 2 elements of the [16, 8] tile
    // based on the standard m16n8 fragment layout.
    //   thread(g, tj) holds rows (m_base + g, m_base + g + 8) at
    //   columns (n_base + tj*2, n_base + tj*2 + 1) for the 4 acc lanes.
    const int n_out_lo = n_base + tj * 2 + 0;
    const int n_out_hi = n_base + tj * 2 + 1;
    if (va && n_out_lo < two_n_loc && row_a < n_e) {
        __half * dst_h = (__half *) dst
            + ((size_t)act_idx * max_n_e + row_a) * two_n_loc + n_out_lo;
        *dst_h = __float2half(gate_acc_lo);
    }
    if (vb && n_out_lo < two_n_loc && row_b < n_e) {
        __half * dst_h = (__half *) dst
            + ((size_t)act_idx * max_n_e + row_b) * two_n_loc + n_out_lo;
        *dst_h = __float2half(gate_acc_hi);
    }

    (void)num_experts;
    (void)up_acc_lo;
    (void)up_acc_hi;
}
