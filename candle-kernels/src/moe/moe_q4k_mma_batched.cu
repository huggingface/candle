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
// Grid layout:
//   gridDim.x = ceil(2N / 16)      (16 weight rows per block — m16 of mma op)
//   gridDim.y = ceil(max_n_e / 8)  (8 input rows per block — n8 of mma op)
//   gridDim.z = N_active
//
// Per block: one warp (32 threads, blockDim.x = 32).
//   Lane (g, tj) with g = lane/4 (0..7) and tj = lane%4 (0..3) holds:
//     A frag (weights, m16k32, row layout): 4 INT32 regs = 16 INT8s
//       A0 = weight row (m_base + g)        cols 4tj..4tj+3
//       A1 = weight row (m_base + g + 8)    cols 4tj..4tj+3
//       A2 = weight row (m_base + g)        cols 16+4tj..16+4tj+3
//       A3 = weight row (m_base + g + 8)    cols 16+4tj..16+4tj+3
//     B frag (inputs, n8k32, col layout): 2 INT32 regs = 8 INT8s
//       B0 = input row (n_base + g)         cols 4tj..4tj+3
//       B1 = input row (n_base + g)         cols 16+4tj..16+4tj+3
//     D frag (output [16,8], INT32): 4 regs
//       D0 = D[m_base+g,   n_base+2tj+0]    (input row 2tj+0)
//       D1 = D[m_base+g,   n_base+2tj+1]    (input row 2tj+1)
//       D2 = D[m_base+g+8, n_base+2tj+0]    (input row 2tj+0)
//       D3 = D[m_base+g+8, n_base+2tj+1]    (input row 2tj+1)
//
// Q4_K scaling: each output D* is the raw INT32 dot of (sc-scaled) weights
// × Q8_1 inputs. Post-MMA we apply:
//   y[m_w, n_in] = dall_w * sum_s (sc_w_s * D_s) - dmin_w * d8_in * sum_s (m_w_s * sum32_in_s)
// where d8_in is the input row's Q8_1 scale and sum32_in_s is the sum of
// the 32 INT8 values in input row n_in for K-tile s. d8_in and sum32_in_s
// depend on the OUTPUT input-row index (2tj+0 or 2tj+1), which is NOT the
// same as this lane's loaded input row (g). We use __shfl_sync to fetch
// the values from lanes (2tj+0)*4 and (2tj+1)*4 respectively.
extern "C" __global__ void moe_q4k_mma_batched_gate_up_kernel(
    const void * __restrict__ gate_up_w,         // [num_experts, 2N, K/256] Q4_K
    const void * __restrict__ inputs_y,          // [N_active, max_n_e, K/32] Q8_1
    const int32_t * __restrict__ active_expert_ids,
    const int32_t * __restrict__ expert_offsets,
    void * __restrict__ dst,                     // [N_active, max_n_e, 2N] F16
    int num_experts,
    int max_n_e,
    int two_n,
    int K
) {
    const int act_idx = blockIdx.z;
    const int n_tile  = blockIdx.y;             // input-row tile (n8 of mma)
    const int m_tile  = blockIdx.x;             // weight-row tile (m16 of mma)

    const int expert  = active_expert_ids[act_idx];
    if (expert < 0 || expert >= num_experts) return;
    const int start   = expert_offsets[expert];
    const int end     = expert_offsets[expert + 1];
    const int n_e     = end - start;

    const int m_base = m_tile * MMA_BATCHED_M;   // weight-row base, 16 rows
    const int n_base = n_tile * MMA_BATCHED_N;   // input-row base, 8 rows

    if (m_base >= two_n || n_base >= max_n_e) return;

    using moe_q4k_mma_batched_ns::block_q4_K_mma;
    using moe_q4k_mma_batched_ns::block_q8_1_mma;
    using moe_q4k_mma_batched_ns::get_scale_min_k4_dev;

    const int lane = threadIdx.x;
    const int g    = lane >> 2;
    const int tj   = lane & 3;
    const int num_super = K / MMA_BATCHED_QK_K;

    const int row_a = m_base + g;
    const int row_b = m_base + g + 8;
    const bool va = row_a < two_n;
    const bool vb = row_b < two_n;

    // This lane's input row (the row it loads). 0..7 within the n8 block.
    const int my_in_row = n_base + g;
    const bool v_in = my_in_row < n_e;

    const block_q4_K_mma * w_expert =
        (const block_q4_K_mma *) gate_up_w
        + (size_t)expert * (size_t)two_n * num_super;
    const block_q8_1_mma * y_expert =
        (const block_q8_1_mma *) inputs_y
        + ((size_t)act_idx * max_n_e + n_base) * num_super * 8;

    // Per-thread accumulators for the 4 output positions D0..D3.
    // Each is INT32 raw dot, scaled per K-tile by sc and per super-block
    // by dall. The m sub-term is float, accumulated separately.
    float acc_d_0 = 0.f, acc_d_1 = 0.f, acc_d_2 = 0.f, acc_d_3 = 0.f;
    float acc_m_0 = 0.f, acc_m_1 = 0.f, acc_m_2 = 0.f, acc_m_3 = 0.f;

    for (int isb = 0; isb < num_super; ++isb) {
        const block_q4_K_mma * wa_sb = va ? w_expert + (size_t)row_a * num_super + isb : nullptr;
        const block_q4_K_mma * wb_sb = vb ? w_expert + (size_t)row_b * num_super + isb : nullptr;

        const float dall_a = wa_sb ? __low2float (wa_sb->dm) : 0.f;
        const float dmin_a = wa_sb ? __high2float(wa_sb->dm) : 0.f;
        const float dall_b = wb_sb ? __low2float (wb_sb->dm) : 0.f;
        const float dmin_b = wb_sb ? __high2float(wb_sb->dm) : 0.f;

        float sub_d_0 = 0.f, sub_d_1 = 0.f, sub_d_2 = 0.f, sub_d_3 = 0.f;
        float sub_m_0 = 0.f, sub_m_1 = 0.f, sub_m_2 = 0.f, sub_m_3 = 0.f;

        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int il = s >> 1;
            const int ip = s & 1;
            const int qs_off = 32 * il + 8 * tj;

            uint32_t qa_lo = wa_sb ? *(const uint32_t *)(wa_sb->qs + qs_off + 0) : 0;
            uint32_t qa_hi = wa_sb ? *(const uint32_t *)(wa_sb->qs + qs_off + 4) : 0;
            uint32_t qb_lo = wb_sb ? *(const uint32_t *)(wb_sb->qs + qs_off + 0) : 0;
            uint32_t qb_hi = wb_sb ? *(const uint32_t *)(wb_sb->qs + qs_off + 4) : 0;

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

            // B fragment: this lane loads input row (n_base + g)'s K-tile s.
            // The K-permutation matches IMMA's A-fragment loading (K =
            // 8tj..8tj+3 in slot b0, K = 8tj+4..8tj+7 in slot b1). What
            // matters for the mma op is that A and B's fragment slots
            // hold MATCHING K positions per lane — both at K=8tj..8tj+7
            // here.
            const block_q8_1_mma * yb_my = v_in
                ? y_expert + (size_t)g * num_super * 8 + isb * 8 + s
                : nullptr;
            int B0 = yb_my ? ((const int *)yb_my->qs)[2 * tj + 0] : 0;
            int B1 = yb_my ? ((const int *)yb_my->qs)[2 * tj + 1] : 0;
            const float d8_my = yb_my ? __low2float(yb_my->ds) : 0.f;

            int D0 = 0, D1 = 0, D2 = 0, D3 = 0;
            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+r"(D0), "+r"(D1), "+r"(D2), "+r"(D3)
                : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1));

            // Per-input-row sum32 (used for the dmin sub-term). dot2 holds
            // the sum of INT8 values for THIS lane's input row across the
            // 32 K cols of K-tile s; shfl_xor across tj sums the 4
            // K-segments to give the total per K-tile per input row.
            int dot2 = __dp4a(0x01010101, B1, __dp4a(0x01010101, B0, 0));
            dot2 += __shfl_xor_sync(0xffffffff, dot2, 1);
            dot2 += __shfl_xor_sync(0xffffffff, dot2, 2);

            // Fetch d8 and dot2 for the OUTPUT input-rows (2tj+0, 2tj+1).
            // Source lanes are (2tj+0)*4 and (2tj+1)*4 (any tj within g).
            const int src_lane_a = (2 * tj + 0) * 4;
            const int src_lane_b = (2 * tj + 1) * 4;
            const float d8_out_a = __shfl_sync(0xffffffff, d8_my, src_lane_a);
            const float d8_out_b = __shfl_sync(0xffffffff, d8_my, src_lane_b);
            const float dot_out_a = (float)__shfl_sync(0xffffffff, dot2, src_lane_a);
            const float dot_out_b = (float)__shfl_sync(0xffffffff, dot2, src_lane_b);


            uint8_t sc_a, m_a, sc_b, m_b;
            if (wa_sb) get_scale_min_k4_dev(s, wa_sb->scales, sc_a, m_a); else { sc_a = m_a = 0; }
            if (wb_sb) get_scale_min_k4_dev(s, wb_sb->scales, sc_b, m_b); else { sc_b = m_b = 0; }

            // D0,D1 are at weight-row a; D2,D3 at weight-row b.
            // D0,D2 at input-row 2tj+0; D1,D3 at input-row 2tj+1.
            sub_d_0 += d8_out_a * (float)D0 * (float)sc_a;
            sub_d_1 += d8_out_b * (float)D1 * (float)sc_a;
            sub_d_2 += d8_out_a * (float)D2 * (float)sc_b;
            sub_d_3 += d8_out_b * (float)D3 * (float)sc_b;

            sub_m_0 += d8_out_a * dot_out_a * (float)m_a;
            sub_m_1 += d8_out_b * dot_out_b * (float)m_a;
            sub_m_2 += d8_out_a * dot_out_a * (float)m_b;
            sub_m_3 += d8_out_b * dot_out_b * (float)m_b;
        }

        acc_d_0 += dall_a * sub_d_0 - dmin_a * sub_m_0;
        acc_d_1 += dall_a * sub_d_1 - dmin_a * sub_m_1;
        acc_d_2 += dall_b * sub_d_2 - dmin_b * sub_m_2;
        acc_d_3 += dall_b * sub_d_3 - dmin_b * sub_m_3;
    }

    // Output writes. Per-thread positions:
    //   D0 → dst[act_idx, n_base+2tj+0, row_a]
    //   D1 → dst[act_idx, n_base+2tj+1, row_a]
    //   D2 → dst[act_idx, n_base+2tj+0, row_b]
    //   D3 → dst[act_idx, n_base+2tj+1, row_b]
    const int in_row_0 = n_base + 2 * tj + 0;
    const int in_row_1 = n_base + 2 * tj + 1;
    __half * dst_base = (__half *) dst + (size_t)act_idx * max_n_e * two_n;

if (va && in_row_0 < n_e) {
        dst_base[(size_t)in_row_0 * two_n + row_a] = __float2half(acc_d_0);
    }
    if (va && in_row_1 < n_e) {
        dst_base[(size_t)in_row_1 * two_n + row_a] = __float2half(acc_d_1);
    }
    if (vb && in_row_0 < n_e) {
        dst_base[(size_t)in_row_0 * two_n + row_b] = __float2half(acc_d_2);
    }
    if (vb && in_row_1 < n_e) {
        dst_base[(size_t)in_row_1 * two_n + row_b] = __float2half(acc_d_3);
    }

    (void)num_experts;
}

// Host launcher.
extern "C" void moe_q4k_mma_batched_gate_up(
    const void * gate_up_w,
    const void * inputs_q81,
    const int32_t * active_expert_ids,
    const int32_t * expert_offsets,
    void * dst_f16,
    int num_experts,
    int n_active,
    int max_n_e,
    int two_n,
    int K,
    cudaStream_t stream
) {
    if (n_active <= 0 || max_n_e <= 0 || two_n <= 0 || K <= 0) return;
    using namespace moe_q4k_mma_batched_ns;
    dim3 grid((two_n + MMA_BATCHED_M - 1) / MMA_BATCHED_M,
              (max_n_e + MMA_BATCHED_N - 1) / MMA_BATCHED_N,
              n_active);
    dim3 blk(32, 1, 1);
    moe_q4k_mma_batched_gate_up_kernel<<<grid, blk, 0, stream>>>(
        gate_up_w, inputs_q81, active_expert_ids, expert_offsets,
        dst_f16, num_experts, max_n_e, two_n, K
    );
}
