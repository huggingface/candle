/**
 * @file moe_q4k_imma_m8_down.cu
 *
 * Per-pair-tile IMMA M=8 for the MoE DOWN step.
 *
 * The down matmul takes [size_m sorted pairs, K_inter] F32 input
 * (= gate||up output) and Q4_K weight [num_experts, hidden, K_inter].
 * Output is [num_real_tokens, hidden] F32, accumulated via atomicAdd
 * scaled by topk_weights[pair] (sum across topk pairs per token).
 *
 * Structure mirrors moe_q4k_imma_m8 (gate||up) but:
 *   - No gate/up split (one matmul, no GELU)
 *   - Output atomicAdd to per-token slot with topk_weights scaling
 *   - Inputs pre-quantized to Q8_1 by the orchestrator (one Q8_1 row
 *     per sorted pair, NOT per real token — down inputs are
 *     per-pair-distinct after gate||up·GELU·mul).
 *
 * Grid:
 *   gridDim.x = ceil(hidden / 16)
 *   gridDim.y = ceil(size_m / 8)
 * blockDim = 32
 */
#include "gguf.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace moe_q4k_imma_m8_down_ns {

#define IDN_K_SUPER 256
#define IDN_M       16
#define IDN_N        8

typedef struct {
    __half2 dm;
    uint8_t scales[12];
    uint8_t qs[128];
} block_q4_K_idn;

typedef struct {
    __half2 ds;
    int8_t  qs[32];
} block_q8_1_idn;

static_assert(sizeof(block_q4_K_idn) == 144, "block_q4_K_idn size");
static_assert(sizeof(block_q8_1_idn) == 36,  "block_q8_1_idn size");

__device__ __forceinline__ void idn_get_scale_min_k4(
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

extern "C" __global__ void moe_q4k_imma_m8_down_kernel(
    const void * __restrict__ down_w,            // [num_experts, hidden, K_inter] Q4_K
    const void * __restrict__ inputs_q81,        // [size_m, K_inter/32] Q8_1 (per-pair)
    const int32_t * __restrict__ sorted_token_ids,
    const int32_t * __restrict__ expert_ids,
    const float   * __restrict__ topk_weights,
    float * __restrict__ dst,                    // [num_real_tokens, hidden] F32
    int num_experts,
    int topk,
    int size_m,
    int hidden,
    int K
) {
    using namespace moe_q4k_imma_m8_down_ns;

    const int n_tile = blockIdx.y;
    const int pair_base = n_tile * IDN_N;
    if (pair_base >= size_m) return;

    const int m_tile = blockIdx.x;
    const int m_base = m_tile * IDN_M;
    if (m_base >= hidden) return;

    const int lane = threadIdx.x;
    const int g    = lane >> 2;
    const int tj   = lane & 3;
    const int num_super = K / IDN_K_SUPER;

    const int row_a = m_base + g;
    const int row_b = m_base + g + 8;
    const bool va = row_a < hidden;
    const bool vb = row_b < hidden;

    const int my_pair = pair_base + g;
    const bool v_pair = my_pair < size_m;
    int my_pair_tok = 0;
    int my_expert   = -1;
    if (v_pair) {
        my_pair_tok = sorted_token_ids[my_pair];
        my_expert   = expert_ids[my_pair];
    }

    // Boundary handling (same as gate||up IMMA M=8).
    int experts_in_block[4] = {-1, -1, -1, -1};
    int num_block_experts = 0;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        const int idx = pair_base + p;
        if (idx >= size_m) break;
        const int e = expert_ids[idx];
        if (e < 0 || e >= num_experts) continue;
        bool seen = false;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (i < num_block_experts && experts_in_block[i] == e) seen = true;
        }
        if (!seen && num_block_experts < 4) {
            experts_in_block[num_block_experts++] = e;
        }
    }
    if (num_block_experts == 0) return;

    for (int e_idx = 0; e_idx < num_block_experts; ++e_idx) {
    const int block_expert = experts_in_block[e_idx];

    const bool v_pair_e = v_pair && (my_expert == block_expert);

    const block_q4_K_idn * w_expert =
        (const block_q4_K_idn *) down_w
        + (size_t)block_expert * (size_t)hidden * num_super;

    // Down inputs are PER-PAIR (not per-real-token like gate||up).
    const block_q8_1_idn * y_pair = nullptr;
    if (v_pair_e) {
        y_pair = (const block_q8_1_idn *) inputs_q81
               + (size_t)my_pair * num_super * 8;
    }

    float out_0 = 0.f, out_1 = 0.f, out_2 = 0.f, out_3 = 0.f;

    for (int isb = 0; isb < num_super; ++isb) {
        const block_q4_K_idn * wa_sb = va ? w_expert + (size_t)row_a * num_super + isb : nullptr;
        const block_q4_K_idn * wb_sb = vb ? w_expert + (size_t)row_b * num_super + isb : nullptr;

        const float dall_a = wa_sb ? __low2float (wa_sb->dm) : 0.f;
        const float dmin_a = wa_sb ? __high2float(wa_sb->dm) : 0.f;
        const float dall_b = wb_sb ? __low2float (wb_sb->dm) : 0.f;
        const float dmin_b = wb_sb ? __high2float(wb_sb->dm) : 0.f;

        float dall_sc_a[8], dmin_m_a[8], dall_sc_b[8], dmin_m_b[8];
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            uint8_t sc, m;
            if (wa_sb) { idn_get_scale_min_k4(s, wa_sb->scales, sc, m); dall_sc_a[s] = dall_a * (float)sc; dmin_m_a[s] = dmin_a * (float)m; }
            else       { dall_sc_a[s] = 0.f; dmin_m_a[s] = 0.f; }
            if (wb_sb) { idn_get_scale_min_k4(s, wb_sb->scales, sc, m); dall_sc_b[s] = dall_b * (float)sc; dmin_m_b[s] = dmin_b * (float)m; }
            else       { dall_sc_b[s] = 0.f; dmin_m_b[s] = 0.f; }
        }

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

            const block_q8_1_idn * yb_my = y_pair ? y_pair + isb * 8 + s : nullptr;
            int B0 = yb_my ? ((const int *)yb_my->qs)[2 * tj + 0] : 0;
            int B1 = yb_my ? ((const int *)yb_my->qs)[2 * tj + 1] : 0;
            const float d8_my = yb_my ? __low2float(yb_my->ds) : 0.f;

            int D0 = 0, D1 = 0, D2 = 0, D3 = 0;
            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+r"(D0), "+r"(D1), "+r"(D2), "+r"(D3)
                : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1));

            int dot2 = __dp4a(0x01010101, B1, __dp4a(0x01010101, B0, 0));
            dot2 += __shfl_xor_sync(0xffffffff, dot2, 1);
            dot2 += __shfl_xor_sync(0xffffffff, dot2, 2);

            const int src_lane_a = (2 * tj + 0) * 4;
            const int src_lane_b = (2 * tj + 1) * 4;
            const float d8_out_a  = __shfl_sync(0xffffffff, d8_my, src_lane_a);
            const float d8_out_b  = __shfl_sync(0xffffffff, d8_my, src_lane_b);
            const float dot_out_a = (float)__shfl_sync(0xffffffff, dot2, src_lane_a);
            const float dot_out_b = (float)__shfl_sync(0xffffffff, dot2, src_lane_b);

            const float da_a = dall_sc_a[s], dm_a = dmin_m_a[s];
            const float da_b = dall_sc_b[s], dm_b = dmin_m_b[s];

            out_0 += da_a * d8_out_a * (float)D0 - dm_a * d8_out_a * dot_out_a;
            out_1 += da_a * d8_out_b * (float)D1 - dm_a * d8_out_b * dot_out_b;
            out_2 += da_b * d8_out_a * (float)D2 - dm_b * d8_out_a * dot_out_a;
            out_3 += da_b * d8_out_b * (float)D3 - dm_b * d8_out_b * dot_out_b;
        }
    }

    // Atomic scatter with topk_weights scaling.
    auto do_atomic = [&](int pair_local, int weight_row, float v) {
        const int pair_idx = pair_base + pair_local;
        if (pair_idx >= size_m || weight_row >= hidden) return;
        const int e = expert_ids[pair_idx];
        if (e != block_expert) return;
        const int pair_tok = sorted_token_ids[pair_idx];
        const float scale = topk_weights[pair_tok];
        const int real_token = pair_tok / topk;
        atomicAdd(&dst[(size_t)real_token * hidden + weight_row], v * scale);
    };

    if (va) {
        do_atomic(2 * tj + 0, row_a, out_0);
        do_atomic(2 * tj + 1, row_a, out_1);
    }
    if (vb) {
        do_atomic(2 * tj + 0, row_b, out_2);
        do_atomic(2 * tj + 1, row_b, out_3);
    }

    } // end outer expert loop

    (void)num_experts;
}

extern "C" void moe_q4k_imma_m8_down(
    const void * down_w,
    const void * inputs_q81,
    const int32_t * sorted_token_ids,
    const int32_t * expert_ids,
    const float * topk_weights,
    float * dst_f32,
    int num_experts,
    int topk,
    int size_m,
    int hidden,
    int K,
    cudaStream_t stream
) {
    if (size_m <= 0 || hidden <= 0 || K <= 0) return;
    using namespace moe_q4k_imma_m8_down_ns;
    dim3 grid((hidden + IDN_M - 1) / IDN_M,
              (size_m + IDN_N - 1) / IDN_N,
              1);
    dim3 blk(32, 1, 1);
    moe_q4k_imma_m8_down_kernel<<<grid, blk, 0, stream>>>(
        down_w, inputs_q81, sorted_token_ids, expert_ids, topk_weights,
        dst_f32, num_experts, topk, size_m, hidden, K
    );
}
