/**
 * @file moe_q4k_imma_m8.cu
 *
 * Per-pair-tile IMMA: extends the M=1 broadcast moe_q4k_imma kernel to
 * M=8 batched form. Each block processes 8 CONSECUTIVE sorted pairs at
 * 16 weight rows. When the 8 pairs share an expert (the common case for
 * pairs sorted by expert), one mma.sync.m16n8k32 covers all 8 outputs.
 * At expert boundaries, the block falls back to per-pair processing
 * inside the block (slow path).
 *
 * Dispatch contract matches the existing IMMA orchestrator:
 *   - inputs: pre-quantized Q8_1 per real token
 *   - sorted_token_ids: per-pair token index
 *   - expert_ids: per-pair expert index
 *   - output: [size_m, N] F32, post-GELU·mul
 *
 * Grid:
 *   gridDim.x = ceil(N / 16)
 *   gridDim.y = ceil(size_m / 8)
 * blockDim = 32 (1 warp)
 *
 * Hypothesis: M=1 broadcast IMMA wastes 8× compute (mma produces 16×8
 * but only col 0 is used). M=8 batched IMMA recovers the 8× via real
 * column outputs.
 */
#include "gguf.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace moe_q4k_imma_m8_ns {

#define IM8_K_SUPER 256
#define IM8_M        16    // weight rows per block
#define IM8_N         8    // sorted-pairs per block

typedef struct {
    __half2 dm;
    uint8_t scales[12];
    uint8_t qs[128];
} block_q4_K_im8;

typedef struct {
    __half2 ds;
    int8_t  qs[32];
} block_q8_1_im8;

static_assert(sizeof(block_q4_K_im8) == 144, "block_q4_K_im8 size");
static_assert(sizeof(block_q8_1_im8) == 36,  "block_q8_1_im8 size");

__device__ __forceinline__ void im8_get_scale_min_k4(
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

// Per-pair-tile IMMA kernel. Output layout: [size_m, N] F32, gelu(gate)*up.
extern "C" __global__ void moe_q4k_imma_m8_kernel(
    const void * __restrict__ gate_up_w,         // [num_experts, 2N, K/256] Q4_K
    const void * __restrict__ inputs_q81,        // [num_real_tokens, K/32] Q8_1
    const int32_t * __restrict__ sorted_token_ids,
    const int32_t * __restrict__ expert_ids,
    float * __restrict__ dst,                    // [size_m, N] F32
    int num_experts,
    int topk,
    int size_m,
    int N,
    int K
) {
    using namespace moe_q4k_imma_m8_ns;

    const int n_tile  = blockIdx.y;   // sorted-pair tile
    const int m_tile  = blockIdx.x;   // weight-row tile

    const int pair_base = n_tile * IM8_N;
    if (pair_base >= size_m) return;

    const int m_base = m_tile * IM8_M;
    if (m_base >= N) return;

    const int lane = threadIdx.x;
    const int g    = lane >> 2;
    const int tj   = lane & 3;
    const int num_super = K / IM8_K_SUPER;
    const int two_n = N << 1;

    // GATE weight rows handled by this block.
    const int row_a = m_base + g;
    const int row_b = m_base + g + 8;
    const bool va = row_a < N;
    const bool vb = row_b < N;
    // UP weight rows = N + gate rows.
    const int row_a_up = N + row_a;
    const int row_b_up = N + row_b;

    // Each lane covers ONE input pair (the n=g dimension of mma).
    const int my_pair = pair_base + g;
    const bool v_pair = my_pair < size_m;

    int my_pair_tok = 0;
    int my_expert   = -1;
    if (v_pair) {
        my_pair_tok = sorted_token_ids[my_pair];
        my_expert   = expert_ids[my_pair];
    }

    // Boundary handling: sorted pairs are typically all-same-expert
    // across 8 consecutive entries, but occasionally span 2 (rarely
    // more) experts. Gather distinct experts present in the 8-pair tile
    // and loop the mma pass over each — single-expert blocks do 1 pass
    // (common case), boundary blocks do 2.
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

    // Outer loop: iterate over distinct experts in this 8-pair tile.
    for (int e_idx = 0; e_idx < num_block_experts; ++e_idx) {
    const int block_expert = experts_in_block[e_idx];

    const bool v_pair_e = v_pair && (my_expert == block_expert);

    // Q4_K weight pointers (gate + up halves) for the chosen block_expert.
    const block_q4_K_im8 * w_expert =
        (const block_q4_K_im8 *) gate_up_w
        + (size_t)block_expert * (size_t)two_n * num_super;

    // Pre-quantized Q8_1 input — pick MY pair's token row.
    const block_q8_1_im8 * y_expert = nullptr;
    if (v_pair_e) {
        const int real_token = my_pair_tok / topk;
        y_expert = (const block_q8_1_im8 *) inputs_q81
                 + (size_t)real_token * num_super * 8;
    }

    // Per-expert-pass accumulators (reset each outer iteration).
    float gate_0 = 0.f, gate_1 = 0.f, gate_2 = 0.f, gate_3 = 0.f;
    float up_0   = 0.f, up_1   = 0.f, up_2   = 0.f, up_3   = 0.f;

    for (int isb = 0; isb < num_super; ++isb) {
        const block_q4_K_im8 * gwa = va ? w_expert + (size_t)row_a    * num_super + isb : nullptr;
        const block_q4_K_im8 * gwb = vb ? w_expert + (size_t)row_b    * num_super + isb : nullptr;
        const block_q4_K_im8 * uwa = va ? w_expert + (size_t)row_a_up * num_super + isb : nullptr;
        const block_q4_K_im8 * uwb = vb ? w_expert + (size_t)row_b_up * num_super + isb : nullptr;

        const float g_dall_a = gwa ? __low2float (gwa->dm) : 0.f;
        const float g_dmin_a = gwa ? __high2float(gwa->dm) : 0.f;
        const float g_dall_b = gwb ? __low2float (gwb->dm) : 0.f;
        const float g_dmin_b = gwb ? __high2float(gwb->dm) : 0.f;
        const float u_dall_a = uwa ? __low2float (uwa->dm) : 0.f;
        const float u_dmin_a = uwa ? __high2float(uwa->dm) : 0.f;
        const float u_dall_b = uwb ? __low2float (uwb->dm) : 0.f;
        const float u_dmin_b = uwb ? __high2float(uwb->dm) : 0.f;

        // Hoist per-K-tile (sc, m) scales out of the inner s-loop AND
        // fold them with (dall, dmin) into F32 products. Saves both the
        // get_scale_min_k4 work and the per-K-tile uint8→float casts +
        // dall/dmin multiplies in the inner loop.
        float g_dall_sc_a[8], g_dmin_m_a[8], g_dall_sc_b[8], g_dmin_m_b[8];
        float u_dall_sc_a[8], u_dmin_m_a[8], u_dall_sc_b[8], u_dmin_m_b[8];
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            uint8_t sc, m;
            if (gwa) { im8_get_scale_min_k4(s, gwa->scales, sc, m); g_dall_sc_a[s] = g_dall_a * (float)sc; g_dmin_m_a[s]  = g_dmin_a * (float)m; }
            else     { g_dall_sc_a[s] = 0.f; g_dmin_m_a[s] = 0.f; }
            if (gwb) { im8_get_scale_min_k4(s, gwb->scales, sc, m); g_dall_sc_b[s] = g_dall_b * (float)sc; g_dmin_m_b[s]  = g_dmin_b * (float)m; }
            else     { g_dall_sc_b[s] = 0.f; g_dmin_m_b[s] = 0.f; }
            if (uwa) { im8_get_scale_min_k4(s, uwa->scales, sc, m); u_dall_sc_a[s] = u_dall_a * (float)sc; u_dmin_m_a[s]  = u_dmin_a * (float)m; }
            else     { u_dall_sc_a[s] = 0.f; u_dmin_m_a[s] = 0.f; }
            if (uwb) { im8_get_scale_min_k4(s, uwb->scales, sc, m); u_dall_sc_b[s] = u_dall_b * (float)sc; u_dmin_m_b[s]  = u_dmin_b * (float)m; }
            else     { u_dall_sc_b[s] = 0.f; u_dmin_m_b[s] = 0.f; }
        }

        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int il = s >> 1;
            const int ip = s & 1;
            const int qs_off = 32 * il + 8 * tj;

            // GATE A fragments.
            uint32_t gqa_lo = gwa ? *(const uint32_t *)(gwa->qs + qs_off + 0) : 0;
            uint32_t gqa_hi = gwa ? *(const uint32_t *)(gwa->qs + qs_off + 4) : 0;
            uint32_t gqb_lo = gwb ? *(const uint32_t *)(gwb->qs + qs_off + 0) : 0;
            uint32_t gqb_hi = gwb ? *(const uint32_t *)(gwb->qs + qs_off + 4) : 0;
            int GA0, GA1, GA2, GA3;
            if (ip == 0) {
                GA0 = (int)(gqa_lo & 0x0F0F0F0F);
                GA2 = (int)(gqa_hi & 0x0F0F0F0F);
                GA1 = (int)(gqb_lo & 0x0F0F0F0F);
                GA3 = (int)(gqb_hi & 0x0F0F0F0F);
            } else {
                GA0 = (int)((gqa_lo >> 4) & 0x0F0F0F0F);
                GA2 = (int)((gqa_hi >> 4) & 0x0F0F0F0F);
                GA1 = (int)((gqb_lo >> 4) & 0x0F0F0F0F);
                GA3 = (int)((gqb_hi >> 4) & 0x0F0F0F0F);
            }

            // UP A fragments.
            uint32_t uqa_lo = uwa ? *(const uint32_t *)(uwa->qs + qs_off + 0) : 0;
            uint32_t uqa_hi = uwa ? *(const uint32_t *)(uwa->qs + qs_off + 4) : 0;
            uint32_t uqb_lo = uwb ? *(const uint32_t *)(uwb->qs + qs_off + 0) : 0;
            uint32_t uqb_hi = uwb ? *(const uint32_t *)(uwb->qs + qs_off + 4) : 0;
            int UA0, UA1, UA2, UA3;
            if (ip == 0) {
                UA0 = (int)(uqa_lo & 0x0F0F0F0F);
                UA2 = (int)(uqa_hi & 0x0F0F0F0F);
                UA1 = (int)(uqb_lo & 0x0F0F0F0F);
                UA3 = (int)(uqb_hi & 0x0F0F0F0F);
            } else {
                UA0 = (int)((uqa_lo >> 4) & 0x0F0F0F0F);
                UA2 = (int)((uqa_hi >> 4) & 0x0F0F0F0F);
                UA1 = (int)((uqb_lo >> 4) & 0x0F0F0F0F);
                UA3 = (int)((uqb_hi >> 4) & 0x0F0F0F0F);
            }

            // B fragment: this lane's pair → Q8_1 K-tile s (input row).
            const block_q8_1_im8 * yb_my = y_expert
                ? y_expert + isb * 8 + s
                : nullptr;
            int B0 = yb_my ? ((const int *)yb_my->qs)[2 * tj + 0] : 0;
            int B1 = yb_my ? ((const int *)yb_my->qs)[2 * tj + 1] : 0;
            const float d8_my = yb_my ? __low2float(yb_my->ds) : 0.f;

            int GD0 = 0, GD1 = 0, GD2 = 0, GD3 = 0;
            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+r"(GD0), "+r"(GD1), "+r"(GD2), "+r"(GD3)
                : "r"(GA0), "r"(GA1), "r"(GA2), "r"(GA3), "r"(B0), "r"(B1));
            int UD0 = 0, UD1 = 0, UD2 = 0, UD3 = 0;
            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+r"(UD0), "+r"(UD1), "+r"(UD2), "+r"(UD3)
                : "r"(UA0), "r"(UA1), "r"(UA2), "r"(UA3), "r"(B0), "r"(B1));

            int dot2 = __dp4a(0x01010101, B1, __dp4a(0x01010101, B0, 0));
            dot2 += __shfl_xor_sync(0xffffffff, dot2, 1);
            dot2 += __shfl_xor_sync(0xffffffff, dot2, 2);

            const int src_lane_a = (2 * tj + 0) * 4;
            const int src_lane_b = (2 * tj + 1) * 4;
            const float d8_out_a  = __shfl_sync(0xffffffff, d8_my, src_lane_a);
            const float d8_out_b  = __shfl_sync(0xffffffff, d8_my, src_lane_b);
            const float dot_out_a = (float)__shfl_sync(0xffffffff, dot2, src_lane_a);
            const float dot_out_b = (float)__shfl_sync(0xffffffff, dot2, src_lane_b);

            const float g_da = g_dall_sc_a[s], g_dm = g_dmin_m_a[s];
            const float g_db = g_dall_sc_b[s], g_dn = g_dmin_m_b[s];
            const float u_da = u_dall_sc_a[s], u_dm = u_dmin_m_a[s];
            const float u_db = u_dall_sc_b[s], u_dn = u_dmin_m_b[s];

            // Folded: gate_*  += (dall_a*sc_s) * d8 * D - (dmin_a*m_s) * d8 * dot
            gate_0 += g_da * d8_out_a * (float)GD0  - g_dm * d8_out_a * dot_out_a;
            gate_1 += g_da * d8_out_b * (float)GD1  - g_dm * d8_out_b * dot_out_b;
            gate_2 += g_db * d8_out_a * (float)GD2  - g_dn * d8_out_a * dot_out_a;
            gate_3 += g_db * d8_out_b * (float)GD3  - g_dn * d8_out_b * dot_out_b;
            up_0   += u_da * d8_out_a * (float)UD0  - u_dm * d8_out_a * dot_out_a;
            up_1   += u_da * d8_out_b * (float)UD1  - u_dm * d8_out_b * dot_out_b;
            up_2   += u_db * d8_out_a * (float)UD2  - u_dn * d8_out_a * dot_out_a;
            up_3   += u_db * d8_out_b * (float)UD3  - u_dn * d8_out_b * dot_out_b;
        }
    }

    // GELU·mul + scatter to [size_m, N] F32.
    const float k0 = 0.7978845608028654f;
    const float k1 = 0.044715f;

    auto do_write = [&](int pair_local, int weight_row, float gv, float uv) {
        const int pair_idx = pair_base + pair_local;
        if (pair_idx >= size_m || weight_row >= N) return;
        const int e = expert_ids[pair_idx];
        if (e != block_expert) return;   // wrong-expert slot — skip
        const float gelu = 0.5f * gv * (1.f + tanhf(k0 * (gv + k1 * gv * gv * gv)));
        const int tok = sorted_token_ids[pair_idx];
        dst[(size_t)tok * N + weight_row] = gelu * uv;
    };

    if (va) {
        do_write(2 * tj + 0, row_a, gate_0, up_0);
        do_write(2 * tj + 1, row_a, gate_1, up_1);
    }
    if (vb) {
        do_write(2 * tj + 0, row_b, gate_2, up_2);
        do_write(2 * tj + 1, row_b, gate_3, up_3);
    }

    } // end outer expert loop

    (void)num_experts;
}

extern "C" void moe_q4k_imma_m8_gate_up(
    const void * gate_up_w,
    const void * inputs_q81,
    const int32_t * sorted_token_ids,
    const int32_t * expert_ids,
    float * dst_f32,
    int num_experts,
    int topk,
    int size_m,
    int N,
    int K,
    cudaStream_t stream
) {
    if (size_m <= 0 || N <= 0 || K <= 0) return;
    using namespace moe_q4k_imma_m8_ns;
    dim3 grid((N      + IM8_M - 1) / IM8_M,
              (size_m + IM8_N - 1) / IM8_N,
              1);
    dim3 blk(32, 1, 1);
    moe_q4k_imma_m8_kernel<<<grid, blk, 0, stream>>>(
        gate_up_w, inputs_q81, sorted_token_ids, expert_ids,
        dst_f32, num_experts, topk, size_m, N, K
    );
}
