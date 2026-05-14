/**
 * @file moe_q4k_imma_m8_m32.cu
 *
 * IMMA M=8 variant with M_TILE=32: each block produces 32 weight rows
 * × 8 sorted pairs = 256 outputs (vs 128 in M=16 base). Halves the
 * number of blocks for the same compute → more outputs per block.
 *
 * The mma m16n8k32 op handles 16 m rows at a time, so this kernel does
 * 2 mma ops per K-tile per warp (one for m=0..15, one for m=16..31).
 * Same K and B fragments shared between the two mma calls.
 */
#include "gguf.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace moe_q4k_imma_m8_m32_ns {

#define IM32_K_SUPER 256
#define IM32_M       32
#define IM32_N        8

typedef struct {
    __half2 dm;
    uint8_t scales[12];
    uint8_t qs[128];
} block_q4_K_m32;

typedef struct {
    __half2 ds;
    int8_t  qs[32];
} block_q8_1_m32;

static_assert(sizeof(block_q4_K_m32) == 144, "block_q4_K_m32 size");
static_assert(sizeof(block_q8_1_m32) == 36,  "block_q8_1_m32 size");

__device__ __forceinline__ void m32_get_scale_min_k4(
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

extern "C" __global__ void moe_q4k_imma_m8_m32_kernel(
    const void * __restrict__ gate_up_w,
    const void * __restrict__ inputs_q81,
    const int32_t * __restrict__ sorted_token_ids,
    const int32_t * __restrict__ expert_ids,
    float * __restrict__ dst,
    int num_experts,
    int topk,
    int size_m,
    int N,
    int K
) {
    using namespace moe_q4k_imma_m8_m32_ns;

    const int n_tile = blockIdx.y;
    const int pair_base = n_tile * IM32_N;
    if (pair_base >= size_m) return;

    const int m_tile = blockIdx.x;
    const int m_base = m_tile * IM32_M;
    if (m_base >= N) return;

    const int lane = threadIdx.x;
    const int g    = lane >> 2;
    const int tj   = lane & 3;
    const int num_super = K / IM32_K_SUPER;
    const int two_n = N << 1;

    // 4 weight rows per thread: m_base + g, +8, +16, +24
    // Plus gate=rows 0..N-1, up=rows N..2N-1
    const int rows[4] = { m_base + g, m_base + g + 8, m_base + g + 16, m_base + g + 24 };
    const bool vr[4]  = { rows[0] < N, rows[1] < N, rows[2] < N, rows[3] < N };
    const int rows_up[4] = { N + rows[0], N + rows[1], N + rows[2], N + rows[3] };

    const int my_pair = pair_base + g;
    const bool v_pair = my_pair < size_m;
    int my_pair_tok = 0;
    int my_expert   = -1;
    if (v_pair) {
        my_pair_tok = sorted_token_ids[my_pair];
        my_expert   = expert_ids[my_pair];
    }

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

    const block_q4_K_m32 * w_expert =
        (const block_q4_K_m32 *) gate_up_w
        + (size_t)block_expert * (size_t)two_n * num_super;
    const block_q8_1_m32 * y_expert = nullptr;
    if (v_pair_e) {
        const int real_token = my_pair_tok / topk;
        y_expert = (const block_q8_1_m32 *) inputs_q81
                 + (size_t)real_token * num_super * 8;
    }

    // 8 accumulators per thread: 4 rows × 2 pairs (2tj+0 / 2tj+1) for gate
    // 8 more for up.
    float g_acc[8] = {0,0,0,0,0,0,0,0};
    float u_acc[8] = {0,0,0,0,0,0,0,0};

    for (int isb = 0; isb < num_super; ++isb) {
        // Load weight pointers + scales for the 4 row pairs (gate) and (up).
        const block_q4_K_m32 * gw[4];
        const block_q4_K_m32 * uw[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            gw[r] = vr[r] ? w_expert + (size_t)rows[r]    * num_super + isb : nullptr;
            uw[r] = vr[r] ? w_expert + (size_t)rows_up[r] * num_super + isb : nullptr;
        }

        float g_dall[4], g_dmin[4], u_dall[4], u_dmin[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            g_dall[r] = gw[r] ? __low2float(gw[r]->dm) : 0.f;
            g_dmin[r] = gw[r] ? __high2float(gw[r]->dm) : 0.f;
            u_dall[r] = uw[r] ? __low2float(uw[r]->dm) : 0.f;
            u_dmin[r] = uw[r] ? __high2float(uw[r]->dm) : 0.f;
        }

        float g_dall_sc[4][8], g_dmin_m[4][8], u_dall_sc[4][8], u_dmin_m[4][8];
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            #pragma unroll
            for (int s = 0; s < 8; ++s) {
                uint8_t sc, m;
                if (gw[r]) { m32_get_scale_min_k4(s, gw[r]->scales, sc, m); g_dall_sc[r][s] = g_dall[r] * (float)sc; g_dmin_m[r][s] = g_dmin[r] * (float)m; }
                else       { g_dall_sc[r][s] = 0.f; g_dmin_m[r][s] = 0.f; }
                if (uw[r]) { m32_get_scale_min_k4(s, uw[r]->scales, sc, m); u_dall_sc[r][s] = u_dall[r] * (float)sc; u_dmin_m[r][s] = u_dmin[r] * (float)m; }
                else       { u_dall_sc[r][s] = 0.f; u_dmin_m[r][s] = 0.f; }
            }
        }

        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int il = s >> 1;
            const int ip = s & 1;
            const int qs_off = 32 * il + 8 * tj;

            // For each pair of weight rows (m_base+g, m_base+g+8) and
            // (m_base+g+16, m_base+g+24), do an m16 mma op.
            // mma m16n8k32: A holds rows m=g (a0,a2) and m=g+8 (a1,a3).
            // For the FIRST mma: weight rows g, g+8 (indices 0, 1 in rows[])
            // For the SECOND mma: weight rows g+16, g+24 (indices 2, 3)

            const block_q8_1_m32 * yb_my = y_expert ? y_expert + isb * 8 + s : nullptr;
            int B0 = yb_my ? ((const int *)yb_my->qs)[2 * tj + 0] : 0;
            int B1 = yb_my ? ((const int *)yb_my->qs)[2 * tj + 1] : 0;
            const float d8_my = yb_my ? __low2float(yb_my->ds) : 0.f;

            int dot2 = __dp4a(0x01010101, B1, __dp4a(0x01010101, B0, 0));
            dot2 += __shfl_xor_sync(0xffffffff, dot2, 1);
            dot2 += __shfl_xor_sync(0xffffffff, dot2, 2);

            const int src_lane_a = (2 * tj + 0) * 4;
            const int src_lane_b = (2 * tj + 1) * 4;
            const float d8_a  = __shfl_sync(0xffffffff, d8_my, src_lane_a);
            const float d8_b  = __shfl_sync(0xffffffff, d8_my, src_lane_b);
            const float dot_a = (float)__shfl_sync(0xffffffff, dot2, src_lane_a);
            const float dot_b = (float)__shfl_sync(0xffffffff, dot2, src_lane_b);

            #pragma unroll
            for (int mma_idx = 0; mma_idx < 2; ++mma_idx) {
                // mma_idx=0: rows[0]=g, rows[1]=g+8
                // mma_idx=1: rows[2]=g+16, rows[3]=g+24
                const int r0 = mma_idx * 2 + 0;
                const int r1 = mma_idx * 2 + 1;
                const block_q4_K_m32 * gw0 = gw[r0];
                const block_q4_K_m32 * gw1 = gw[r1];
                const block_q4_K_m32 * uw0 = uw[r0];
                const block_q4_K_m32 * uw1 = uw[r1];

                // GATE A unpack
                uint32_t gq0_lo = gw0 ? *(const uint32_t *)(gw0->qs + qs_off + 0) : 0;
                uint32_t gq0_hi = gw0 ? *(const uint32_t *)(gw0->qs + qs_off + 4) : 0;
                uint32_t gq1_lo = gw1 ? *(const uint32_t *)(gw1->qs + qs_off + 0) : 0;
                uint32_t gq1_hi = gw1 ? *(const uint32_t *)(gw1->qs + qs_off + 4) : 0;
                int GA0, GA1, GA2, GA3;
                if (ip == 0) {
                    GA0 = (int)(gq0_lo & 0x0F0F0F0F);
                    GA2 = (int)(gq0_hi & 0x0F0F0F0F);
                    GA1 = (int)(gq1_lo & 0x0F0F0F0F);
                    GA3 = (int)(gq1_hi & 0x0F0F0F0F);
                } else {
                    GA0 = (int)((gq0_lo >> 4) & 0x0F0F0F0F);
                    GA2 = (int)((gq0_hi >> 4) & 0x0F0F0F0F);
                    GA1 = (int)((gq1_lo >> 4) & 0x0F0F0F0F);
                    GA3 = (int)((gq1_hi >> 4) & 0x0F0F0F0F);
                }
                // UP A unpack
                uint32_t uq0_lo = uw0 ? *(const uint32_t *)(uw0->qs + qs_off + 0) : 0;
                uint32_t uq0_hi = uw0 ? *(const uint32_t *)(uw0->qs + qs_off + 4) : 0;
                uint32_t uq1_lo = uw1 ? *(const uint32_t *)(uw1->qs + qs_off + 0) : 0;
                uint32_t uq1_hi = uw1 ? *(const uint32_t *)(uw1->qs + qs_off + 4) : 0;
                int UA0, UA1, UA2, UA3;
                if (ip == 0) {
                    UA0 = (int)(uq0_lo & 0x0F0F0F0F);
                    UA2 = (int)(uq0_hi & 0x0F0F0F0F);
                    UA1 = (int)(uq1_lo & 0x0F0F0F0F);
                    UA3 = (int)(uq1_hi & 0x0F0F0F0F);
                } else {
                    UA0 = (int)((uq0_lo >> 4) & 0x0F0F0F0F);
                    UA2 = (int)((uq0_hi >> 4) & 0x0F0F0F0F);
                    UA1 = (int)((uq1_lo >> 4) & 0x0F0F0F0F);
                    UA3 = (int)((uq1_hi >> 4) & 0x0F0F0F0F);
                }

                int GD0=0, GD1=0, GD2=0, GD3=0, UD0=0, UD1=0, UD2=0, UD3=0;
                asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                    : "+r"(GD0), "+r"(GD1), "+r"(GD2), "+r"(GD3)
                    : "r"(GA0), "r"(GA1), "r"(GA2), "r"(GA3), "r"(B0), "r"(B1));
                asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                    : "+r"(UD0), "+r"(UD1), "+r"(UD2), "+r"(UD3)
                    : "r"(UA0), "r"(UA1), "r"(UA2), "r"(UA3), "r"(B0), "r"(B1));

                const float g_da_r0 = g_dall_sc[r0][s], g_dm_r0 = g_dmin_m[r0][s];
                const float g_da_r1 = g_dall_sc[r1][s], g_dm_r1 = g_dmin_m[r1][s];
                const float u_da_r0 = u_dall_sc[r0][s], u_dm_r0 = u_dmin_m[r0][s];
                const float u_da_r1 = u_dall_sc[r1][s], u_dm_r1 = u_dmin_m[r1][s];

                // g_acc layout: [r0_pair0, r0_pair1, r1_pair0, r1_pair1] for mma_idx=0
                //                [r2_pair0, r2_pair1, r3_pair0, r3_pair1] for mma_idx=1
                const int base = mma_idx * 4;
                g_acc[base + 0] += g_da_r0 * d8_a * (float)GD0 - g_dm_r0 * d8_a * dot_a;
                g_acc[base + 1] += g_da_r0 * d8_b * (float)GD1 - g_dm_r0 * d8_b * dot_b;
                g_acc[base + 2] += g_da_r1 * d8_a * (float)GD2 - g_dm_r1 * d8_a * dot_a;
                g_acc[base + 3] += g_da_r1 * d8_b * (float)GD3 - g_dm_r1 * d8_b * dot_b;
                u_acc[base + 0] += u_da_r0 * d8_a * (float)UD0 - u_dm_r0 * d8_a * dot_a;
                u_acc[base + 1] += u_da_r0 * d8_b * (float)UD1 - u_dm_r0 * d8_b * dot_b;
                u_acc[base + 2] += u_da_r1 * d8_a * (float)UD2 - u_dm_r1 * d8_a * dot_a;
                u_acc[base + 3] += u_da_r1 * d8_b * (float)UD3 - u_dm_r1 * d8_b * dot_b;
            }
        }
    }

    // GELU·mul + scatter — 8 outputs per thread (4 rows × 2 in_pairs).
    const float k0 = 0.7978845608028654f;
    const float k1 = 0.044715f;
    auto do_write = [&](int pair_local, int weight_row, float gv, float uv) {
        const int pair_idx = pair_base + pair_local;
        if (pair_idx >= size_m || weight_row >= N) return;
        const int e = expert_ids[pair_idx];
        if (e != block_expert) return;
        const float gelu = 0.5f * gv * (1.f + tanhf(k0 * (gv + k1 * gv * gv * gv)));
        const int tok = sorted_token_ids[pair_idx];
        dst[(size_t)tok * N + weight_row] = gelu * uv;
    };

    if (vr[0]) {
        do_write(2 * tj + 0, rows[0], g_acc[0], u_acc[0]);
        do_write(2 * tj + 1, rows[0], g_acc[1], u_acc[1]);
    }
    if (vr[1]) {
        do_write(2 * tj + 0, rows[1], g_acc[2], u_acc[2]);
        do_write(2 * tj + 1, rows[1], g_acc[3], u_acc[3]);
    }
    if (vr[2]) {
        do_write(2 * tj + 0, rows[2], g_acc[4], u_acc[4]);
        do_write(2 * tj + 1, rows[2], g_acc[5], u_acc[5]);
    }
    if (vr[3]) {
        do_write(2 * tj + 0, rows[3], g_acc[6], u_acc[6]);
        do_write(2 * tj + 1, rows[3], g_acc[7], u_acc[7]);
    }

    } // end outer expert loop

    (void)num_experts;
}

extern "C" void moe_q4k_imma_m8_m32_gate_up(
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
    using namespace moe_q4k_imma_m8_m32_ns;
    dim3 grid((N      + IM32_M - 1) / IM32_M,
              (size_m + IM32_N - 1) / IM32_N,
              1);
    dim3 blk(32, 1, 1);
    moe_q4k_imma_m8_m32_kernel<<<grid, blk, 0, stream>>>(
        gate_up_w, inputs_q81, sorted_token_ids, expert_ids,
        dst_f32, num_experts, topk, size_m, N, K
    );
}
