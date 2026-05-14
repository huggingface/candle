/**
 * @file moe_q4k_imma_m8_mw.cu
 *
 * Multi-warp extension of the IMMA M=8 kernel. Each block has 4 warps;
 * all 4 warps share the SAME 8 sorted pairs (n_tile) but cover
 * DIFFERENT 16-row weight slices. Block produces 64 weight rows × 8
 * pairs = 512 outputs.
 *
 * Hypothesis: at 2622 prompt-tok/s the IMMA M=8 is already at ~65% of
 * HBM peak. Reducing kernel-launch overhead (4× fewer blocks for same
 * compute) should yield a few % more without hitting the memory wall.
 *
 * Grid:
 *   gridDim.x = ceil(N / 64)        (vs 16 in single-warp variant)
 *   gridDim.y = ceil(size_m / 8)
 * blockDim = (32, 4)
 */
#include "gguf.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace moe_q4k_imma_m8_mw_ns {

#define MW_K_SUPER 256
#define MW_M_PER_WARP 16
#define MW_N           8
#define MW_NWARPS      4
#define MW_M_PER_BLOCK (MW_M_PER_WARP * MW_NWARPS)

typedef struct {
    __half2 dm;
    uint8_t scales[12];
    uint8_t qs[128];
} block_q4_K_mw;

typedef struct {
    __half2 ds;
    int8_t  qs[32];
} block_q8_1_mw;

static_assert(sizeof(block_q4_K_mw) == 144, "block_q4_K_mw size");
static_assert(sizeof(block_q8_1_mw) == 36,  "block_q8_1_mw size");

__device__ __forceinline__ void mw_get_scale_min_k4(
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

extern "C" __global__ void moe_q4k_imma_m8_mw_kernel(
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
    using namespace moe_q4k_imma_m8_mw_ns;

    const int n_tile = blockIdx.y;
    const int pair_base = n_tile * MW_N;
    if (pair_base >= size_m) return;

    const int m_block_tile = blockIdx.x;
    const int m_block_base = m_block_tile * MW_M_PER_BLOCK;
    if (m_block_base >= N) return;

    const int warp_id = threadIdx.y;
    const int m_base = m_block_base + warp_id * MW_M_PER_WARP;
    if (m_base >= N) return;

    const int lane = threadIdx.x;
    const int g    = lane >> 2;
    const int tj   = lane & 3;
    const int num_super = K / MW_K_SUPER;
    const int two_n = N << 1;

    const int row_a = m_base + g;
    const int row_b = m_base + g + 8;
    const bool va = row_a < N;
    const bool vb = row_b < N;
    const int row_a_up = N + row_a;
    const int row_b_up = N + row_b;

    const int my_pair = pair_base + g;
    const bool v_pair = my_pair < size_m;
    int my_pair_tok = 0;
    int my_expert   = -1;
    if (v_pair) {
        my_pair_tok = sorted_token_ids[my_pair];
        my_expert   = expert_ids[my_pair];
    }
    const int block_expert = (pair_base + 0 < size_m) ? expert_ids[pair_base + 0] : -1;
    if (block_expert < 0 || block_expert >= num_experts) return;
    const bool v_pair_e = v_pair && (my_expert == block_expert);

    const block_q4_K_mw * w_expert =
        (const block_q4_K_mw *) gate_up_w
        + (size_t)block_expert * (size_t)two_n * num_super;

    const block_q8_1_mw * y_expert = nullptr;
    if (v_pair_e) {
        const int real_token = my_pair_tok / topk;
        y_expert = (const block_q8_1_mw *) inputs_q81
                 + (size_t)real_token * num_super * 8;
    }

    float gate_0 = 0.f, gate_1 = 0.f, gate_2 = 0.f, gate_3 = 0.f;
    float up_0   = 0.f, up_1   = 0.f, up_2   = 0.f, up_3   = 0.f;

    for (int isb = 0; isb < num_super; ++isb) {
        const block_q4_K_mw * gwa = va ? w_expert + (size_t)row_a    * num_super + isb : nullptr;
        const block_q4_K_mw * gwb = vb ? w_expert + (size_t)row_b    * num_super + isb : nullptr;
        const block_q4_K_mw * uwa = va ? w_expert + (size_t)row_a_up * num_super + isb : nullptr;
        const block_q4_K_mw * uwb = vb ? w_expert + (size_t)row_b_up * num_super + isb : nullptr;

        const float g_dall_a = gwa ? __low2float (gwa->dm) : 0.f;
        const float g_dmin_a = gwa ? __high2float(gwa->dm) : 0.f;
        const float g_dall_b = gwb ? __low2float (gwb->dm) : 0.f;
        const float g_dmin_b = gwb ? __high2float(gwb->dm) : 0.f;
        const float u_dall_a = uwa ? __low2float (uwa->dm) : 0.f;
        const float u_dmin_a = uwa ? __high2float(uwa->dm) : 0.f;
        const float u_dall_b = uwb ? __low2float (uwb->dm) : 0.f;
        const float u_dmin_b = uwb ? __high2float(uwb->dm) : 0.f;

        float g_sd_0 = 0.f, g_sd_1 = 0.f, g_sd_2 = 0.f, g_sd_3 = 0.f;
        float g_sm_0 = 0.f, g_sm_1 = 0.f, g_sm_2 = 0.f, g_sm_3 = 0.f;
        float u_sd_0 = 0.f, u_sd_1 = 0.f, u_sd_2 = 0.f, u_sd_3 = 0.f;
        float u_sm_0 = 0.f, u_sm_1 = 0.f, u_sm_2 = 0.f, u_sm_3 = 0.f;

        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int il = s >> 1;
            const int ip = s & 1;
            const int qs_off = 32 * il + 8 * tj;

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

            const block_q8_1_mw * yb_my = y_expert ? y_expert + isb * 8 + s : nullptr;
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

            uint8_t g_sc_a, g_m_a, g_sc_b, g_m_b;
            uint8_t u_sc_a, u_m_a, u_sc_b, u_m_b;
            if (gwa) mw_get_scale_min_k4(s, gwa->scales, g_sc_a, g_m_a); else { g_sc_a = g_m_a = 0; }
            if (gwb) mw_get_scale_min_k4(s, gwb->scales, g_sc_b, g_m_b); else { g_sc_b = g_m_b = 0; }
            if (uwa) mw_get_scale_min_k4(s, uwa->scales, u_sc_a, u_m_a); else { u_sc_a = u_m_a = 0; }
            if (uwb) mw_get_scale_min_k4(s, uwb->scales, u_sc_b, u_m_b); else { u_sc_b = u_m_b = 0; }

            g_sd_0 += d8_out_a * (float)GD0 * (float)g_sc_a;
            g_sd_1 += d8_out_b * (float)GD1 * (float)g_sc_a;
            g_sd_2 += d8_out_a * (float)GD2 * (float)g_sc_b;
            g_sd_3 += d8_out_b * (float)GD3 * (float)g_sc_b;
            g_sm_0 += d8_out_a * dot_out_a * (float)g_m_a;
            g_sm_1 += d8_out_b * dot_out_b * (float)g_m_a;
            g_sm_2 += d8_out_a * dot_out_a * (float)g_m_b;
            g_sm_3 += d8_out_b * dot_out_b * (float)g_m_b;

            u_sd_0 += d8_out_a * (float)UD0 * (float)u_sc_a;
            u_sd_1 += d8_out_b * (float)UD1 * (float)u_sc_a;
            u_sd_2 += d8_out_a * (float)UD2 * (float)u_sc_b;
            u_sd_3 += d8_out_b * (float)UD3 * (float)u_sc_b;
            u_sm_0 += d8_out_a * dot_out_a * (float)u_m_a;
            u_sm_1 += d8_out_b * dot_out_b * (float)u_m_a;
            u_sm_2 += d8_out_a * dot_out_a * (float)u_m_b;
            u_sm_3 += d8_out_b * dot_out_b * (float)u_m_b;
        }

        gate_0 += g_dall_a * g_sd_0 - g_dmin_a * g_sm_0;
        gate_1 += g_dall_a * g_sd_1 - g_dmin_a * g_sm_1;
        gate_2 += g_dall_b * g_sd_2 - g_dmin_b * g_sm_2;
        gate_3 += g_dall_b * g_sd_3 - g_dmin_b * g_sm_3;
        up_0   += u_dall_a * u_sd_0 - u_dmin_a * u_sm_0;
        up_1   += u_dall_a * u_sd_1 - u_dmin_a * u_sm_1;
        up_2   += u_dall_b * u_sd_2 - u_dmin_b * u_sm_2;
        up_3   += u_dall_b * u_sd_3 - u_dmin_b * u_sm_3;
    }

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

    if (va) {
        do_write(2 * tj + 0, row_a, gate_0, up_0);
        do_write(2 * tj + 1, row_a, gate_1, up_1);
    }
    if (vb) {
        do_write(2 * tj + 0, row_b, gate_2, up_2);
        do_write(2 * tj + 1, row_b, gate_3, up_3);
    }

    (void)num_experts;
}

extern "C" void moe_q4k_imma_m8_mw_gate_up(
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
    using namespace moe_q4k_imma_m8_mw_ns;
    dim3 grid((N      + MW_M_PER_BLOCK - 1) / MW_M_PER_BLOCK,
              (size_m + MW_N           - 1) / MW_N,
              1);
    dim3 blk(32, MW_NWARPS, 1);
    moe_q4k_imma_m8_mw_kernel<<<grid, blk, 0, stream>>>(
        gate_up_w, inputs_q81, sorted_token_ids, expert_ids,
        dst_f32, num_experts, topk, size_m, N, K
    );
}
