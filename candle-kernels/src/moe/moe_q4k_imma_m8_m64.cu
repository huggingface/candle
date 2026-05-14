/**
 * @file moe_q4k_imma_m8_m64.cu
 *
 * 4-warp IMMA M=8: each block has 4 independent warps, each handling a
 * different 16-weight-row group. Block produces 64 weight rows × 8
 * pairs = 512 outputs.
 *
 * Independent warps — no __syncthreads, no shared memory. The extra
 * warps just amortize per-block launch overhead.
 *
 * Grid:
 *   gridDim.x = ceil(N / 64)
 *   gridDim.y = ceil(size_m / 8)
 * blockDim = (32, 4) = 128 threads
 */
#include "gguf.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace moe_q4k_imma_m8_m64_ns {

#define I64_K_SUPER 256
#define I64_M_PER_WARP 16
#define I64_N           8
#define I64_NWARPS      4
#define I64_M_PER_BLOCK (I64_M_PER_WARP * I64_NWARPS)

typedef struct {
    __half2 dm;
    uint8_t scales[12];
    uint8_t qs[128];
} block_q4_K_i64;

typedef struct {
    __half2 ds;
    int8_t  qs[32];
} block_q8_1_i64;

static_assert(sizeof(block_q4_K_i64) == 144, "block_q4_K_i64 size");
static_assert(sizeof(block_q8_1_i64) == 36,  "block_q8_1_i64 size");

__device__ __forceinline__ void i64_get_scale_min_k4(
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

extern "C" __global__ void moe_q4k_imma_m8_m64_kernel(
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
    using namespace moe_q4k_imma_m8_m64_ns;

    const int n_tile = blockIdx.y;
    const int pair_base = n_tile * I64_N;
    if (pair_base >= size_m) return;

    const int m_block_tile = blockIdx.x;
    const int m_block_base = m_block_tile * I64_M_PER_BLOCK;
    if (m_block_base >= N) return;

    const int warp_id = threadIdx.y;
    const int m_base = m_block_base + warp_id * I64_M_PER_WARP;
    if (m_base >= N) return;

    const int lane = threadIdx.x;
    const int g    = lane >> 2;
    const int tj   = lane & 3;
    const int num_super = K / I64_K_SUPER;
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

    const block_q4_K_i64 * w_expert =
        (const block_q4_K_i64 *) gate_up_w
        + (size_t)block_expert * (size_t)two_n * num_super;
    const block_q8_1_i64 * y_expert = nullptr;
    if (v_pair_e) {
        const int real_token = my_pair_tok / topk;
        y_expert = (const block_q8_1_i64 *) inputs_q81
                 + (size_t)real_token * num_super * 8;
    }

    float gate_0 = 0.f, gate_1 = 0.f, gate_2 = 0.f, gate_3 = 0.f;
    float up_0   = 0.f, up_1   = 0.f, up_2   = 0.f, up_3   = 0.f;

    for (int isb = 0; isb < num_super; ++isb) {
        const block_q4_K_i64 * gwa = va ? w_expert + (size_t)row_a    * num_super + isb : nullptr;
        const block_q4_K_i64 * gwb = vb ? w_expert + (size_t)row_b    * num_super + isb : nullptr;
        const block_q4_K_i64 * uwa = va ? w_expert + (size_t)row_a_up * num_super + isb : nullptr;
        const block_q4_K_i64 * uwb = vb ? w_expert + (size_t)row_b_up * num_super + isb : nullptr;

        const float g_dall_a = gwa ? __low2float (gwa->dm) : 0.f;
        const float g_dmin_a = gwa ? __high2float(gwa->dm) : 0.f;
        const float g_dall_b = gwb ? __low2float (gwb->dm) : 0.f;
        const float g_dmin_b = gwb ? __high2float(gwb->dm) : 0.f;
        const float u_dall_a = uwa ? __low2float (uwa->dm) : 0.f;
        const float u_dmin_a = uwa ? __high2float(uwa->dm) : 0.f;
        const float u_dall_b = uwb ? __low2float (uwb->dm) : 0.f;
        const float u_dmin_b = uwb ? __high2float(uwb->dm) : 0.f;

        float g_da_a[8], g_dm_a[8], g_da_b[8], g_dm_b[8];
        float u_da_a[8], u_dm_a[8], u_da_b[8], u_dm_b[8];
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            uint8_t sc, m;
            if (gwa) { i64_get_scale_min_k4(s, gwa->scales, sc, m); g_da_a[s] = g_dall_a * (float)sc; g_dm_a[s] = g_dmin_a * (float)m; }
            else     { g_da_a[s] = 0.f; g_dm_a[s] = 0.f; }
            if (gwb) { i64_get_scale_min_k4(s, gwb->scales, sc, m); g_da_b[s] = g_dall_b * (float)sc; g_dm_b[s] = g_dmin_b * (float)m; }
            else     { g_da_b[s] = 0.f; g_dm_b[s] = 0.f; }
            if (uwa) { i64_get_scale_min_k4(s, uwa->scales, sc, m); u_da_a[s] = u_dall_a * (float)sc; u_dm_a[s] = u_dmin_a * (float)m; }
            else     { u_da_a[s] = 0.f; u_dm_a[s] = 0.f; }
            if (uwb) { i64_get_scale_min_k4(s, uwb->scales, sc, m); u_da_b[s] = u_dall_b * (float)sc; u_dm_b[s] = u_dmin_b * (float)m; }
            else     { u_da_b[s] = 0.f; u_dm_b[s] = 0.f; }
        }

        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int il = s >> 1;
            const int ip = s & 1;
            const int qs_off = 32 * il + 8 * tj;

            uint32_t gqa_lo = gwa ? __ldg((const uint32_t *)(gwa->qs + qs_off + 0)) : 0;
            uint32_t gqa_hi = gwa ? __ldg((const uint32_t *)(gwa->qs + qs_off + 4)) : 0;
            uint32_t gqb_lo = gwb ? __ldg((const uint32_t *)(gwb->qs + qs_off + 0)) : 0;
            uint32_t gqb_hi = gwb ? __ldg((const uint32_t *)(gwb->qs + qs_off + 4)) : 0;
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
            uint32_t uqa_lo = uwa ? __ldg((const uint32_t *)(uwa->qs + qs_off + 0)) : 0;
            uint32_t uqa_hi = uwa ? __ldg((const uint32_t *)(uwa->qs + qs_off + 4)) : 0;
            uint32_t uqb_lo = uwb ? __ldg((const uint32_t *)(uwb->qs + qs_off + 0)) : 0;
            uint32_t uqb_hi = uwb ? __ldg((const uint32_t *)(uwb->qs + qs_off + 4)) : 0;
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

            const block_q8_1_i64 * yb_my = y_expert ? y_expert + isb * 8 + s : nullptr;
            int B0 = yb_my ? ((const int *)yb_my->qs)[2 * tj + 0] : 0;
            int B1 = yb_my ? ((const int *)yb_my->qs)[2 * tj + 1] : 0;
            const float d8_my    = yb_my ? __low2float (yb_my->ds) : 0.f;
            const float sumxd_my = yb_my ? __high2float(yb_my->ds) : 0.f;

            int GD0=0, GD1=0, GD2=0, GD3=0, UD0=0, UD1=0, UD2=0, UD3=0;
            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+r"(GD0), "+r"(GD1), "+r"(GD2), "+r"(GD3)
                : "r"(GA0), "r"(GA1), "r"(GA2), "r"(GA3), "r"(B0), "r"(B1));
            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+r"(UD0), "+r"(UD1), "+r"(UD2), "+r"(UD3)
                : "r"(UA0), "r"(UA1), "r"(UA2), "r"(UA3), "r"(B0), "r"(B1));

            const int src_lane_a = (2 * tj + 0) * 4;
            const int src_lane_b = (2 * tj + 1) * 4;
            const float d8_a    = __shfl_sync(0xffffffff, d8_my,    src_lane_a);
            const float d8_b    = __shfl_sync(0xffffffff, d8_my,    src_lane_b);
            const float sumxd_a = __shfl_sync(0xffffffff, sumxd_my, src_lane_a);
            const float sumxd_b = __shfl_sync(0xffffffff, sumxd_my, src_lane_b);

            const float gda_a = g_da_a[s], gdm_a = g_dm_a[s];
            const float gda_b = g_da_b[s], gdm_b = g_dm_b[s];
            const float uda_a = u_da_a[s], udm_a = u_dm_a[s];
            const float uda_b = u_da_b[s], udm_b = u_dm_b[s];

            gate_0 += gda_a * d8_a * (float)GD0 - gdm_a * sumxd_a;
            gate_1 += gda_a * d8_b * (float)GD1 - gdm_a * sumxd_b;
            gate_2 += gda_b * d8_a * (float)GD2 - gdm_b * sumxd_a;
            gate_3 += gda_b * d8_b * (float)GD3 - gdm_b * sumxd_b;
            up_0   += uda_a * d8_a * (float)UD0 - udm_a * sumxd_a;
            up_1   += uda_a * d8_b * (float)UD1 - udm_a * sumxd_b;
            up_2   += uda_b * d8_a * (float)UD2 - udm_b * sumxd_a;
            up_3   += uda_b * d8_b * (float)UD3 - udm_b * sumxd_b;
        }
    }

    const float k0 = 0.7978845608028654f;
    const float k1 = 0.044715f;
    const int src_lane_a_e = (2 * tj + 0) * 4;
    const int src_lane_b_e = (2 * tj + 1) * 4;
    const int exp_a = __shfl_sync(0xffffffff, my_expert,   src_lane_a_e);
    const int exp_b = __shfl_sync(0xffffffff, my_expert,   src_lane_b_e);
    const int tok_a = __shfl_sync(0xffffffff, my_pair_tok, src_lane_a_e);
    const int tok_b = __shfl_sync(0xffffffff, my_pair_tok, src_lane_b_e);

    auto do_write = [&](int pair_local, int weight_row, float gv, float uv,
                        int e_cached, int tok_cached) {
        const int pair_idx = pair_base + pair_local;
        if (pair_idx >= size_m || weight_row >= N) return;
        if (e_cached != block_expert) return;
        const float gelu = 0.5f * gv * (1.f + tanhf(k0 * (gv + k1 * gv * gv * gv)));
        dst[(size_t)tok_cached * N + weight_row] = gelu * uv;
    };

    if (va) {
        do_write(2 * tj + 0, row_a, gate_0, up_0, exp_a, tok_a);
        do_write(2 * tj + 1, row_a, gate_1, up_1, exp_b, tok_b);
    }
    if (vb) {
        do_write(2 * tj + 0, row_b, gate_2, up_2, exp_a, tok_a);
        do_write(2 * tj + 1, row_b, gate_3, up_3, exp_b, tok_b);
    }

    } // end expert loop

    (void)num_experts;
}

extern "C" void moe_q4k_imma_m8_m64_gate_up(
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
    using namespace moe_q4k_imma_m8_m64_ns;
    dim3 grid((N      + I64_M_PER_BLOCK - 1) / I64_M_PER_BLOCK,
              (size_m + I64_N           - 1) / I64_N,
              1);
    dim3 blk(32, I64_NWARPS, 1);
    moe_q4k_imma_m8_m64_kernel<<<grid, blk, 0, stream>>>(
        gate_up_w, inputs_q81, sorted_token_ids, expert_ids,
        dst_f32, num_experts, topk, size_m, N, K
    );
}
