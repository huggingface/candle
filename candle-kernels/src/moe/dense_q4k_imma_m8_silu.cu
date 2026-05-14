/**
 * @file dense_q4k_imma_m8_silu.cu
 *
 * IMMA M=8 for DENSE Q4_K gate+up*silu matmul (qwen3, deepcoder,
 * devstral, llama-family). Same per-pair-tile MMA pattern as
 * moe_q4k_imma_m8 but:
 *   - Gate and up are SEPARATE Q4_K tensors (not concatenated)
 *   - No expert lookup (num_experts=1)
 *   - silu(gate) * up activation instead of gelu·mul
 *
 * Input pre-quantized to Q8_1 per token. Each block produces 16 weight
 * rows × 8 consecutive tokens.
 *
 * Grid:
 *   gridDim.x = ceil(N / 16)
 *   gridDim.y = ceil(size_m / 8)
 * blockDim = 32
 */
#include "gguf.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace dense_q4k_imma_m8_silu_ns {

#define DSL_K_SUPER 256
#define DSL_M       16
#define DSL_N        8

typedef struct {
    __half2 dm;
    uint8_t scales[12];
    uint8_t qs[128];
} block_q4_K_dsl;

typedef struct {
    __half2 ds;
    int8_t  qs[32];
} block_q8_1_dsl;

static_assert(sizeof(block_q4_K_dsl) == 144, "block_q4_K_dsl size");
static_assert(sizeof(block_q8_1_dsl) == 36,  "block_q8_1_dsl size");

__device__ __forceinline__ void dsl_get_scale_min_k4(
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

extern "C" __global__ void dense_q4k_imma_m8_silu_kernel(
    const void * __restrict__ gate_w,            // [N, K] Q4_K
    const void * __restrict__ up_w,              // [N, K] Q4_K
    const void * __restrict__ inputs_q81,        // [size_m, K/32] Q8_1
    float * __restrict__ dst,                    // [size_m, N] F32
    int size_m,
    int N,
    int K
) {
    using namespace dense_q4k_imma_m8_silu_ns;

    const int n_tile = blockIdx.y;
    const int tok_base = n_tile * DSL_N;
    if (tok_base >= size_m) return;

    const int m_tile = blockIdx.x;
    const int m_base = m_tile * DSL_M;
    if (m_base >= N) return;

    const int lane = threadIdx.x;
    const int g    = lane >> 2;
    const int tj   = lane & 3;
    const int num_super = K / DSL_K_SUPER;

    const int row_a = m_base + g;
    const int row_b = m_base + g + 8;
    const bool va = row_a < N;
    const bool vb = row_b < N;

    const int my_tok = tok_base + g;
    const bool v_tok = my_tok < size_m;

    const block_q4_K_dsl * gate_base = (const block_q4_K_dsl *) gate_w;
    const block_q4_K_dsl * up_base   = (const block_q4_K_dsl *) up_w;
    const block_q8_1_dsl * y_base    = v_tok
        ? (const block_q8_1_dsl *) inputs_q81 + (size_t)my_tok * num_super * 8
        : nullptr;

    float gate_0 = 0.f, gate_1 = 0.f, gate_2 = 0.f, gate_3 = 0.f;
    float up_0   = 0.f, up_1   = 0.f, up_2   = 0.f, up_3   = 0.f;

    for (int isb = 0; isb < num_super; ++isb) {
        const block_q4_K_dsl * gwa = va ? gate_base + (size_t)row_a * num_super + isb : nullptr;
        const block_q4_K_dsl * gwb = vb ? gate_base + (size_t)row_b * num_super + isb : nullptr;
        const block_q4_K_dsl * uwa = va ? up_base   + (size_t)row_a * num_super + isb : nullptr;
        const block_q4_K_dsl * uwb = vb ? up_base   + (size_t)row_b * num_super + isb : nullptr;

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
            if (gwa) { dsl_get_scale_min_k4(s, gwa->scales, sc, m); g_da_a[s] = g_dall_a * (float)sc; g_dm_a[s] = g_dmin_a * (float)m; }
            else     { g_da_a[s] = 0.f; g_dm_a[s] = 0.f; }
            if (gwb) { dsl_get_scale_min_k4(s, gwb->scales, sc, m); g_da_b[s] = g_dall_b * (float)sc; g_dm_b[s] = g_dmin_b * (float)m; }
            else     { g_da_b[s] = 0.f; g_dm_b[s] = 0.f; }
            if (uwa) { dsl_get_scale_min_k4(s, uwa->scales, sc, m); u_da_a[s] = u_dall_a * (float)sc; u_dm_a[s] = u_dmin_a * (float)m; }
            else     { u_da_a[s] = 0.f; u_dm_a[s] = 0.f; }
            if (uwb) { dsl_get_scale_min_k4(s, uwb->scales, sc, m); u_da_b[s] = u_dall_b * (float)sc; u_dm_b[s] = u_dmin_b * (float)m; }
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

            const block_q8_1_dsl * yb_my = y_base ? y_base + isb * 8 + s : nullptr;
            int B0 = yb_my ? ((const int *)yb_my->qs)[2 * tj + 0] : 0;
            int B1 = yb_my ? ((const int *)yb_my->qs)[2 * tj + 1] : 0;
            const float d8_my = yb_my ? __low2float(yb_my->ds) : 0.f;

            int GD0=0, GD1=0, GD2=0, GD3=0, UD0=0, UD1=0, UD2=0, UD3=0;
            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+r"(GD0), "+r"(GD1), "+r"(GD2), "+r"(GD3)
                : "r"(GA0), "r"(GA1), "r"(GA2), "r"(GA3), "r"(B0), "r"(B1));
            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+r"(UD0), "+r"(UD1), "+r"(UD2), "+r"(UD3)
                : "r"(UA0), "r"(UA1), "r"(UA2), "r"(UA3), "r"(B0), "r"(B1));

            int dot2 = __dp4a(0x01010101, B1, __dp4a(0x01010101, B0, 0));
            dot2 += __shfl_xor_sync(0xffffffff, dot2, 1);
            dot2 += __shfl_xor_sync(0xffffffff, dot2, 2);

            const int src_lane_a = (2 * tj + 0) * 4;
            const int src_lane_b = (2 * tj + 1) * 4;
            const float d8_a  = __shfl_sync(0xffffffff, d8_my, src_lane_a);
            const float d8_b  = __shfl_sync(0xffffffff, d8_my, src_lane_b);
            const float dot_a = (float)__shfl_sync(0xffffffff, dot2, src_lane_a);
            const float dot_b = (float)__shfl_sync(0xffffffff, dot2, src_lane_b);

            const float gda_a = g_da_a[s], gdm_a = g_dm_a[s];
            const float gda_b = g_da_b[s], gdm_b = g_dm_b[s];
            const float uda_a = u_da_a[s], udm_a = u_dm_a[s];
            const float uda_b = u_da_b[s], udm_b = u_dm_b[s];

            gate_0 += gda_a * d8_a * (float)GD0 - gdm_a * d8_a * dot_a;
            gate_1 += gda_a * d8_b * (float)GD1 - gdm_a * d8_b * dot_b;
            gate_2 += gda_b * d8_a * (float)GD2 - gdm_b * d8_a * dot_a;
            gate_3 += gda_b * d8_b * (float)GD3 - gdm_b * d8_b * dot_b;
            up_0   += uda_a * d8_a * (float)UD0 - udm_a * d8_a * dot_a;
            up_1   += uda_a * d8_b * (float)UD1 - udm_a * d8_b * dot_b;
            up_2   += uda_b * d8_a * (float)UD2 - udm_b * d8_a * dot_a;
            up_3   += uda_b * d8_b * (float)UD3 - udm_b * d8_b * dot_b;
        }
    }

    const int tok_a = tok_base + 2 * tj + 0;
    const int tok_b = tok_base + 2 * tj + 1;

    auto silu_mul_write = [&](int tok, int weight_row, float gv, float uv) {
        if (tok >= size_m || weight_row >= N) return;
        // SiLU(g) = g / (1 + exp(-g))
        const float silu = gv / (1.f + __expf(-gv));
        dst[(size_t)tok * N + weight_row] = silu * uv;
    };

    if (va) {
        silu_mul_write(tok_a, row_a, gate_0, up_0);
        silu_mul_write(tok_b, row_a, gate_1, up_1);
    }
    if (vb) {
        silu_mul_write(tok_a, row_b, gate_2, up_2);
        silu_mul_write(tok_b, row_b, gate_3, up_3);
    }
}

extern "C" void dense_q4k_imma_m8_silu(
    const void * gate_w,
    const void * up_w,
    const void * inputs_q81,
    float * dst_f32,
    int size_m,
    int N,
    int K,
    cudaStream_t stream
) {
    if (size_m <= 0 || N <= 0 || K <= 0) return;
    using namespace dense_q4k_imma_m8_silu_ns;
    dim3 grid((N      + DSL_M - 1) / DSL_M,
              (size_m + DSL_N - 1) / DSL_N,
              1);
    dim3 blk(32, 1, 1);
    dense_q4k_imma_m8_silu_kernel<<<grid, blk, 0, stream>>>(
        gate_w, up_w, inputs_q81, dst_f32, size_m, N, K
    );
}
