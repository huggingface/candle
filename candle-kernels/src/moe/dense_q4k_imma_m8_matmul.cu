/**
 * @file dense_q4k_imma_m8_matmul.cu
 *
 * Plain dense Q4_K × Q8_1 IMMA M=8 matmul (no gate/up split, no
 * activation, no scatter). Drop-in fast path for QMatMul::forward on
 * Q4_K weights when input has ≥8 rows and K % 256 == 0.
 *
 * Use case: attention QKV projection, out_proj, embedding output proj —
 * anywhere a Q4_K matmul is called on multi-row inputs.
 *
 * Output: F32 [size_m, N].
 *
 * Grid:
 *   gridDim.x = ceil(N / 16)
 *   gridDim.y = ceil(size_m / 8)
 * blockDim = 32 (1 warp)
 */
#include "gguf.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace dense_q4k_imma_m8_matmul_ns {

#define DMM_K_SUPER 256
#define DMM_M       16
#define DMM_N        8

typedef struct {
    __half2 dm;
    uint8_t scales[12];
    uint8_t qs[128];
} block_q4_K_dmm;

typedef struct {
    __half2 ds;
    int8_t  qs[32];
} block_q8_1_dmm;

static_assert(sizeof(block_q4_K_dmm) == 144, "block_q4_K_dmm size");
static_assert(sizeof(block_q8_1_dmm) == 36,  "block_q8_1_dmm size");

__device__ __forceinline__ void dmm_get_scale_min_k4(
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

extern "C" __global__ void dense_q4k_imma_m8_matmul_kernel(
    const void * __restrict__ w,                 // [N, K] Q4_K
    const void * __restrict__ inputs_q81,        // [size_m, K/32] Q8_1
    float * __restrict__ dst,                    // [size_m, N] F32
    int size_m,
    int N,
    int K
) {
    using namespace dense_q4k_imma_m8_matmul_ns;

    const int n_tile = blockIdx.y;
    const int tok_base = n_tile * DMM_N;
    if (tok_base >= size_m) return;

    const int m_tile = blockIdx.x;
    const int m_base = m_tile * DMM_M;
    if (m_base >= N) return;

    const int lane = threadIdx.x;
    const int g    = lane >> 2;
    const int tj   = lane & 3;
    const int num_super = K / DMM_K_SUPER;

    const int row_a = m_base + g;
    const int row_b = m_base + g + 8;
    const bool va = row_a < N;
    const bool vb = row_b < N;

    const int my_tok = tok_base + g;
    const bool v_tok = my_tok < size_m;

    const block_q4_K_dmm * w_base = (const block_q4_K_dmm *) w;
    const block_q8_1_dmm * y_base = v_tok
        ? (const block_q8_1_dmm *) inputs_q81 + (size_t)my_tok * num_super * 8
        : nullptr;

    float out_0 = 0.f, out_1 = 0.f, out_2 = 0.f, out_3 = 0.f;

    for (int isb = 0; isb < num_super; ++isb) {
        const block_q4_K_dmm * wa_sb = va ? w_base + (size_t)row_a * num_super + isb : nullptr;
        const block_q4_K_dmm * wb_sb = vb ? w_base + (size_t)row_b * num_super + isb : nullptr;

        const float dall_a = wa_sb ? __low2float (wa_sb->dm) : 0.f;
        const float dmin_a = wa_sb ? __high2float(wa_sb->dm) : 0.f;
        const float dall_b = wb_sb ? __low2float (wb_sb->dm) : 0.f;
        const float dmin_b = wb_sb ? __high2float(wb_sb->dm) : 0.f;

        float dall_sc_a[8], dmin_m_a[8], dall_sc_b[8], dmin_m_b[8];
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            uint8_t sc, m;
            if (wa_sb) { dmm_get_scale_min_k4(s, wa_sb->scales, sc, m); dall_sc_a[s] = dall_a * (float)sc; dmin_m_a[s] = dmin_a * (float)m; }
            else       { dall_sc_a[s] = 0.f; dmin_m_a[s] = 0.f; }
            if (wb_sb) { dmm_get_scale_min_k4(s, wb_sb->scales, sc, m); dall_sc_b[s] = dall_b * (float)sc; dmin_m_b[s] = dmin_b * (float)m; }
            else       { dall_sc_b[s] = 0.f; dmin_m_b[s] = 0.f; }
        }

        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int il = s >> 1;
            const int ip = s & 1;
            const int qs_off = 32 * il + 8 * tj;

            uint32_t qa_lo = wa_sb ? __ldg((const uint32_t *)(wa_sb->qs + qs_off + 0)) : 0;
            uint32_t qa_hi = wa_sb ? __ldg((const uint32_t *)(wa_sb->qs + qs_off + 4)) : 0;
            uint32_t qb_lo = wb_sb ? __ldg((const uint32_t *)(wb_sb->qs + qs_off + 0)) : 0;
            uint32_t qb_hi = wb_sb ? __ldg((const uint32_t *)(wb_sb->qs + qs_off + 4)) : 0;

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

            const block_q8_1_dmm * yb_my = y_base ? y_base + isb * 8 + s : nullptr;
            int B0 = yb_my ? ((const int *)yb_my->qs)[2 * tj + 0] : 0;
            int B1 = yb_my ? ((const int *)yb_my->qs)[2 * tj + 1] : 0;
            const float d8_my    = yb_my ? __low2float (yb_my->ds) : 0.f;
            const float sumxd_my = yb_my ? __high2float(yb_my->ds) : 0.f;

            int D0 = 0, D1 = 0, D2 = 0, D3 = 0;
            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+r"(D0), "+r"(D1), "+r"(D2), "+r"(D3)
                : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1));

            const int src_lane_a = (2 * tj + 0) * 4;
            const int src_lane_b = (2 * tj + 1) * 4;
            const float d8_a    = __shfl_sync(0xffffffff, d8_my,    src_lane_a);
            const float d8_b    = __shfl_sync(0xffffffff, d8_my,    src_lane_b);
            const float sumxd_a = __shfl_sync(0xffffffff, sumxd_my, src_lane_a);
            const float sumxd_b = __shfl_sync(0xffffffff, sumxd_my, src_lane_b);

            const float da_a = dall_sc_a[s], dm_a = dmin_m_a[s];
            const float da_b = dall_sc_b[s], dm_b = dmin_m_b[s];

            out_0 += da_a * d8_a * (float)D0 - dm_a * sumxd_a;
            out_1 += da_a * d8_b * (float)D1 - dm_a * sumxd_b;
            out_2 += da_b * d8_a * (float)D2 - dm_b * sumxd_a;
            out_3 += da_b * d8_b * (float)D3 - dm_b * sumxd_b;
        }
    }

    const int tok_a = tok_base + 2 * tj + 0;
    const int tok_b = tok_base + 2 * tj + 1;

    if (va) {
        if (tok_a < size_m) dst[(size_t)tok_a * N + row_a] = out_0;
        if (tok_b < size_m) dst[(size_t)tok_b * N + row_a] = out_1;
    }
    if (vb) {
        if (tok_a < size_m) dst[(size_t)tok_a * N + row_b] = out_2;
        if (tok_b < size_m) dst[(size_t)tok_b * N + row_b] = out_3;
    }
}

extern "C" void dense_q4k_imma_m8_matmul(
    const void * w,
    const void * inputs_q81,
    float * dst_f32,
    int size_m,
    int N,
    int K,
    cudaStream_t stream
) {
    if (size_m <= 0 || N <= 0 || K <= 0) return;
    using namespace dense_q4k_imma_m8_matmul_ns;
    dim3 grid((N      + DMM_M - 1) / DMM_M,
              (size_m + DMM_N - 1) / DMM_N,
              1);
    dim3 blk(32, 1, 1);
    dense_q4k_imma_m8_matmul_kernel<<<grid, blk, 0, stream>>>(
        w, inputs_q81, dst_f32, size_m, N, K
    );
}
