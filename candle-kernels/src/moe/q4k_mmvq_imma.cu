/**
 * @brief Tensor-core Q4_K × Q8_1 mul-mat-vec for batch=1 decode.
 *
 * Replaces the dp4a inner loop in the existing Q4_K MMVQ with a single
 * mma.sync.aligned.m16n8k32 IMMA per 32-K chunk, processing 16 weight
 * rows in parallel.
 *
 * Layout per Q4_K super-block (256 quants, 8 sub-blocks of 32):
 *   A: 16 weight rows × 32 K (s8 dequant from 4-bit), per IMMA
 *   B: 32 K of q8_1 input (broadcast across 8 N — N>0 wasted)
 *   D: 16 × 8 s32 (only col 0 used)
 *
 * Per-thread fragment mapping (g = lane/4, tj = lane%4):
 *   A0: row g  , K-cols 8tj+0..8tj+3   (packed int = 4 s8)
 *   A1: row g+8, K-cols 8tj+0..8tj+3
 *   A2: row g  , K-cols 8tj+4..8tj+7
 *   A3: row g+8, K-cols 8tj+4..8tj+7
 *   B0: K-rows 2tj+0..2tj+1?, col g — see test_mma packing
 *   D0: row g  , col 2tj+0 (only valid when tj==0 → col 0)
 *   D2: row g+8, col 2tj+0
 *
 * Q4_K layout: per row, qs[128] holds 256 quants as 4-bit nibbles.
 * Sub-block s (0..7): il=s/2, ip=s&1. Bytes qs[32*il + 0..31], nibble
 * selected by ip (low/high). scales[12] holds 8×(6-bit scale, 6-bit min).
 *
 * Reference dp4a impl: vec_dot_q4_K_q8_1 in gguf.cuh:1038.
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace ll_q4k_imma {

#define QK_K 256
#define QK8_1 32

typedef struct {
    half2 dm;
    uint8_t scales[12];
    uint8_t qs[128];
} block_q4_K;

typedef struct {
    half2 ds;
    int8_t qs[32];
} block_q8_1;

static_assert(sizeof(block_q4_K) == 4 + 12 + 128, "block_q4_K size");
static_assert(sizeof(block_q8_1) == 4 + 32, "block_q8_1 size");

__device__ __forceinline__ void get_scale_min_k4_dev(
    int j, const uint8_t * q, uint8_t & d, uint8_t & m
) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// One block = one warp = 16 weight rows × 1 input col.
// gridDim.x = ceil(nrows_x / 16), blockDim.x = 32.
__global__ void q4k_mmvq_imma_kernel(
    const block_q4_K * __restrict__ vx,   // [nrows_x × num_super]
    const block_q8_1 * __restrict__ vy,   // [num_super × 8]
    float * __restrict__ dst,             // [nrows_x]
    int num_super,
    int nrows_x
) {
    constexpr int ROW_GROUP = 16;
    const int row0 = blockIdx.x * ROW_GROUP;
    if (row0 >= nrows_x) return;

    const int lane = threadIdx.x;
    const int g  = lane >> 2;       // 0..7
    const int tj = lane & 3;        // 0..3

    const int row_g  = row0 + g;
    const int row_g8 = row0 + g + 8;
    const bool valid_g  = row_g  < nrows_x;
    const bool valid_g8 = row_g8 < nrows_x;

    float result_lo = 0.f, result_hi = 0.f;

    const block_q4_K * vx_g  = vx + row_g  * num_super;
    const block_q4_K * vx_g8 = vx + row_g8 * num_super;

    for (int isb = 0; isb < num_super; ++isb) {
        const block_q4_K * x_g  = vx_g  + isb;
        const block_q4_K * x_g8 = vx_g8 + isb;
        const block_q8_1 * yb_base = vy + isb * 8;

        const float dall_g  = valid_g  ? __low2float (x_g->dm) : 0.f;
        const float dmin_g  = valid_g  ? __high2float(x_g->dm) : 0.f;
        const float dall_g8 = valid_g8 ? __low2float (x_g8->dm) : 0.f;
        const float dmin_g8 = valid_g8 ? __high2float(x_g8->dm) : 0.f;

        float sf_d_lo = 0.f, sf_m_lo = 0.f;
        float sf_d_hi = 0.f, sf_m_hi = 0.f;

        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int il = s >> 1;
            const int ip = s & 1;
            const int qs_off = 32 * il + 8 * tj;

            // Read 8 quant bytes per row (2 ints)
            uint32_t q_g_lo = valid_g  ? *(const uint32_t *)(x_g->qs  + qs_off + 0) : 0;
            uint32_t q_g_hi = valid_g  ? *(const uint32_t *)(x_g->qs  + qs_off + 4) : 0;
            uint32_t q_g8_lo = valid_g8 ? *(const uint32_t *)(x_g8->qs + qs_off + 0) : 0;
            uint32_t q_g8_hi = valid_g8 ? *(const uint32_t *)(x_g8->qs + qs_off + 4) : 0;

            int A0, A1, A2, A3;
            if (ip == 0) {
                A0 = (int)(q_g_lo  & 0x0F0F0F0F);
                A2 = (int)(q_g_hi  & 0x0F0F0F0F);
                A1 = (int)(q_g8_lo & 0x0F0F0F0F);
                A3 = (int)(q_g8_hi & 0x0F0F0F0F);
            } else {
                A0 = (int)((q_g_lo  >> 4) & 0x0F0F0F0F);
                A2 = (int)((q_g_hi  >> 4) & 0x0F0F0F0F);
                A1 = (int)((q_g8_lo >> 4) & 0x0F0F0F0F);
                A3 = (int)((q_g8_hi >> 4) & 0x0F0F0F0F);
            }

            // B fragment: same broadcast input across all 8 N
            const block_q8_1 * yb = yb_base + s;
            const int * yqs = (const int *)yb->qs;
            int B0 = yqs[2 * tj + 0];
            int B1 = yqs[2 * tj + 1];

            int D0 = 0, D1 = 0, D2 = 0, D3 = 0;
            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+r"(D0), "+r"(D1), "+r"(D2), "+r"(D3)
                : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1));

            // sum of 32 q8 values for sub-min term
            int dot2 = __dp4a(0x01010101, B1, __dp4a(0x01010101, B0, 0));
            dot2 += __shfl_xor_sync(0xffffffff, dot2, 1);
            dot2 += __shfl_xor_sync(0xffffffff, dot2, 2);

            float d8 = __low2float(yb->ds);

            uint8_t sc_g, m_g, sc_g8, m_g8;
            get_scale_min_k4_dev(s, x_g->scales,  sc_g,  m_g);
            get_scale_min_k4_dev(s, x_g8->scales, sc_g8, m_g8);

            sf_d_lo += d8 * (float)D0   * (float)sc_g;
            sf_m_lo += d8 * (float)dot2 * (float)m_g;
            sf_d_hi += d8 * (float)D2   * (float)sc_g8;
            sf_m_hi += d8 * (float)dot2 * (float)m_g8;
        }

        result_lo += dall_g  * sf_d_lo - dmin_g  * sf_m_lo;
        result_hi += dall_g8 * sf_d_hi - dmin_g8 * sf_m_hi;
    }

    if (tj == 0) {
        if (valid_g)  dst[row_g]  = result_lo;
        if (valid_g8) dst[row_g8] = result_hi;
    }
}

} // namespace ll_q4k_imma

extern "C" void q4k_mmvq_imma(
    const void * vx,           // device ptr [nrows_x × num_super × sizeof(block_q4_K)]
    const void * vy,           // device ptr [num_super × 8 × sizeof(block_q8_1)]
    void * dst,                // device ptr [nrows_x] f32
    int ncols_x,               // K dim, must be multiple of 256
    int nrows_x,
    cudaStream_t stream
) {
    const int num_super = ncols_x / 256;
    const int row_groups = (nrows_x + 15) / 16;
    dim3 grid(row_groups, 1, 1);
    dim3 block(32, 1, 1);
    ll_q4k_imma::q4k_mmvq_imma_kernel<<<grid, block, 0, stream>>>(
        (const ll_q4k_imma::block_q4_K *)vx,
        (const ll_q4k_imma::block_q8_1 *)vy,
        (float *)dst,
        num_super,
        nrows_x
    );
}
