/**
 * @file moe_q4k_mmq_splitk.cu
 *
 * Multi-warp Q4_K MMQ kernel with K-split parallelism.
 *
 * Hypothesis: dp4a (2172 tok/s) beats our prior MMA kernels because
 * dp4a launches ~M=1M short-lived warps (a few hundred cycles each) and
 * keeps the 84 SMs × 48 warps/SM = 4032 concurrent warps saturated. Our
 * MMA approaches launch ~50K-150K warps each doing ~2-3K cycles, which
 * under-fills the SM scheduler.
 *
 * Fix: SPLIT-K. 4 warps per block, all share the SAME 16 weight rows ×
 * 8 input rows output tile, but each warp handles 1/4 of the K-loop.
 * After K, warps reduce partial sums to a single final output via
 * shared memory.
 *
 * Per warp: num_super/4 super-blocks (e.g., 3 of 11 for gemma4:26b)
 * worth of work — about 1/4 the prior per-warp work. Same total
 * compute, 4× more warps live concurrently → better SM scheduling.
 *
 * Output: F32 [N_active, max_n_e, 2N], same as the non-split MMA path.
 */
#include "gguf.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace moe_q4k_mmq_splitk_ns {

#define SK_K_SUPER  256
#define SK_M         16
#define SK_N          8
#define SK_NWARPS     4

typedef struct {
    __half2 dm;
    uint8_t scales[12];
    uint8_t qs[128];
} block_q4_K_sk;

typedef struct {
    __half2 ds;
    int8_t  qs[32];
} block_q8_1_sk;

static_assert(sizeof(block_q4_K_sk) == 144, "block_q4_K_sk size");
static_assert(sizeof(block_q8_1_sk) == 36,  "block_q8_1_sk size");

__device__ __forceinline__ void sk_get_scale_min_k4(
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

// K-split: each block produces ONE 16×8 output tile (same as single-warp
// MMA), but 4 warps share the K-loop work and reduce at the end.
//
// Grid:
//   gridDim.x = ceil(2N / 16)
//   gridDim.y = ceil(max_n_e / 8)      -- same as single-warp MMA
//   gridDim.z = N_active
// blockDim = (32, SK_NWARPS=4)
extern "C" __global__ void moe_q4k_mmq_splitk_kernel(
    const void * __restrict__ gate_up_w,         // [num_experts, 2N, K/256] Q4_K
    const void * __restrict__ inputs_q81,        // [N_active, max_n_e, K/32] Q8_1
    const int32_t * __restrict__ active_expert_ids,
    const int32_t * __restrict__ expert_offsets,
    void * __restrict__ dst,                     // [N_active, max_n_e, 2N] F32
    int num_experts,
    int max_n_e,
    int two_n,
    int K
) {
    using namespace moe_q4k_mmq_splitk_ns;

    const int act_idx = blockIdx.z;
    const int n_tile  = blockIdx.y;
    const int m_tile  = blockIdx.x;

    const int expert  = active_expert_ids[act_idx];
    if (expert < 0 || expert >= num_experts) return;
    const int start   = expert_offsets[expert];
    const int end     = expert_offsets[expert + 1];
    const int n_e     = end - start;

    const int m_base = m_tile * SK_M;
    const int n_base = n_tile * SK_N;
    if (m_base >= two_n || n_base >= max_n_e) return;

    const int warp_id = threadIdx.y;
    const int lane    = threadIdx.x;
    const int g       = lane >> 2;
    const int tj      = lane & 3;
    const int num_super = K / SK_K_SUPER;

    const int row_a = m_base + g;
    const int row_b = m_base + g + 8;
    const bool va = row_a < two_n;
    const bool vb = row_b < two_n;

    const int my_in_row = n_base + g;
    const bool v_in = my_in_row < n_e;

    const block_q4_K_sk * w_expert =
        (const block_q4_K_sk *) gate_up_w
        + (size_t)expert * (size_t)two_n * num_super;
    const block_q8_1_sk * y_expert =
        (const block_q8_1_sk *) inputs_q81
        + ((size_t)act_idx * max_n_e + n_base) * num_super * 8;

    // K-split: assign each warp a slab of super-blocks.
    // For 11 super-blocks across 4 warps: 3,3,3,2.
    const int sb_per_warp_base = num_super / SK_NWARPS;
    const int sb_extra         = num_super - sb_per_warp_base * SK_NWARPS;
    const int sb_start = warp_id < sb_extra
        ? warp_id * (sb_per_warp_base + 1)
        : warp_id * sb_per_warp_base + sb_extra;
    const int sb_end   = sb_start
        + (warp_id < sb_extra ? sb_per_warp_base + 1 : sb_per_warp_base);

    float acc_0 = 0.f, acc_1 = 0.f, acc_2 = 0.f, acc_3 = 0.f;

    for (int isb = sb_start; isb < sb_end; ++isb) {
        const block_q4_K_sk * wa_sb = va ? w_expert + (size_t)row_a * num_super + isb : nullptr;
        const block_q4_K_sk * wb_sb = vb ? w_expert + (size_t)row_b * num_super + isb : nullptr;

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

            const block_q8_1_sk * yb_my = v_in
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

            int dot2 = __dp4a(0x01010101, B1, __dp4a(0x01010101, B0, 0));
            dot2 += __shfl_xor_sync(0xffffffff, dot2, 1);
            dot2 += __shfl_xor_sync(0xffffffff, dot2, 2);

            const int src_lane_a = (2 * tj + 0) * 4;
            const int src_lane_b = (2 * tj + 1) * 4;
            const float d8_out_a  = __shfl_sync(0xffffffff, d8_my, src_lane_a);
            const float d8_out_b  = __shfl_sync(0xffffffff, d8_my, src_lane_b);
            const float dot_out_a = (float)__shfl_sync(0xffffffff, dot2, src_lane_a);
            const float dot_out_b = (float)__shfl_sync(0xffffffff, dot2, src_lane_b);

            uint8_t sc_a, m_a, sc_b, m_b;
            if (wa_sb) sk_get_scale_min_k4(s, wa_sb->scales, sc_a, m_a); else { sc_a = m_a = 0; }
            if (wb_sb) sk_get_scale_min_k4(s, wb_sb->scales, sc_b, m_b); else { sc_b = m_b = 0; }

            sub_d_0 += d8_out_a * (float)D0 * (float)sc_a;
            sub_d_1 += d8_out_b * (float)D1 * (float)sc_a;
            sub_d_2 += d8_out_a * (float)D2 * (float)sc_b;
            sub_d_3 += d8_out_b * (float)D3 * (float)sc_b;
            sub_m_0 += d8_out_a * dot_out_a * (float)m_a;
            sub_m_1 += d8_out_b * dot_out_b * (float)m_a;
            sub_m_2 += d8_out_a * dot_out_a * (float)m_b;
            sub_m_3 += d8_out_b * dot_out_b * (float)m_b;
        }

        acc_0 += dall_a * sub_d_0 - dmin_a * sub_m_0;
        acc_1 += dall_a * sub_d_1 - dmin_a * sub_m_1;
        acc_2 += dall_b * sub_d_2 - dmin_b * sub_m_2;
        acc_3 += dall_b * sub_d_3 - dmin_b * sub_m_3;
    }

    // Reduce partial sums across warps via shared memory.
    // Each thread (lane, warp_id) contributes acc_0..acc_3 → final value
    // for (lane, *) — sum across warp_id.
    __shared__ float s_acc[SK_NWARPS][32][4];
    s_acc[warp_id][lane][0] = acc_0;
    s_acc[warp_id][lane][1] = acc_1;
    s_acc[warp_id][lane][2] = acc_2;
    s_acc[warp_id][lane][3] = acc_3;
    __syncthreads();

    // Warp 0 reduces and writes output.
    if (warp_id == 0) {
        float a0 = s_acc[0][lane][0] + s_acc[1][lane][0] + s_acc[2][lane][0] + s_acc[3][lane][0];
        float a1 = s_acc[0][lane][1] + s_acc[1][lane][1] + s_acc[2][lane][1] + s_acc[3][lane][1];
        float a2 = s_acc[0][lane][2] + s_acc[1][lane][2] + s_acc[2][lane][2] + s_acc[3][lane][2];
        float a3 = s_acc[0][lane][3] + s_acc[1][lane][3] + s_acc[2][lane][3] + s_acc[3][lane][3];

        const int in_row_0 = n_base + 2 * tj + 0;
        const int in_row_1 = n_base + 2 * tj + 1;
        float * dst_base = (float *) dst + (size_t)act_idx * max_n_e * two_n;

        if (va && in_row_0 < n_e) dst_base[(size_t)in_row_0 * two_n + row_a] = a0;
        if (va && in_row_1 < n_e) dst_base[(size_t)in_row_1 * two_n + row_a] = a1;
        if (vb && in_row_0 < n_e) dst_base[(size_t)in_row_0 * two_n + row_b] = a2;
        if (vb && in_row_1 < n_e) dst_base[(size_t)in_row_1 * two_n + row_b] = a3;
    }

    (void)num_experts;
}

extern "C" void moe_q4k_mmq_splitk_gate_up(
    const void * gate_up_w,
    const void * inputs_q81,
    const int32_t * active_expert_ids,
    const int32_t * expert_offsets,
    void * dst_f32,
    int num_experts,
    int n_active,
    int max_n_e,
    int two_n,
    int K,
    cudaStream_t stream
) {
    if (n_active <= 0 || max_n_e <= 0 || two_n <= 0 || K <= 0) return;
    using namespace moe_q4k_mmq_splitk_ns;
    dim3 grid((two_n   + SK_M - 1) / SK_M,
              (max_n_e + SK_N - 1) / SK_N,
              n_active);
    dim3 blk(32, SK_NWARPS, 1);
    moe_q4k_mmq_splitk_kernel<<<grid, blk, 0, stream>>>(
        gate_up_w, inputs_q81, active_expert_ids, expert_offsets,
        dst_f32, num_experts, max_n_e, two_n, K
    );
}
