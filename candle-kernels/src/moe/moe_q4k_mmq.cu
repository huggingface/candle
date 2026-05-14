/**
 * @file moe_q4k_mmq.cu
 *
 * Multi-warp Q4_K MMQ kernel for MoE per-expert prefill, modeled on
 * llama.cpp's mmq.cu pattern (load_tiles_q4_K + vec_dot_q8_1_q8_1_mma).
 *
 * Key difference from the single-warp moe_q4k_mma_batched_gate_up:
 *
 *   - 4 warps per block (128 threads) share the SAME 16 weight rows.
 *   - Each warp owns 8 different input rows (warp w → rows w*8..w*8+7).
 *   - Per super-block, all 4 warps cooperatively LOAD Q4_K nibbles into
 *     shared memory, then read from shared memory in their per-warp
 *     mma.sync inner loop. Q4_K HBM weight reads are shared 4× across
 *     the warps (no longer per-warp duplicated).
 *
 *   Block output tile: 16 weight rows × 32 input rows = 512 outputs.
 *
 * Grid:
 *   gridDim.x = ceil(2N / 16)
 *   gridDim.y = ceil(max_n_e / 32)
 *   gridDim.z = N_active
 * blockDim = (32, 4, 1)
 *
 * Output layout: [N_active, max_n_e, 2N] F32 (same as the unfused
 * non-MMQ kernel — re-uses the existing F32 GELU·mul scatter).
 *
 * Q4_K block layout matches block_q4_K (144 bytes):
 *   half2 dm (4 bytes: dall, dmin)
 *   uint8_t scales[12]  (6-bit packed sc/m for 8 sub-blocks)
 *   uint8_t qs[128]     (256 nibbles)
 *
 * Q8_1 block (36 bytes): half2 ds (d, sum_scaled), int8_t qs[32].
 */
#include "gguf.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace moe_q4k_mmq_ns {

#define MMQ_K_SUPER       256   // Q4_K super-block size
#define MMQ_K_TILE         32   // m16n8k32 K-tile
#define MMQ_M              16   // weight rows per block
#define MMQ_X              32   // input rows per block
#define MMQ_NWARPS          4   // warps per block

typedef struct {
    __half2 dm;
    uint8_t scales[12];
    uint8_t qs[128];
} block_q4_K_mmq;

typedef struct {
    __half2 ds;
    int8_t  qs[32];
} block_q8_1_mmq;

static_assert(sizeof(block_q4_K_mmq) == 4 + 12 + 128, "block_q4_K_mmq size");
static_assert(sizeof(block_q8_1_mmq) == 4 + 32,        "block_q8_1_mmq size");

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

// Multi-warp Q4_K MMQ kernel.
//
// Per block (4 warps × 32 threads = 128 threads):
//   - Per super-block: cooperative shared-memory weight load + 4 warp MMAs
//   - Each warp processes 8 input rows × 16 weight rows
//   - K-loop runs num_super super-blocks; per super-block, 8 K-tiles
//
// Shared memory layout (per block, ~3.5 KB):
//   s_qs[MMQ_M][32]   — 4 INT32 packs (16 lanes) of nibbles per row, per K-tile
//                       Stored as int packs so each warp can load fragment via shfl.
//   s_dm[MMQ_M]       — half2 (dall, dmin) per weight row, per super-block
//   s_sc[MMQ_M][8]    — uint8 sc per K-tile within super-block per weight row
//   s_m [MMQ_M][8]    — uint8 m  per K-tile within super-block per weight row
extern "C" __global__ void moe_q4k_mmq_kernel(
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
    using namespace moe_q4k_mmq_ns;

    const int act_idx = blockIdx.z;
    const int n_block = blockIdx.y;     // input-row block (32 rows each)
    const int m_block = blockIdx.x;     // weight-row block (16 rows each)

    const int expert  = active_expert_ids[act_idx];
    if (expert < 0 || expert >= num_experts) return;
    const int start   = expert_offsets[expert];
    const int end     = expert_offsets[expert + 1];
    const int n_e     = end - start;

    const int m_base  = m_block * MMQ_M;
    const int n_base  = n_block * MMQ_X;
    if (m_base >= two_n || n_base >= max_n_e) return;

    const int warp_id = threadIdx.y;
    const int lane    = threadIdx.x;
    const int g       = lane >> 2;
    const int tj      = lane & 3;
    const int num_super = K / MMQ_K_SUPER;

    // This warp's 8-input-row range within the block.
    const int n_warp_base = n_base + warp_id * 8;
    const int my_in_row   = n_warp_base + g;
    const bool v_in       = my_in_row < n_e;

    // Weight rows handled by THE BLOCK (all warps see the same 16 rows).
    const int row_a = m_base + g;
    const int row_b = m_base + g + 8;
    const bool va   = row_a < two_n;
    const bool vb   = row_b < two_n;

    const block_q4_K_mmq * w_expert =
        (const block_q4_K_mmq *) gate_up_w
        + (size_t)expert * (size_t)two_n * num_super;
    const block_q8_1_mmq * y_expert =
        (const block_q8_1_mmq *) inputs_q81
        + ((size_t)act_idx * max_n_e + n_warp_base) * num_super * 8;

    // Shared memory: per-super-block Q4_K weight nibbles + scales.
    // qs:  16 rows × 128 bytes/row = 2048 bytes
    // dm:  16 × 4 bytes = 64 bytes
    // sc/m: 16 × 8 × 2 = 256 bytes
    //   Total: ~2.4 KB
    __shared__ uint8_t s_qs[MMQ_M][128];
    __shared__ __half2 s_dm[MMQ_M];
    __shared__ uint8_t s_sc[MMQ_M][8];
    __shared__ uint8_t s_m [MMQ_M][8];

    // Per-thread output accumulators. Each thread covers 4 output positions:
    //   D0 = (m=row_a,   n=n_warp_base+2tj+0)
    //   D1 = (m=row_a,   n=n_warp_base+2tj+1)
    //   D2 = (m=row_b,   n=n_warp_base+2tj+0)
    //   D3 = (m=row_b,   n=n_warp_base+2tj+1)
    float acc_0 = 0.f, acc_1 = 0.f, acc_2 = 0.f, acc_3 = 0.f;

    for (int isb = 0; isb < num_super; ++isb) {
        // ─── Step 1: cooperative shared-memory load of Q4_K weights ──
        // 128 threads × 16 bytes each = 2048 bytes (== qs size for 16 rows).
        // Threads are organised as (lane=0..31, warp_id=0..3). Each
        // (warp_id, lane) loads byte (warp_id*32 + lane) of one row's
        // qs[128] for several rows. We iterate over rows.
        //
        // Simpler scheme: each warp handles 4 rows (16/4=4). Lane writes
        // 32 bytes of that row from threadIdx.x*4..threadIdx.x*4+3.
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = warp_id * 4 + r;
            const int weight_row = m_base + row;
            const block_q4_K_mmq * wb = (weight_row < two_n)
                ? w_expert + (size_t)weight_row * num_super + isb
                : nullptr;

            // 32 lanes load 128 bytes of qs (4 bytes each).
            if (wb) {
                const uint32_t qs_word = *(const uint32_t *)(wb->qs + lane * 4);
                *(uint32_t *)(s_qs[row] + lane * 4) = qs_word;
            } else {
                *(uint32_t *)(s_qs[row] + lane * 4) = 0;
            }
            // Lane 0 of each (warp, r) loads dm + scales.
            if (lane == 0) {
                if (wb) {
                    s_dm[row] = wb->dm;
                    uint8_t sc, m;
                    #pragma unroll
                    for (int s = 0; s < 8; ++s) {
                        get_scale_min_k4_dev(s, wb->scales, sc, m);
                        s_sc[row][s] = sc;
                        s_m [row][s] = m;
                    }
                } else {
                    s_dm[row] = __floats2half2_rn(0.f, 0.f);
                    #pragma unroll
                    for (int s = 0; s < 8; ++s) {
                        s_sc[row][s] = 0;
                        s_m [row][s] = 0;
                    }
                }
            }
        }
        __syncthreads();

        // ─── Step 2: per-warp inner mma loop ─────────────────────────
        // For super-block isb, run 8 K-tiles. Each K-tile is 32 K
        // positions = MMA's k=32 dim.
        const float dall_a = va ? __low2float (s_dm[g])     : 0.f;
        const float dmin_a = va ? __high2float(s_dm[g])     : 0.f;
        const float dall_b = vb ? __low2float (s_dm[g + 8]) : 0.f;
        const float dmin_b = vb ? __high2float(s_dm[g + 8]) : 0.f;

        float sub_d_0 = 0.f, sub_d_1 = 0.f, sub_d_2 = 0.f, sub_d_3 = 0.f;
        float sub_m_0 = 0.f, sub_m_1 = 0.f, sub_m_2 = 0.f, sub_m_3 = 0.f;

        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int il = s >> 1;
            const int ip = s & 1;
            const int qs_off = 32 * il + 8 * tj;

            // Read Q4_K nibbles from shared memory for weight rows a and b.
            uint32_t qa_lo = va ? *(const uint32_t *)(s_qs[g]     + qs_off + 0) : 0;
            uint32_t qa_hi = va ? *(const uint32_t *)(s_qs[g]     + qs_off + 4) : 0;
            uint32_t qb_lo = vb ? *(const uint32_t *)(s_qs[g + 8] + qs_off + 0) : 0;
            uint32_t qb_hi = vb ? *(const uint32_t *)(s_qs[g + 8] + qs_off + 4) : 0;

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

            // B fragment: this warp's input row 'g' Q8_1 K-tile.
            const block_q8_1_mmq * yb_my = v_in
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

            // Scales from shared memory.
            const uint8_t sc_a = va ? s_sc[g    ][s] : 0;
            const uint8_t m_a  = va ? s_m [g    ][s] : 0;
            const uint8_t sc_b = vb ? s_sc[g + 8][s] : 0;
            const uint8_t m_b  = vb ? s_m [g + 8][s] : 0;

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

        __syncthreads();   // Before next super-block re-uses s_qs.
    }

    // Output writes: dst[act_idx, in_row, weight_row].
    const int in_row_0 = n_warp_base + 2 * tj + 0;
    const int in_row_1 = n_warp_base + 2 * tj + 1;
    float * dst_base = (float *) dst + (size_t)act_idx * max_n_e * two_n;

    if (va && in_row_0 < n_e) {
        dst_base[(size_t)in_row_0 * two_n + row_a] = acc_0;
    }
    if (va && in_row_1 < n_e) {
        dst_base[(size_t)in_row_1 * two_n + row_a] = acc_1;
    }
    if (vb && in_row_0 < n_e) {
        dst_base[(size_t)in_row_0 * two_n + row_b] = acc_2;
    }
    if (vb && in_row_1 < n_e) {
        dst_base[(size_t)in_row_1 * two_n + row_b] = acc_3;
    }

    (void)num_experts;
}

extern "C" void moe_q4k_mmq_gate_up(
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
    using namespace moe_q4k_mmq_ns;
    dim3 grid((two_n   + MMQ_M - 1) / MMQ_M,
              (max_n_e + MMQ_X - 1) / MMQ_X,
              n_active);
    dim3 blk(32, MMQ_NWARPS, 1);
    moe_q4k_mmq_kernel<<<grid, blk, 0, stream>>>(
        gate_up_w, inputs_q81, active_expert_ids, expert_offsets,
        dst_f32, num_experts, max_n_e, two_n, K
    );
}
