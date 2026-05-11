/**
 * @brief Tensor-core MoE Q4_K × Q8_1 mul-mat-vec for the gate||up concat
 * layout, with fused GELU(tanh) and elementwise multiply. Drop-in
 * tensor-core replacement for moe_gemm_gguf_gate_up_gelu_mul_concat
 * (dp4a path) for Q4_K weights with K divisible by 256.
 *
 * Weight layout per expert: [2*N, K] Q4_K, gate rows in [0..N), up rows
 * in [N..2N). One block computes 16 gate rows AND 16 up rows for a
 * single (token, expert) pair, then writes GELU(gate) * up to the
 * caller-provided [size_m, N] output buffer.
 *
 * Adapted from q4k_mmvq_imma.cu — same Q4K decode + mma.sync structure,
 * with MoE routing on top and gate||up fusion.
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace ll_moe_q4k_imma {

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

// One block = one warp = 16 gate rows + 16 up rows × 1 (token, expert) pair.
// gridDim.x = ceil(N / 16), gridDim.y = size_m, blockDim.x = 32.
__global__ void moe_q4k_imma_gate_up_gelu_mul_concat_kernel(
    const block_q4_K * __restrict__ gate_up_w,   // [num_experts × 2N × num_super]
    const block_q8_1 * __restrict__ vy,          // [num_real_tokens × num_super × 8]
    const int32_t * __restrict__ sorted_token_ids, // [size_m]
    const int32_t * __restrict__ expert_ids,     // [size_m]
    float * __restrict__ dst,                    // [size_m × N]  (gelu(gate) * up)
    int num_experts,
    int topk,
    int size_m,
    int N,                                       // gate half rows
    int num_super
) {
    constexpr int ROW_GROUP = 16;
    const int row0 = blockIdx.x * ROW_GROUP;
    const int m_idx = blockIdx.y;
    if (row0 >= N || m_idx >= size_m) return;

    const int expert = expert_ids[m_idx];
    if (expert < 0 || expert >= num_experts) return;

    const int token_id  = sorted_token_ids[m_idx];
    const int input_idx = token_id / topk;

    const int lane = threadIdx.x;
    const int g  = lane >> 2;       // 0..7
    const int tj = lane & 3;        // 0..3

    const int row_g  = row0 + g;
    const int row_g8 = row0 + g + 8;
    const bool valid_g  = row_g  < N;
    const bool valid_g8 = row_g8 < N;

    const size_t expert_stride_rows = (size_t)2 * N;
    const block_q4_K * exp_base =
        gate_up_w + (size_t)expert * expert_stride_rows * num_super;

    // Gate weight rows: [row0..row0+15] of this expert slab.
    const block_q4_K * gate_g  = exp_base + (size_t)row_g  * num_super;
    const block_q4_K * gate_g8 = exp_base + (size_t)row_g8 * num_super;
    // Up weight rows: [N+row0..N+row0+15] of this expert slab.
    const block_q4_K * up_g    = exp_base + (size_t)(N + row_g)  * num_super;
    const block_q4_K * up_g8   = exp_base + (size_t)(N + row_g8) * num_super;
    // Input (pre-quantized Q8_1): one vector per real token.
    const block_q8_1 * y_ptr = vy + (size_t)input_idx * num_super * 8;

    float gate_lo = 0.f, gate_hi = 0.f, up_lo = 0.f, up_hi = 0.f;

    for (int isb = 0; isb < num_super; ++isb) {
        const block_q4_K * gw_g  = gate_g  + isb;
        const block_q4_K * gw_g8 = gate_g8 + isb;
        const block_q4_K * uw_g  = up_g    + isb;
        const block_q4_K * uw_g8 = up_g8   + isb;
        const block_q8_1 * yb_base = y_ptr + isb * 8;

        const float g_dall_g  = valid_g  ? __low2float (gw_g->dm)  : 0.f;
        const float g_dmin_g  = valid_g  ? __high2float(gw_g->dm)  : 0.f;
        const float g_dall_g8 = valid_g8 ? __low2float (gw_g8->dm) : 0.f;
        const float g_dmin_g8 = valid_g8 ? __high2float(gw_g8->dm) : 0.f;
        const float u_dall_g  = valid_g  ? __low2float (uw_g->dm)  : 0.f;
        const float u_dmin_g  = valid_g  ? __high2float(uw_g->dm)  : 0.f;
        const float u_dall_g8 = valid_g8 ? __low2float (uw_g8->dm) : 0.f;
        const float u_dmin_g8 = valid_g8 ? __high2float(uw_g8->dm) : 0.f;

        float gf_d_lo = 0.f, gf_m_lo = 0.f;
        float gf_d_hi = 0.f, gf_m_hi = 0.f;
        float uf_d_lo = 0.f, uf_m_lo = 0.f;
        float uf_d_hi = 0.f, uf_m_hi = 0.f;

        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int il = s >> 1;
            const int ip = s & 1;
            const int qs_off = 32 * il + 8 * tj;

            // Read 8 quant bytes per row (2 ints) — GATE side
            uint32_t qg_g_lo  = valid_g  ? *(const uint32_t *)(gw_g->qs  + qs_off + 0) : 0;
            uint32_t qg_g_hi  = valid_g  ? *(const uint32_t *)(gw_g->qs  + qs_off + 4) : 0;
            uint32_t qg_g8_lo = valid_g8 ? *(const uint32_t *)(gw_g8->qs + qs_off + 0) : 0;
            uint32_t qg_g8_hi = valid_g8 ? *(const uint32_t *)(gw_g8->qs + qs_off + 4) : 0;
            // UP side
            uint32_t qu_g_lo  = valid_g  ? *(const uint32_t *)(uw_g->qs  + qs_off + 0) : 0;
            uint32_t qu_g_hi  = valid_g  ? *(const uint32_t *)(uw_g->qs  + qs_off + 4) : 0;
            uint32_t qu_g8_lo = valid_g8 ? *(const uint32_t *)(uw_g8->qs + qs_off + 0) : 0;
            uint32_t qu_g8_hi = valid_g8 ? *(const uint32_t *)(uw_g8->qs + qs_off + 4) : 0;

            int GA0, GA1, GA2, GA3;
            int UA0, UA1, UA2, UA3;
            if (ip == 0) {
                GA0 = (int)(qg_g_lo  & 0x0F0F0F0F);
                GA2 = (int)(qg_g_hi  & 0x0F0F0F0F);
                GA1 = (int)(qg_g8_lo & 0x0F0F0F0F);
                GA3 = (int)(qg_g8_hi & 0x0F0F0F0F);
                UA0 = (int)(qu_g_lo  & 0x0F0F0F0F);
                UA2 = (int)(qu_g_hi  & 0x0F0F0F0F);
                UA1 = (int)(qu_g8_lo & 0x0F0F0F0F);
                UA3 = (int)(qu_g8_hi & 0x0F0F0F0F);
            } else {
                GA0 = (int)((qg_g_lo  >> 4) & 0x0F0F0F0F);
                GA2 = (int)((qg_g_hi  >> 4) & 0x0F0F0F0F);
                GA1 = (int)((qg_g8_lo >> 4) & 0x0F0F0F0F);
                GA3 = (int)((qg_g8_hi >> 4) & 0x0F0F0F0F);
                UA0 = (int)((qu_g_lo  >> 4) & 0x0F0F0F0F);
                UA2 = (int)((qu_g_hi  >> 4) & 0x0F0F0F0F);
                UA1 = (int)((qu_g8_lo >> 4) & 0x0F0F0F0F);
                UA3 = (int)((qu_g8_hi >> 4) & 0x0F0F0F0F);
            }

            // B fragment: input Q8_1, shared across gate and up.
            const block_q8_1 * yb = yb_base + s;
            const int * yqs = (const int *)yb->qs;
            int B0 = yqs[2 * tj + 0];
            int B1 = yqs[2 * tj + 1];

            // mma.sync for gate
            int GD0 = 0, GD1 = 0, GD2 = 0, GD3 = 0;
            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+r"(GD0), "+r"(GD1), "+r"(GD2), "+r"(GD3)
                : "r"(GA0), "r"(GA1), "r"(GA2), "r"(GA3), "r"(B0), "r"(B1));

            // mma.sync for up — same B inputs, different A
            int UD0 = 0, UD1 = 0, UD2 = 0, UD3 = 0;
            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+r"(UD0), "+r"(UD1), "+r"(UD2), "+r"(UD3)
                : "r"(UA0), "r"(UA1), "r"(UA2), "r"(UA3), "r"(B0), "r"(B1));

            // sum of 32 q8 values for sub-min term (shared between gate and up)
            int dot2 = __dp4a(0x01010101, B1, __dp4a(0x01010101, B0, 0));
            dot2 += __shfl_xor_sync(0xffffffff, dot2, 1);
            dot2 += __shfl_xor_sync(0xffffffff, dot2, 2);

            float d8 = __low2float(yb->ds);

            uint8_t sc_g_g,  m_g_g,  sc_g_g8,  m_g_g8;
            uint8_t sc_u_g,  m_u_g,  sc_u_g8,  m_u_g8;
            get_scale_min_k4_dev(s, gw_g->scales,  sc_g_g,  m_g_g);
            get_scale_min_k4_dev(s, gw_g8->scales, sc_g_g8, m_g_g8);
            get_scale_min_k4_dev(s, uw_g->scales,  sc_u_g,  m_u_g);
            get_scale_min_k4_dev(s, uw_g8->scales, sc_u_g8, m_u_g8);

            gf_d_lo += d8 * (float)GD0  * (float)sc_g_g;
            gf_m_lo += d8 * (float)dot2 * (float)m_g_g;
            gf_d_hi += d8 * (float)GD2  * (float)sc_g_g8;
            gf_m_hi += d8 * (float)dot2 * (float)m_g_g8;
            uf_d_lo += d8 * (float)UD0  * (float)sc_u_g;
            uf_m_lo += d8 * (float)dot2 * (float)m_u_g;
            uf_d_hi += d8 * (float)UD2  * (float)sc_u_g8;
            uf_m_hi += d8 * (float)dot2 * (float)m_u_g8;
        }

        gate_lo += g_dall_g  * gf_d_lo - g_dmin_g  * gf_m_lo;
        gate_hi += g_dall_g8 * gf_d_hi - g_dmin_g8 * gf_m_hi;
        up_lo   += u_dall_g  * uf_d_lo - u_dmin_g  * uf_m_lo;
        up_hi   += u_dall_g8 * uf_d_hi - u_dmin_g8 * uf_m_hi;
    }

    if (tj == 0) {
        // GELU(tanh-approx) — matches gemma4's gelu_pytorch_tanh:
        //   0.5*g*(1 + tanh(sqrt(2/pi)*(g + 0.044715*g^3)))
        const float k0 = 0.7978845608028654f;
        const float k1 = 0.044715f;
        if (valid_g) {
            const float gv = gate_lo;
            const float gelu_g = 0.5f * gv * (1.0f + tanhf(k0 * (gv + k1 * gv * gv * gv)));
            dst[(size_t)token_id * N + row_g] = gelu_g * up_lo;
        }
        if (valid_g8) {
            const float gv = gate_hi;
            const float gelu_g = 0.5f * gv * (1.0f + tanhf(k0 * (gv + k1 * gv * gv * gv)));
            dst[(size_t)token_id * N + row_g8] = gelu_g * up_hi;
        }
    }
}

} // namespace ll_moe_q4k_imma

// Forward declaration of upstream quantize-Q8_1 launcher (defined in
// mmvq_gguf.cu / fast_mmvq). Quantizes F32 input rows to per-32-block
// Q8_1 (int8 + half2 ds) into the caller-provided scratch buffer.
extern "C" void launch_mmvq_gguf_quantize_q8_1_f32(
    const void * x,
    void * dst,
    int ncols_x,
    int k_padded,
    int b_size,
    void * stream
);

extern "C" void moe_q4k_imma_gate_up_gelu_mul_concat(
    const void   * inputs,             // [num_real_tokens × K] f32
    const void   * gate_up_w,          // [num_experts × 2N × num_super × sizeof(block_q4_K)]
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    void         * dst,                // [size_m × N] f32
    int num_experts,
    int topk,
    int size_m,                        // M*topk (pairs)
    int N,                             // gate half rows
    int K,                             // K dim, must be multiple of 256
    long long stream_handle            // matches Rust i64 FFI
) {
    cudaStream_t stream = (cudaStream_t)stream_handle;

    // Pre-quantize input rows to Q8_1 in a scratch buffer.
    const int num_real_tokens = size_m / topk;
    const size_t y_bytes = (size_t)num_real_tokens
                         * (size_t)(K / QK8_1)
                         * sizeof(ll_moe_q4k_imma::block_q8_1);
    void * y_q8_1 = nullptr;
    cudaMallocAsync(&y_q8_1, y_bytes, stream);
    launch_mmvq_gguf_quantize_q8_1_f32(
        inputs, y_q8_1, K, K, num_real_tokens, (void *)stream
    );
    constexpr int ROW_GROUP = 16;
    const int num_super = K / 256;
    const int row_groups = (N + ROW_GROUP - 1) / ROW_GROUP;
    dim3 grid(row_groups, size_m, 1);
    dim3 block(32, 1, 1);
    ll_moe_q4k_imma::moe_q4k_imma_gate_up_gelu_mul_concat_kernel
        <<<grid, block, 0, stream>>>(
            (const ll_moe_q4k_imma::block_q4_K *)gate_up_w,
            (const ll_moe_q4k_imma::block_q8_1 *)y_q8_1,
            sorted_token_ids,
            expert_ids,
            (float *)dst,
            num_experts,
            topk,
            size_m,
            N,
            num_super
        );
    cudaFreeAsync(y_q8_1, stream);
}
