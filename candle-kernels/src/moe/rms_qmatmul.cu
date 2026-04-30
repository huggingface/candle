/**
 * @brief Fused RMS norm + Q4_K matmul (single-token decode).
 *
 * Replaces three-launch sequence
 *   x_normed = rms_norm(x, w_norm)        // 1 launch
 *   x_q8_1   = quantize_q8_1(x_normed)    // 1 launch
 *   y        = dequantize_mul_mat(W_q4k, x_q8_1)  // 1 launch
 * with a single CUDA launch for the typical attn_norm + wqkv (or
 * ffn_norm + wgate-router) flow at decode time.
 *
 * Per-block layout:
 *   blockIdx.x = output row index (one block per row of W^T).
 *   blockDim.x = WARP_SIZE (32) per warp, blockDim.y = nwarps (default 4).
 *
 * Each block recomputes x_normed and its q8_1 representation in shared
 * memory, then runs the standard Q4_K_Q8_1 dot product to produce the
 * row's output element. The norm cost is amortized across the matmul's
 * existing global-memory traffic — net win is the elimination of the
 * two upstream launches and the round-trip through global F32
 * x_normed / q8_1 buffers.
 *
 * The dequantize input path (F32 → Q8_1 in shared) follows
 * `quantize_q8_1` from candle's quantized.cu but operates on the
 * post-norm data we just produced in shared, so we don't need a
 * separate quantize kernel.
 */
#include "gguf.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace ll_rmm {

// Block-wide reduction across `nthreads` threads.
__device__ __forceinline__ float block_reduce_sum(float v, int nthreads, float* tmp) {
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, mask, WARP_SIZE);
    }
    const int lane = threadIdx.x & 31;
    const int wid  = (threadIdx.y * blockDim.x + threadIdx.x) >> 5;
    if (lane == 0) tmp[wid] = v;
    __syncthreads();
    if (wid == 0) {
        const int n_warps = (nthreads + WARP_SIZE - 1) / WARP_SIZE;
        v = (lane < n_warps) ? tmp[lane] : 0.f;
        #pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
            v += __shfl_xor_sync(0xFFFFFFFFu, v, mask, WARP_SIZE);
        }
        if (lane == 0) tmp[0] = v;
    }
    __syncthreads();
    return tmp[0];
}

// Quantize one F32 vector of length `hidden` (must be divisible by QK8_1=32)
// to block_q8_1 in shared memory. Each warp handles one or more blocks.
// `dst` length = hidden / QK8_1 blocks.
__device__ __forceinline__ void quantize_to_q8_1_shared(
    const float* __restrict__ src,
    block_q8_1*  __restrict__ dst,
    int hidden,
    int tid,                     // global thread index in block
    int nthreads
) {
    const int n_blocks = hidden / QK8_1;
    for (int b = tid / WARP_SIZE; b < n_blocks; b += nthreads / WARP_SIZE) {
        const int lane = tid & 31;
        // Each warp does one block of 32 elements
        float v = src[b * QK8_1 + lane];
        // Find absmax + sum across the warp. (`sum` here is the sum of
        // ORIGINAL float values — that's what candle's vec_dot_q*_K_q8_1
        // expects in `ds.y`. Earlier mistake: storing `sum_int8 * d`
        // produced gibberish output.)
        float amax = fabsf(v);
        float sum = v;
        #pragma unroll
        for (int mask = 16; mask > 0; mask /= 2) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, mask, WARP_SIZE));
            sum  = sum + __shfl_xor_sync(0xFFFFFFFFu, sum,  mask, WARP_SIZE);
        }
        const float d = amax / 127.0f;
        const int8_t q = (amax == 0.0f) ? (int8_t)0 : (int8_t)__float2int_rn(v / d);
        if (lane == 0) {
            reinterpret_cast<__half&>(dst[b].ds.x) = __float2half(d);
            reinterpret_cast<__half&>(dst[b].ds.y) = __float2half(sum);
        }
        dst[b].qs[lane] = q;
    }
}

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
__global__ void rms_qmatmul_kernel(
    const float* __restrict__ x,           // [hidden] F32 input
    const float* __restrict__ w_norm,      // [hidden] F32 RMS norm weight
    const block_q_t* __restrict__ w_mm,    // [out_rows, hidden] quantized matmul weight
    float* __restrict__ out,                // [out_rows] F32
    int hidden,
    int out_rows,
    float rms_eps
) {
    const int laneId = threadIdx.x;
    const int wrapId = threadIdx.y;
    const int nWraps = blockDim.y;
    const int tid    = wrapId * WARP_SIZE + laneId;
    const int nthr   = nWraps * WARP_SIZE;
    const int row    = blockIdx.x;
    if (row >= out_rows) return;

    const int blocks_per_row_x = hidden / qk;
    const int q8_blocks        = hidden / QK8_1;
    // Stride must include nWraps so warps cover DISJOINT kbx values;
    // otherwise all warps redundantly compute the full row and the
    // cross-warp sum at the end multiplies by nWraps. (Bit-mismatch
    // bug observed: output was exactly nWraps× the reference.)
    const int blocks_per_iter  = vdr * nWraps * WARP_SIZE / qi;

    // Shared memory layout:
    //   [hidden floats] x_normed
    //   [q8_blocks * sizeof(block_q8_1)] q8_1 quantized x_normed
    //   [nWraps floats] warp_tmp for block reduction
    extern __shared__ unsigned char s_raw[];
    float* x_normed = reinterpret_cast<float*>(s_raw);
    block_q8_1* y_q8 = reinterpret_cast<block_q8_1*>(x_normed + hidden);
    float* warp_tmp = reinterpret_cast<float*>(y_q8 + q8_blocks);

    // 1) Load x and accumulate sum-of-squares for RMS norm.
    float local_sumsq = 0.0f;
    for (int i = tid; i < hidden; i += nthr) {
        const float v = x[i];
        x_normed[i] = v;
        local_sumsq += v * v;
    }
    __syncthreads();

    // 2) Block reduce sumsq, compute inv_rms.
    const float sumsq = block_reduce_sum(local_sumsq, nthr, warp_tmp);
    const float inv_rms = rsqrtf(sumsq / (float)hidden + rms_eps);

    // 3) Apply RMS scaling: x_normed[i] = x[i] * inv_rms * w_norm[i].
    for (int i = tid; i < hidden; i += nthr) {
        x_normed[i] = x_normed[i] * inv_rms * w_norm[i];
    }
    __syncthreads();

    // 4) Quantize x_normed to q8_1 in shared memory.
    quantize_to_q8_1_shared(x_normed, y_q8, hidden, tid, nthr);
    __syncthreads();

    // 5) Standard Q*_K_Q8_1 dot product. Each warp processes its strided
    //    portion of the K-dim blocks, partial sums combined cross-warp
    //    via warp_tmp.
    const block_q_t* w_row = w_mm + (size_t)row * blocks_per_row_x;

    float acc = 0.0f;
    // Use the global thread index `tid` to spread kbx across all warps
    // in the block; matches candle's mul_mat_vec_q template.
    #pragma unroll
    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (laneId % (qi / vdr));
        acc += vec_dot_q_cuda(&w_row[kbx], &y_q8[kby], kqs);
    }

    // Sum across warps (through warp_tmp[0..nwarps-1]).
    // Reuse warp_tmp scratch space.
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
        acc += __shfl_xor_sync(0xFFFFFFFFu, acc, mask, WARP_SIZE);
    }
    if (laneId == 0) warp_tmp[wrapId] = acc;
    __syncthreads();
    if (wrapId == 0) {
        float v = (laneId < nWraps) ? warp_tmp[laneId] : 0.0f;
        #pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
            v += __shfl_xor_sync(0xFFFFFFFFu, v, mask, WARP_SIZE);
        }
        if (laneId == 0) {
            out[row] = v;
        }
    }
}

} // namespace ll_rmm

#define LAUNCH_RMS_QMATMUL(qk, qi, block_q_t, vdr, vec_dot_q_cuda) \
    do { \
        const int q8_blocks = hidden / QK8_1; \
        const size_t smem_x = (size_t)hidden * sizeof(float); \
        const size_t smem_q = (size_t)q8_blocks * sizeof(block_q8_1); \
        const size_t smem_t = (size_t)nWraps * sizeof(float); \
        const size_t smem   = smem_x + smem_q + smem_t + 64; \
        ll_rmm::rms_qmatmul_kernel<qk, qi, block_q_t, vdr, vec_dot_q_cuda> \
            <<<grid, block, smem, stream>>>( \
                x, w_norm, (const block_q_t*)w_mm, out, hidden, out_rows, rms_eps); \
    } while (0)

// ─── Fused RMS norm + Q8_1 quantize (NO matmul) ─────────────────────────
// Replaces 2-launch sequence:
//   x_normed = rms_norm(x, w_norm)        // 1 launch
//   x_q8_1   = quantize_q8_1(x_normed)    // 1 launch
// with ONE launch that produces the q8_1 buffer directly. Caller then
// dispatches a standard mvq kernel against the q8_1 output. Avoids the
// per-block redundancy of the all-fused rms_qmatmul (which recomputes
// norm+quantize on every output row's block).
//
// Single-block kernel: hidden up to 16384 fits in shared mem.
namespace ll_rmm {

template <int block_size>
__global__ void rms_quantize_q8_1_kernel(
    const float* __restrict__ x,         // [hidden] F32
    const float* __restrict__ w_norm,    // [hidden] F32
    block_q8_1* __restrict__ y,          // [hidden / QK8_1] q8_1
    int hidden,
    float rms_eps
) {
    extern __shared__ unsigned char s_raw[];
    float* x_normed = reinterpret_cast<float*>(s_raw);
    float* warp_tmp = reinterpret_cast<float*>(x_normed + hidden);

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int nthr = blockDim.x * blockDim.y;

    // Pass 1: load x, compute sum-of-squares.
    float local_sumsq = 0.0f;
    for (int i = tid; i < hidden; i += nthr) {
        const float v = x[i];
        x_normed[i] = v;
        local_sumsq += v * v;
    }
    __syncthreads();
    const float sumsq = block_reduce_sum(local_sumsq, nthr, warp_tmp);
    const float inv_rms = rsqrtf(sumsq / (float)hidden + rms_eps);

    // Pass 2: scale by inv_rms * w_norm.
    for (int i = tid; i < hidden; i += nthr) {
        x_normed[i] = x_normed[i] * inv_rms * w_norm[i];
    }
    __syncthreads();

    // Pass 3: quantize to q8_1 (one warp per block, write directly to global y).
    quantize_to_q8_1_shared(x_normed, y, hidden, tid, nthr);
}

} // namespace ll_rmm

extern "C" void rms_quantize_q8_1(
    const float* x,
    const float* w_norm,
    void* y_q8_1,
    int hidden,
    float rms_eps,
    cudaStream_t stream
) {
    constexpr int block_size = 256;
    const size_t smem_x = (size_t)hidden * sizeof(float);
    const size_t smem_t = (size_t)(block_size / WARP_SIZE) * sizeof(float);
    const size_t smem = smem_x + smem_t + 64;
    dim3 grid(1, 1, 1);
    dim3 block(WARP_SIZE, block_size / WARP_SIZE, 1);
    ll_rmm::rms_quantize_q8_1_kernel<block_size><<<grid, block, smem, stream>>>(
        x, w_norm, (block_q8_1*)y_q8_1, hidden, rms_eps
    );
}

extern "C" void rms_qmatmul(
    const float* x,         // [hidden]
    const float* w_norm,    // [hidden]
    const void*  w_mm,      // [out_rows, hidden] quantized
    float*       out,       // [out_rows]
    int hidden,
    int out_rows,
    float rms_eps,
    int quant_type,         // 0=Q8_0, 1=Q4K, 2=Q2K, 3=Q3K, 4=Q5K, 5=Q6K
    cudaStream_t stream
) {
    const int nWraps = 4;
    dim3 grid((unsigned)out_rows, 1, 1);
    dim3 block(WARP_SIZE, nWraps, 1);

    switch (quant_type) {
        case 0: LAUNCH_RMS_QMATMUL(QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1); break;
        case 1: LAUNCH_RMS_QMATMUL(QK_K,  QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1); break;
        case 2: LAUNCH_RMS_QMATMUL(QK_K,  QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1); break;
        case 3: LAUNCH_RMS_QMATMUL(QK_K,  QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1); break;
        case 4: LAUNCH_RMS_QMATMUL(QK_K,  QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1); break;
        case 5: LAUNCH_RMS_QMATMUL(QK_K,  QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1); break;
        default: break;
    }
}
