/**
 * @brief Fused quantized matmul + residual add for single-token decode.
 *
 * Computes: out = alpha * (W_q @ x_q8_1) + residual
 *
 * Replaces the typical two-launch sequence
 *   y  = qmatmul(W, x)         // ~1 launch (q8_1 quantize is internal)
 *   xs = y + residual          // 1 launch (broadcast_add)
 * with a single launch that writes the matmul partial sums via
 * atomicAdd into an output buffer pre-initialized with residual.
 *
 * Used for the post-attention `wo` output projection (and in principle
 * any matmul whose result feeds directly into a residual add).
 *
 * Per-block: one row of W's output. Partial sums per thread, warp
 * reduce, atomicAdd to dst[row]. Thanks to atomicAdd we can run with
 * any nwarps — each warp adds its independent contribution.
 */
#include "gguf.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

#ifndef MATRIX_ROW_PADDING
#define MATRIX_ROW_PADDING 512
#endif

namespace ll_qmma {

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
__global__ void qmatmul_add_kernel(
    const block_q_t* __restrict__ vx,        // [out_rows, hidden] quantized weight
    const block_q8_1* __restrict__ vy,       // [hidden / QK8_1] q8_1 input
    float* __restrict__ dst,                  // [out_rows] F32, PRE-INITIALIZED with residual
    int hidden,
    int out_rows
) {
    const int row = blockIdx.x;
    if (row >= out_rows) return;

    const int laneId = threadIdx.x;
    const int wrapId = threadIdx.y;
    const int nWraps = blockDim.y;
    const int tid    = wrapId * WARP_SIZE + laneId;

    const int blocks_per_row_x = hidden / qk;
    const int blocks_per_iter  = vdr * nWraps * WARP_SIZE / qi;

    const block_q_t* w_row = vx + (size_t)row * blocks_per_row_x;

    float acc = 0.0f;
    #pragma unroll
    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (laneId % (qi / vdr));
        acc += vec_dot_q_cuda(&w_row[kbx], &vy[kby], kqs);
    }

    // Warp reduce within each warp.
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
        acc += __shfl_xor_sync(0xFFFFFFFFu, acc, mask, WARP_SIZE);
    }
    // Cross-warp reduce via shared memory; only ONE thread per block
    // performs the atomicAdd into dst[row] to avoid the no-op
    // atomicAdd-of-zero from warps with no work.
    extern __shared__ float warp_sums[];
    if (laneId == 0) warp_sums[wrapId] = acc;
    __syncthreads();
    if (wrapId == 0 && laneId == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; w++) {
            if (w < nWraps) total += warp_sums[w];
        }
        atomicAdd(&dst[row], total);
    }
}

} // namespace ll_qmma

#define LAUNCH_QMA_ADD(qk, qi, block_q_t, vdr, vec_dot_q_cuda) \
    do { \
        const size_t smem = nWraps * sizeof(float); \
        ll_qmma::qmatmul_add_kernel<qk, qi, block_q_t, vdr, vec_dot_q_cuda> \
            <<<grid, block, smem, stream>>>( \
                (const block_q_t*)w_q, (const block_q8_1*)y_q8_1, dst, hidden, out_rows); \
    } while (0)

// Quantize F32 input → q8_1 then run the matmul-add. Caller must
// pre-initialize `dst` with the residual (a cudaMemcpyDtoDAsync of the
// F32 residual is fine). Caller must also provide a `y_q8_1_scratch`
// buffer of size `(((hidden + 511) / 512) * 512 / 32) * sizeof(block_q8_1)`.
extern "C" void qmatmul_add(
    const float* x,         // [hidden] F32 input
    const void*  w_q,       // [out_rows, hidden] quantized weight
    void*        y_q8_1_scratch, // q8_1 scratch buffer (caller-allocated)
    float*       dst,       // [out_rows] F32, must be pre-init with residual
    int hidden,
    int out_rows,
    int quant_type,         // 0=Q8_0, 1=Q4K, 2=Q2K, 3=Q3K, 4=Q5K, 5=Q6K
    cudaStream_t stream
) {
    const int kx_padded = ((hidden + MATRIX_ROW_PADDING - 1) / MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;
    void* y_q8_1 = y_q8_1_scratch;

    const int QUANTIZE_BLOCK_SIZE = CUDA_QUANTIZE_BLOCK_SIZE;
    const int num_blocks = (kx_padded + QUANTIZE_BLOCK_SIZE - 1) / QUANTIZE_BLOCK_SIZE;
    dim3 grid_q(num_blocks, 1, 1);
    dim3 block_q(QUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<grid_q, block_q, 0, stream>>>(x, y_q8_1, hidden, kx_padded);

    const int nWraps = 2;
    dim3 grid((unsigned)out_rows, 1, 1);
    dim3 block(WARP_SIZE, nWraps, 1);

    switch (quant_type) {
        case 0: LAUNCH_QMA_ADD(QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1); break;
        case 1: LAUNCH_QMA_ADD(QK_K,  QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1); break;
        case 2: LAUNCH_QMA_ADD(QK_K,  QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1); break;
        case 3: LAUNCH_QMA_ADD(QK_K,  QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1); break;
        case 4: LAUNCH_QMA_ADD(QK_K,  QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1); break;
        case 5: LAUNCH_QMA_ADD(QK_K,  QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1); break;
        default: break;
    }
}
