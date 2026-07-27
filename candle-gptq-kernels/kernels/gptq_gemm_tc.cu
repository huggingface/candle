// Tensor-core fused dequantize+GEMM kernel for GPTQ 4-bit weights (bits=4 only).
//
// Same numerical semantics as the scalar kernel in `gptq_gemm.cu` (AutoGPTQ/GPTQModel "old"
// CUDA layout: qweight/qzeros/scales/g_idx, zero point stored as `value - 1`), but the GEMM
// inner product is computed on the SM's tensor cores via the WMMA API instead of scalar FMAs:
// each 16x16x16 partial product runs through `wmma::mma_sync` on FP16 operands with FP32
// accumulation, after dequantizing the relevant 16x16 weight tile into shared memory as FP16.
//
// This is a from-scratch tensor-core kernel, not a port of Neural Magic's Marlin: it reads the
// standard AutoGPTQ tensor layout directly (no offline weight repacking step) and uses a simple
// one-warp-per-output-tile schedule (no `cp.async` pipelining, no Stream-K decomposition). It is
// nonetheless a genuine tensor-core kernel, unlike the scalar-FMA kernel in `gptq_gemm.cu`.
#include <cstdint>
#include <mma.h>

using namespace nvcuda;

#define GPTQ_TC_BITS 4
#define GPTQ_TC_PACK_FACTOR 8
#define GPTQ_TC_MASK 0xF
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

extern "C" __global__ void gptq_gemm_tc_f32(
    const float *__restrict__ x,         // [M, K]
    const int32_t *__restrict__ qweight, // [K / pack_factor, N]
    const int32_t *__restrict__ qzeros,  // [n_groups, N / pack_factor]
    const float *__restrict__ scales,    // [n_groups, N]
    const int32_t *__restrict__ g_idx,   // [K], group index per input row
    float *__restrict__ y,               // [M, N]
    int M, int K, int N, int n_groups_out /* N / pack_factor */) {
  __shared__ __align__(16) half As[WMMA_M][WMMA_K];
  __shared__ __align__(16) half Bs[WMMA_K][WMMA_N];
  __shared__ __align__(16) float Cs[WMMA_M][WMMA_N];

  const int tile_row = blockIdx.y * WMMA_M; // base row into M
  const int tile_col = blockIdx.x * WMMA_N; // base col into N
  const int tid = threadIdx.x;              // 0..31, one warp per block

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
      b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (int k0 = 0; k0 < K; k0 += WMMA_K) {
    // Stage the activation tile into shared memory as half, zero-padding out-of-bounds.
    for (int i = tid; i < WMMA_M * WMMA_K; i += 32) {
      const int r = i / WMMA_K;
      const int c = i % WMMA_K;
      const int row = tile_row + r;
      const int k = k0 + c;
      As[r][c] = (row < M && k < K) ? __float2half(x[row * K + k])
                                     : __float2half(0.0f);
    }

    // Stage the dequantized weight tile into shared memory as half, zero-padding OOB.
    for (int i = tid; i < WMMA_K * WMMA_N; i += 32) {
      const int r = i / WMMA_N; // local k index
      const int c = i % WMMA_N; // local col index
      const int k = k0 + r;
      const int col = tile_col + c;
      if (k < K && col < N) {
        const int g = g_idx[k];
        const int32_t w_word = qweight[(k / GPTQ_TC_PACK_FACTOR) * N + col];
        const int shift_q = (k % GPTQ_TC_PACK_FACTOR) * GPTQ_TC_BITS;
        const int32_t q = (w_word >> shift_q) & GPTQ_TC_MASK;

        const int32_t z_word =
            qzeros[g * n_groups_out + col / GPTQ_TC_PACK_FACTOR];
        const int shift_z = (col % GPTQ_TC_PACK_FACTOR) * GPTQ_TC_BITS;
        const int32_t z = ((z_word >> shift_z) & GPTQ_TC_MASK) + 1;

        const float s = scales[g * N + col];
        Bs[r][c] = __float2half((float)(q - z) * s);
      } else {
        Bs[r][c] = __float2half(0.0f);
      }
    }
    __syncthreads();

    wmma::load_matrix_sync(a_frag, &As[0][0], WMMA_K);
    wmma::load_matrix_sync(b_frag, &Bs[0][0], WMMA_N);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  wmma::store_matrix_sync(&Cs[0][0], c_frag, WMMA_N, wmma::mem_row_major);
  __syncthreads();

  for (int i = tid; i < WMMA_M * WMMA_N; i += 32) {
    const int r = i / WMMA_N;
    const int c = i % WMMA_N;
    const int row = tile_row + r;
    const int col = tile_col + c;
    if (row < M && col < N) {
      y[row * N + col] = Cs[r][c];
    }
  }
}

extern "C" void run_gptq_gemm_tc_f32(const float *x, const int32_t *qweight,
                                      const int32_t *qzeros,
                                      const float *scales,
                                      const int32_t *g_idx, float *y, int M,
                                      int K, int N, int n_groups_out) {
  dim3 block(32, 1);
  dim3 grid((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
  gptq_gemm_tc_f32<<<grid, block>>>(x, qweight, qzeros, scales, g_idx, y, M, K,
                                     N, n_groups_out);
}
