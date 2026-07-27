// Fused dequantize + GEMM kernel for GPTQ (AutoGPTQ/GPTQModel) "old" CUDA layout.
//
// Computes Y[M,N] = X[M,K] @ dequant(qweight,qzeros,scales,g_idx)[K,N] using a
// shared-memory-tiled GEMM, dequantizing each weight element on the fly while it is staged
// into shared memory. Accumulation is scalar FP32 (no tensor cores): a correct,
// reasonably cache-friendly fused kernel, one step above "dequantize then call cuBLAS".
#include <cstdint>

#define TILE 32

extern "C" __global__ void gptq_gemm_f32(
    const float *__restrict__ x,        // [M, K]
    const int32_t *__restrict__ qweight, // [K / pack_factor, N]
    const int32_t *__restrict__ qzeros,  // [n_groups, N / pack_factor]
    const float *__restrict__ scales,    // [n_groups, N]
    const int32_t *__restrict__ g_idx,   // [K], group index per input row
    float *__restrict__ y,               // [M, N]
    int M, int K, int N,
    int bits, int pack_factor, int n_groups_out /* N / pack_factor */) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  const int row = blockIdx.y * TILE + threadIdx.y; // index into M
  const int col = blockIdx.x * TILE + threadIdx.x; // index into N
  const int32_t mask = (1 << bits) - 1;

  float acc = 0.0f;
  const int n_tiles = (K + TILE - 1) / TILE;
  for (int t = 0; t < n_tiles; ++t) {
    const int a_col = t * TILE + threadIdx.x;
    As[threadIdx.y][threadIdx.x] =
        (row < M && a_col < K) ? x[row * K + a_col] : 0.0f;

    const int k = t * TILE + threadIdx.y;
    if (k < K && col < N) {
      const int g = g_idx[k];
      const int32_t w_word = qweight[(k / pack_factor) * N + col];
      const int shift_q = (k % pack_factor) * bits;
      const int32_t q = (w_word >> shift_q) & mask;

      const int32_t z_word = qzeros[g * n_groups_out + col / pack_factor];
      const int shift_z = (col % pack_factor) * bits;
      const int32_t z = ((z_word >> shift_z) & mask) + 1;

      const float s = scales[g * N + col];
      Bs[threadIdx.y][threadIdx.x] = (float)(q - z) * s;
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < TILE; ++i) {
      acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    y[row * N + col] = acc;
  }
}

extern "C" void run_gptq_gemm_f32(
    const float *x, const int32_t *qweight, const int32_t *qzeros,
    const float *scales, const int32_t *g_idx, float *y, int M, int K, int N,
    int bits, int pack_factor, int n_groups_out) {
  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  gptq_gemm_f32<<<grid, block>>>(x, qweight, qzeros, scales, g_idx, y, M, K, N,
                                  bits, pack_factor, n_groups_out);
}
