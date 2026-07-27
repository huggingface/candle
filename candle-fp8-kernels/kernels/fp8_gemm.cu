// Fused dequantize + GEMM kernel for block-wise FP8 (E4M3) checkpoints (DeepSeek-V3 style).
//
// Computes Y[M,N] = X[M,K] @ W^T, where W is stored as `[N, K]` F8E4M3 plus a per-block f32
// scale `scale[N / block_size, K / block_size]` (i.e. `W[n,k] = fp8[n,k] * scale[n/bs, k/bs]`),
// the same convention `candle_transformers::quantized_fp8` dequantizes at load time. This kernel
// fuses that dequantization into a shared-memory-tiled GEMM instead of materializing a dense
// weight. Accumulation is scalar FP32; it does not use tensor cores.
#include <cstdint>
#include <cuda_fp8.h>

#define TILE 32

extern "C" __global__ void fp8_block_gemm_f32(
    const float *__restrict__ x,          // [M, K]
    const __nv_fp8_e4m3 *__restrict__ w,  // [N, K]
    const float *__restrict__ scale,      // [N / block_size, K / block_size]
    float *__restrict__ y,                // [M, N]
    int M, int K, int N, int block_size, int scale_cols /* ceil(K / block_size) */) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  const int row = blockIdx.y * TILE + threadIdx.y; // index into M
  const int col = blockIdx.x * TILE + threadIdx.x; // index into N

  float acc = 0.0f;
  const int n_tiles = (K + TILE - 1) / TILE;
  for (int t = 0; t < n_tiles; ++t) {
    const int a_col = t * TILE + threadIdx.x;
    As[threadIdx.y][threadIdx.x] =
        (row < M && a_col < K) ? x[row * K + a_col] : 0.0f;

    const int k = t * TILE + threadIdx.y;
    if (k < K && col < N) {
      const float w_val = (float)w[col * K + k];
      const float s = scale[(col / block_size) * scale_cols + k / block_size];
      Bs[threadIdx.y][threadIdx.x] = w_val * s;
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

extern "C" void run_fp8_block_gemm_f32(const float *x, const void *w,
                                        const float *scale, float *y, int M,
                                        int K, int N, int block_size,
                                        int scale_cols) {
  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  fp8_block_gemm_f32<<<grid, block>>>(x, (const __nv_fp8_e4m3 *)w, scale, y, M,
                                       K, N, block_size, scale_cols);
}
