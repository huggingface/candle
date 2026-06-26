// Fused dequantize + GEMM kernel for AWQ (AutoAWQ "GEMM" layout), 4-bit only.
//
// Same shared-memory-tiled GEMM structure as the GPTQ kernel in `candle-gptq-kernels`, but AWQ
// packs 4-bit values along the *output*-feature axis with a fixed nibble permutation (inherited
// from the original llm-awq CUDA kernel) instead of the input axis, and uses zero points as-is
// (no `+1` offset).
#include <cstdint>

#define TILE 32
#define AWQ_BITS 4
#define AWQ_PACK_FACTOR 8

// Inverse of AutoAWQ's packing `order_map = [0, 2, 4, 6, 1, 3, 5, 7]`: in-word position `i`
// belongs to output column `col_group * 8 + AWQ_ORDER[i]`.
__device__ __constant__ int AWQ_ORDER[8] = {0, 4, 1, 5, 2, 6, 3, 7};

extern "C" __global__ void awq_gemm_f32(
    const float *__restrict__ x,         // [M, K]
    const int32_t *__restrict__ qweight,  // [K, N / pack_factor]
    const int32_t *__restrict__ qzeros,   // [n_groups, N / pack_factor]
    const float *__restrict__ scales,     // [n_groups, N]
    float *__restrict__ y,                // [M, N]
    int M, int K, int N, int group_size, int n_packed_out /* N / pack_factor */) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  const int row = blockIdx.y * TILE + threadIdx.y; // index into M
  const int col = blockIdx.x * TILE + threadIdx.x; // index into N
  const int32_t mask = (1 << AWQ_BITS) - 1;

  float acc = 0.0f;
  const int n_tiles = (K + TILE - 1) / TILE;
  for (int t = 0; t < n_tiles; ++t) {
    const int a_col = t * TILE + threadIdx.x;
    As[threadIdx.y][threadIdx.x] =
        (row < M && a_col < K) ? x[row * K + a_col] : 0.0f;

    const int k = t * TILE + threadIdx.y;
    if (k < K && col < N) {
      const int g = k / group_size;
      const int col_group = col / AWQ_PACK_FACTOR;
      const int j = col % AWQ_PACK_FACTOR;
      const int shift = AWQ_ORDER[j] * AWQ_BITS;

      const int32_t w_word = qweight[k * n_packed_out + col_group];
      const int32_t q = (w_word >> shift) & mask;

      const int32_t z_word = qzeros[g * n_packed_out + col_group];
      const int32_t z = (z_word >> shift) & mask;

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

extern "C" void run_awq_gemm_f32(const float *x, const int32_t *qweight,
                                  const int32_t *qzeros, const float *scales,
                                  float *y, int M, int K, int N,
                                  int group_size, int n_packed_out) {
  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  awq_gemm_f32<<<grid, block>>>(x, qweight, qzeros, scales, y, M, K, N,
                                 group_size, n_packed_out);
}
