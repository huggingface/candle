#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr int kBlockSize = 16;

__device__ __forceinline__ float e2m1_to_f32(uint8_t x) {
  constexpr float table[16] = {0., .5, 1., 1.5, 2., 3., 4., 6.,
                                0., -.5, -1., -1.5, -2., -3., -4., -6.};
  return table[x & 0xf];
}

// NVIDIA's FP8 E4M3FN encoding, used for the per-16-value NVFP4 scales.
__device__ __forceinline__ float e4m3_to_f32(uint8_t x) {
  const float sign = (x & 0x80) ? -1.f : 1.f;
  const int exponent = (x >> 3) & 0xf;
  const int mantissa = x & 0x7;
  if (exponent == 0) return sign * mantissa * 0.001953125f;
  // E4M3FN has no infinities; 0x7f/0xff encode NaN.
  if (exponent == 0xf && mantissa == 0x7) return nanf("");
  return sign * (1.f + mantissa * .125f) * exp2f(float(exponent - 7));
}

__global__ void linear_f32(const float* input, const uint8_t* weight,
                           const uint8_t* scales, float tensor_scale,
                           float* output, int64_t tokens, int64_t rows,
                           int64_t cols) {
  const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= tokens * rows) return;
  const int64_t token = index / rows;
  const int64_t row = index % rows;
  const int64_t packed_cols = cols / 2;
  const int64_t scale_cols = cols / kBlockSize;
  float sum = 0.f;
  for (int64_t col = 0; col < cols; ++col) {
    const uint8_t packed = weight[row * packed_cols + col / 2];
    const uint8_t nibble = (col & 1) ? packed & 0xf : packed >> 4;
    const float scale = e4m3_to_f32(scales[row * scale_cols + col / kBlockSize]);
    sum += input[token * cols + col] * e2m1_to_f32(nibble) * scale * tensor_scale;
  }
  output[index] = sum;
}

bool valid(int64_t rows, int64_t cols, int64_t tokens) {
  return rows > 0 && cols > 0 && tokens > 0 && cols % kBlockSize == 0;
}
}  // namespace

extern "C" int candle_nvfp4_cuda_is_available() {
  int device;
  cudaDeviceProp properties{};
  return cudaGetDevice(&device) == cudaSuccess &&
                 cudaGetDeviceProperties(&properties, device) == cudaSuccess &&
                 properties.major >= 10
             ? 1
             : 0;
}

extern "C" int candle_nvfp4_linear_f32_host_batched(
    const float* input, const uint8_t* weight, const uint8_t* scales,
    float tensor_scale, float* output, int64_t tokens, int64_t rows, int64_t cols) {
  if (!valid(rows, cols, tokens) || !input || !weight || !scales || !output)
    return cudaErrorInvalidValue;
  float *d_input = nullptr, *d_output = nullptr;
  uint8_t *d_weight = nullptr, *d_scales = nullptr;
  cudaError_t status = cudaMalloc(&d_input, size_t(tokens * cols) * sizeof(float));
  if (status != cudaSuccess) goto cleanup;
  status = cudaMalloc(&d_weight, size_t(rows * cols / 2));
  if (status != cudaSuccess) goto cleanup;
  status = cudaMalloc(&d_scales, size_t(rows * cols / kBlockSize));
  if (status != cudaSuccess) goto cleanup;
  status = cudaMalloc(&d_output, size_t(tokens * rows) * sizeof(float));
  if (status != cudaSuccess) goto cleanup;
  if ((status = cudaMemcpy(d_input, input, size_t(tokens * cols) * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess ||
      (status = cudaMemcpy(d_weight, weight, size_t(rows * cols / 2), cudaMemcpyHostToDevice)) != cudaSuccess ||
      (status = cudaMemcpy(d_scales, scales, size_t(rows * cols / kBlockSize), cudaMemcpyHostToDevice)) != cudaSuccess) goto cleanup;
  linear_f32<<<unsigned((tokens * rows + 255) / 256), 256>>>(d_input, d_weight, d_scales, tensor_scale, d_output, tokens, rows, cols);
  if ((status = cudaGetLastError()) != cudaSuccess) goto cleanup;
  status = cudaMemcpy(output, d_output, size_t(tokens * rows) * sizeof(float), cudaMemcpyDeviceToHost);
cleanup:
  cudaFree(d_output); cudaFree(d_scales); cudaFree(d_weight); cudaFree(d_input);
  return status;
}
