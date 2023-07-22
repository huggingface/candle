#include "reduction_utils.cuh"

template <typename scalar_t>
__device__ void
rms_norm_kernel(scalar_t *__restrict__ out,         // [num_tokens, hidden_size]
                const scalar_t *__restrict__ input, // [num_tokens, hidden_size]
                const scalar_t *__restrict__ weight, // [hidden_size]
                const float epsilon, const int num_tokens,
                const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = (float)input[blockIdx.x * hidden_size + idx];
    variance += x * x;
  }
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)input[blockIdx.x * hidden_size + idx];
    out[blockIdx.x * hidden_size + idx] =
        ((scalar_t)(x * s_variance)) * weight[idx];
  }
}
extern "C" __global__ void rms_norm_kernel_f32(
    float *__restrict__ out,         // [num_tokens, hidden_size]
    const float *__restrict__ input, // [num_tokens, hidden_size]
    const float *__restrict__ weight, // [hidden_size]
    const float epsilon, const int num_tokens,
    const int hidden_size) {
  rms_norm_kernel(out, input, weight, epsilon, num_tokens, hidden_size);
}

