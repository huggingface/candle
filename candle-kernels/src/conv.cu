#include "cuda_utils.cuh"
#include<stdint.h>

template <typename T, typename A>
__device__ void conv1d(
    const size_t src_numel,
    const size_t l_out,
    const size_t stride, 
    const size_t *info,
    const T *src,
    const T *kernel,
    T *dst
) {
  // src: (b_size, c_in, l_in)
  // k: (c_out, c_in, k_size)
  const size_t *src_dims = info;
  const size_t *src_s = info + 3;
  const size_t *k_dims = info + 6;
  const size_t *k_s = info + 9;
  const size_t dst_i = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t k_size = k_dims[2];
  const size_t k_over_2 = k_size / 2;
  const size_t c_out = k_dims[0];
  const size_t c_in = src_dims[1];
  const size_t l_in = src_dims[2];

  // TODO
  const size_t b_idx = dst_i / (l_out * c_out);
  const size_t dst_c_idx = (dst_i / l_out) % c_out;
  const size_t dst_l = dst_i % l_out;

  const size_t src_idx0 = b_idx * src_s[0];
  A d = 0;
  for (size_t offset = 0; offset < k_size; ++offset) {
    const size_t src_l_plus = stride * dst_l + offset;
    if (k_over_2 <= src_l_plus && src_l_plus < l_in + k_over_2) {
      const size_t src_l = src_l_plus - k_over_2;
      for (size_t src_c_idx = 0; src_c_idx < c_in; ++src_c_idx) {
        const size_t src_idx = src_idx0 + src_c_idx * src_s[1] + src_l * src_s[2];
        const size_t k_idx = dst_c_idx * k_s[0] + src_c_idx * k_s[1] + offset * k_s[2];
        d += static_cast<A>(src[src_idx]) * static_cast<A>(kernel[k_idx]);
      }
    }
  }
  dst[dst_i] = static_cast<T>(d);
}


#define CONV1D_OP(TYPENAME, TYPEACC, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t src_numel, \
    const size_t num_dims, \
    const size_t stride, \
    const size_t *info, \
    const TYPENAME *src, \
    const TYPENAME *kernel, \
    TYPENAME *dst \
) {  \
  conv1d<TYPENAME, TYPEACC>(src_numel, num_dims, stride, info, src, kernel, dst); \
} \

#if __CUDA_ARCH__ >= 800
CONV1D_OP(__nv_bfloat16, float, conv1d_bf16)
#endif

#if __CUDA_ARCH__ >= 530
CONV1D_OP(__half, float, conv1d_f16)
#endif

CONV1D_OP(float, float, conv1d_f32)
CONV1D_OP(double, double, conv1d_f64)
CONV1D_OP(uint8_t, uint8_t, conv1d_u8)
CONV1D_OP(uint32_t, uint32_t, conv1d_u32)

