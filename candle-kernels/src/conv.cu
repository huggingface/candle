#include "cuda_utils.cuh"
#include<stdint.h>

template <typename T>
__device__ void conv1d(
    const size_t src_numel,
    const size_t num_dims, 
    const size_t padding, 
    const size_t stride, 
    const size_t *info,
    const T *src,
    const T *kernel,
    T *dst
) {
  const size_t *dims = info;
  const size_t *strides = info + num_dims;
  const size_t *k_dims = info + 2 * num_dims;
  const size_t *k_strides = info + 3 * num_dims;
}


#define CONV1D_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t src_numel, \
    const size_t num_dims, \
    const size_t padding, \
    const size_t stride, \
    const size_t *info, \
    const TYPENAME *src, \
    const TYPENAME *kernel, \
    TYPENAME *dst \
) {  \
  conv1d(src_numel, num_dims, padding, stride, info, src, kernel, dst); \
} \

#if __CUDA_ARCH__ >= 800
CONV1D_OP(__nv_bfloat16, conv1d_bf16)
#endif

#if __CUDA_ARCH__ >= 530
CONV1D_OP(__half, conv1d_f16)
#endif

CONV1D_OP(float, conv1d_f32)
CONV1D_OP(double, conv1d_f64)
CONV1D_OP(uint8_t, conv1d_u8)
CONV1D_OP(uint32_t, conv1d_u32)

