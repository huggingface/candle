#include<stdint.h>
#include "cuda_fp16.h"

template<typename T>
__device__ void fill_with(T *buf, T value, const size_t numel) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        buf[i] = value;
    }
}
extern "C" __global__ void fill_u8(uint8_t *buf, uint8_t value, const size_t numel) { fill_with(buf, value, numel); }
extern "C" __global__ void fill_u32(uint32_t *buf, uint32_t value, const size_t numel) { fill_with(buf, value, numel); }
extern "C" __global__ void fill_i64(int64_t *buf, int64_t value, const size_t numel) { fill_with(buf, value, numel); }
extern "C" __global__ void fill_f16(__half *buf, __half value, const size_t numel) { fill_with(buf, value, numel); }
extern "C" __global__ void fill_f32(float *buf, float value, const size_t numel) { fill_with(buf, value, numel); }
extern "C" __global__ void fill_f64(double *buf, double value, const size_t numel) { fill_with(buf, value, numel); }

#if __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
extern "C" __global__ void fill_bf16(__nv_bfloat16 *buf, __nv_bfloat16 value, const size_t numel) { fill_with(buf, value, numel); }
#endif

extern "C" __global__ void copy2d(const uint8_t *src, uint8_t *dst, uint32_t d1, uint32_t d2, uint32_t src_s, uint32_t dst_s) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= d1) {
    return;
  }
  memcpy(dst + idx * dst_s, src + idx * src_s, d2);
}
