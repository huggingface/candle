#include "cuda_utils.cuh"
#include<stdint.h>

template <typename T>
__device__ __forceinline__ void kvconcat_dim0_kernel(T *ltensor, T* rtensor, T *out,
    const size_t chunk_l, const size_t chunk_r, const size_t lstride, const size_t rstride) {
    size_t idx = GetThreadIdx();
    if (idx < chunk_l * lstride) {
        out[idx] = ltensor[idx];
    } else {
        out[idx] = rtensor[idx - chunk_l * lstride];
    }
}
template <typename T>
__device__ __forceinline__ void kvconcat_dim2_kernel(T *ltensor, T* rtensor, T *out,
    const size_t chunk_l, const size_t chunk_r, const size_t lstride, const size_t rstride) {
    int thread_id = GetThreadIdx();
    int out_stride = lstride + rstride;
    int idx = thread_id / out_stride;
    int j = thread_id % out_stride;
    T* pLeft = ltensor + idx * lstride;
    T* pRight = rtensor + idx * rstride;
    T* pOut = out + idx * out_stride;
    if (idx < chunk_l) {
        if (j < lstride)
            pOut[j] = pLeft[j];
        else
            pOut[j] = pRight[j - lstride];
    }
}

#define KVCONCAT_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(TYPENAME *ltensor, TYPENAME* rtensor, TYPENAME *out, const size_t concat_dim,\
    const size_t chunk_l, const size_t chunk_r, const size_t lstride, const size_t rstride) {\
    if (concat_dim == 2)\
      kvconcat_dim2_kernel<TYPENAME>(ltensor, rtensor, out, chunk_l, chunk_r, lstride, rstride);\
    else if (concat_dim == 0) {\
      if (blockIdx.x == 0 && threadIdx.x ==0) \
        kvconcat_dim0_kernel<TYPENAME>(ltensor, rtensor, out, chunk_l, chunk_r, lstride, rstride);\
    }\
}\

KVCONCAT_OP(uint8_t, kvconcat_u8)
KVCONCAT_OP(double, kvconcat_f64)
KVCONCAT_OP(float, kvconcat_f32)

#if __CUDA_ARCH__ >= 530
KVCONCAT_OP(__half, kvconcat_f16)
#endif

#if __CUDA_ARCH__ >= 800
KVCONCAT_OP(__nv_bfloat16, kvconcat_bf16)
#endif
