// WARNING: THIS IS ONLY VALID ASSUMING THAT inp IS CONTIGUOUS!
// TODO: proper error reporting when ids are larger than v_size.
#include "cuda_utils.cuh"
#include<stdint.h>

#define EMB_OP(TYPENAME, INDEX_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t numel,  \
    const size_t num_dims, \
    const size_t *info, \
    const INDEX_TYPENAME *ids, \
    const TYPENAME *inp, \
    TYPENAME *out, \
    const size_t h_size, \
    const size_t v_size \
) {  \
    const size_t *dims = info; \
    const size_t *strides = info + num_dims; \
    if (is_contiguous(num_dims, dims, strides)) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            memcpy(&out[i * h_size], &inp[ids[i] * h_size], h_size * sizeof(TYPENAME)); \
        } \
    } \
    else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides); \
            memcpy(&out[i * h_size], &inp[ids[strided_i] * h_size], h_size * sizeof(TYPENAME)); \
        } \
    } \
} \

#if __CUDA_ARCH__ >= 800
EMB_OP(__nv_bfloat16, uint32_t, emb_u32_bf16)
EMB_OP(__nv_bfloat16, uint8_t, emb_u8_bf16)
#endif

#if __CUDA_ARCH__ >= 530
EMB_OP(__half, uint32_t, emb_u32_f16)
EMB_OP(__half, uint8_t, emb_u8_f16)
#endif

EMB_OP(float, uint32_t, emb_u32_f32)
EMB_OP(double, uint32_t, emb_u32_f64)
EMB_OP(uint8_t, uint32_t, emb_u32_u8)
EMB_OP(uint32_t, uint32_t, emb_u32_u32)

EMB_OP(float, uint8_t, emb_u8_f32)
EMB_OP(double, uint8_t, emb_u8_f64)
EMB_OP(uint8_t, uint8_t, emb_u8_u8)
EMB_OP(uint32_t, uint8_t, emb_u8_u32)
