// WARNING: THIS IS ONLY VALID ASSUMING THAT inp IS CONTIGUOUS!
// TODO: proper error reporting when ids are larger than v_size.
#include "cuda_utils.cuh"
#include<stdint.h>

#define EMB_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t numel,  \
    const size_t num_dims, \
    const size_t *info, \
    const uint32_t *ids, \
    const TYPENAME *inp, \
    TYPENAME *out, \
    const size_t h_size, \
    const size_t v_size \
) {  \
    const size_t *dims = info; \
    const size_t *strides = info + num_dims; \
    if (is_contiguous(num_dims, dims, strides)) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            memcpy(out + i * h_size, inp + ids[i], h_size); \
        } \
    } \
    else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides); \
            memcpy(out + i * h_size, inp + ids[i], h_size); \
        } \
    } \
} \

EMB_OP(float, emb_f32)
EMB_OP(double, emb_f64)
EMB_OP(uint32_t, emb_u32)
