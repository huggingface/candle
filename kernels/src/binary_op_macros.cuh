#include "cuda_utils.cuh"

#define BINARY_OP(TYPENAME, FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *dims, \
    const size_t *lhs_strides, \
    const size_t *rhs_strides, \
    const TYPENAME *lhs, \
    const TYPENAME *rhs, \
    TYPENAME *out \
) { \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
        unsigned int tmp_i = i; \
        unsigned int lhs_i = 0; \
        unsigned int rhs_i = 0; \
        for (int d = num_dims - 1; d >= 0; d--) { \
            unsigned int i_dim = tmp_i % dims[d]; \
            lhs_i += i_dim * lhs_strides[d]; \
            rhs_i += i_dim * rhs_strides[d]; \
            tmp_i /= dims[d]; \
        } \
        TYPENAME x = lhs ? lhs[lhs_i] : out[i]; \
        TYPENAME y = rhs ? rhs[rhs_i] : out[i]; \
        out[i] = FUNC; \
    } \
} \
