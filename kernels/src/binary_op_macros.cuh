#include "cuda_utils.cuh"

#define LONG_BINARY_OP(TYPENAME, FORWARD, BACKWARD_LHS, BACKWARD_RHS, OP_STRUCT, FUNC, DFDX, DFDY) \
extern "C" __global__ void FORWARD( \
    const OP_STRUCT op, \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const TYPENAME *lhs, \
    const TYPENAME *rhs, \
    TYPENAME *out \
) { \
    const size_t *dims = info; \
    const size_t *lhs_strides = info + num_dims; \
    const size_t *rhs_strides = info + 2 * num_dims; \
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
        TYPENAME fx; \
        FUNC\
        out[i] = fx; \
    } \
} \
\
extern "C" __global__ void BACKWARD_LHS( \
    const OP_STRUCT op, \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const TYPENAME *lhs, \
    TYPENAME *grad_lhs, \
    const size_t chunk_len, \
    const TYPENAME *rhs, \
    const TYPENAME *grad_out \
) { \
    const size_t *dims = info + 0 * num_dims; \
    const size_t *out_strides = info + 1 * num_dims; \
    const size_t *rhs_strides = info + 2 * num_dims; \
    TYPENAME zero = 0.0; \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
        unsigned int tmp_i = i; \
        unsigned int out_i = 0; \
        unsigned int rhs_i = 0; \
        for (int d = num_dims - 1; d >= 0; d--) { \
            unsigned int i_dim = tmp_i % dims[d]; \
            out_i += i_dim * out_strides[d]; \
            rhs_i += i_dim * rhs_strides[d]; \
            tmp_i /= dims[d]; \
        } \
        TYPENAME x = lhs ? lhs[i / chunk_len] : zero; \
        TYPENAME y = rhs ? rhs[rhs_i] : zero; \
        TYPENAME go = grad_out[out_i]; \
        TYPENAME dfdx = (DFDX); \
        chunk_sum(chunk_len, dfdx * go, grad_lhs); \
    } \
} \
\
extern "C" __global__ void BACKWARD_RHS( \
    const OP_STRUCT op, \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const TYPENAME *lhs, \
    const TYPENAME *rhs, \
    TYPENAME *grad_rhs, \
    const size_t chunk_len, \
    const TYPENAME *grad_out \
) { \
    const size_t *dims = info + 3 * num_dims; \
    const size_t *out_strides = info + 4 * num_dims; \
    const size_t *lhs_strides = info + 5 * num_dims; \
    TYPENAME zero = 0.0; \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
        unsigned int tmp_i = i; \
        unsigned int lhs_i = 0; \
        unsigned int out_i = 0; \
        for (int d = num_dims - 1; d >= 0; d--) { \
            unsigned int i_dim = tmp_i % dims[d]; \
            lhs_i += i_dim * lhs_strides[d]; \
            out_i += i_dim * out_strides[d]; \
            tmp_i /= dims[d]; \
        } \
        TYPENAME x = lhs ? lhs[lhs_i] : zero; \
        TYPENAME y = rhs ? rhs[i / chunk_len] : zero; \
        TYPENAME go = grad_out[out_i]; \
        TYPENAME dfdy = (DFDY); \
        chunk_sum(chunk_len, dfdy * go, grad_rhs); \
    } \
}

#define BINARY_OP(TYPENAME, FORWARD, BACKWARD_LHS, BACKWARD_RHS, OP_STRUCT, FUNC, DFDX, DFDY) \
    LONG_BINARY_OP(TYPENAME, FORWARD, BACKWARD_LHS, BACKWARD_RHS, OP_STRUCT, fx = (FUNC);, DFDX, DFDY)
