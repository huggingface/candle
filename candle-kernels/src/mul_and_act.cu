#include "cuda_utils.cuh"

template<typename T>
__device__ __forceinline__ T gelu(T x) {
    T x_sq = x * x;
    T x_cube = x_sq * x;
    T alpha = x + static_cast<T>(0.044715) * x_cube;
    return static_cast<T>(0.5) * x * (static_cast<T>(1.0) + tanhg(static_cast<T>(M_2_SQRTPI * M_SQRT1_2) * alpha));
}

template<typename T>
__device__ __forceinline__ T relu(T x) {
    T zero = 0.;
    return maxg(x, zero);
}

template<typename T>
__device__ __forceinline__ T silu(T x) {
    return x / (static_cast<T>(1) + expg(-x));
}


#define MUL_ACT_OP_OUT(TYPENAME, OUT_TYPENAME, FN_NAME, ACT) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *dims_and_strides, \
    const TYPENAME *lhs, \
    const TYPENAME *rhs, \
    OUT_TYPENAME *out \
) { \
    const size_t *dims = dims_and_strides; \
    const size_t *lhs_strides = dims_and_strides + 1 * num_dims; \
    const size_t *rhs_strides = dims_and_strides + 2 * num_dims; \
    bool lhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, lhs_strides); \
    bool rhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, rhs_strides); \
    if (lhs_cont && rhs_cont) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            TYPENAME x = lhs[i]; \
            TYPENAME y = rhs[i]; \
            out[i] = TYPENAME(ACT(float(x)) * float(y)); \
        } \
    } else if (lhs_cont) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int rhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int i_dim = tmp_i % dims[d]; \
                rhs_i += i_dim * rhs_strides[d]; \
                tmp_i /= dims[d]; \
            } \
            TYPENAME x = lhs[i]; \
            TYPENAME y = rhs[rhs_i]; \
            out[i] = TYPENAME(ACT(float(x)) * float(y)); \
        } \
    } else if (rhs_cont) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int lhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int i_dim = tmp_i % dims[d]; \
                lhs_i += i_dim * lhs_strides[d]; \
                tmp_i /= dims[d]; \
            } \
            TYPENAME x = lhs[lhs_i]; \
            TYPENAME y = rhs[i]; \
            out[i] = TYPENAME(ACT(float(x)) * float(y)); \
        } \
    } else { \
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
            TYPENAME x = lhs[lhs_i]; \
            TYPENAME y = rhs[rhs_i]; \
            out[i] = TYPENAME(ACT(float(x)) * float(y)); \
        } \
    } \
} \


#define MUL_ACT_OP(TYPENAME, FN_NAME, ACT) \
  MUL_ACT_OP_OUT(TYPENAME, TYPENAME, FN_NAME, ACT)

#if __CUDA_ARCH__ >= 800
#include "cuda_bf16.h"

MUL_ACT_OP(__nv_bfloat16, mul_act_gelu_bf16, gelu)
MUL_ACT_OP(__nv_bfloat16, mul_act_relu_bf16, relu)
MUL_ACT_OP(__nv_bfloat16, mul_act_silu_bf16, silu)
#endif

#if __CUDA_ARCH__ >= 530
MUL_ACT_OP(__half, mul_act_gelu_f16, gelu)
MUL_ACT_OP(__half, mul_act_relu_f16, relu)
MUL_ACT_OP(__half, mul_act_silu_f16, silu)
#endif

MUL_ACT_OP(float, mul_act_gelu_f32, gelu)
MUL_ACT_OP(float, mul_act_relu_f32, relu)
MUL_ACT_OP(float, mul_act_silu_f32, silu)
