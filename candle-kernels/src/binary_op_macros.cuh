#include "cuda_utils.cuh"

#define BINARY_OP_OUT(TYPENAME, OUT_TYPENAME, FN_NAME, FUNC) \
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
            out[i] = FUNC; \
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
            out[i] = FUNC; \
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
            out[i] = FUNC; \
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
            out[i] = FUNC; \
        } \
    } \
} \


#define BINARY_OP(TYPENAME, FN_NAME, FUNC) \
  BINARY_OP_OUT(TYPENAME, TYPENAME, FN_NAME, FUNC)

// Vectorized binary ops for bf16 — 8 elements per float4 load for contiguous case
#if __CUDA_ARCH__ >= 800
#define BINARY_OP_BF16_VEC(FN_NAME, FLOAT_OP) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *dims_and_strides, \
    const __nv_bfloat16 *lhs, \
    const __nv_bfloat16 *rhs, \
    __nv_bfloat16 *out \
) { \
    const size_t *dims = dims_and_strides; \
    const size_t *lhs_strides = dims_and_strides + 1 * num_dims; \
    const size_t *rhs_strides = dims_and_strides + 2 * num_dims; \
    bool lhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, lhs_strides); \
    bool rhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, rhs_strides); \
    if (lhs_cont && rhs_cont) { \
        const size_t vec_numel = numel / 8; \
        const float4 *lhs4 = reinterpret_cast<const float4*>(lhs); \
        const float4 *rhs4 = reinterpret_cast<const float4*>(rhs); \
        float4 *out4 = reinterpret_cast<float4*>(out); \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_numel; i += blockDim.x * gridDim.x) { \
            float4 a = lhs4[i]; \
            float4 b = rhs4[i]; \
            __nv_bfloat16 *ap = reinterpret_cast<__nv_bfloat16*>(&a); \
            __nv_bfloat16 *bp = reinterpret_cast<__nv_bfloat16*>(&b); \
            _Pragma("unroll") \
            for (int j = 0; j < 8; j++) { \
                float xf = __bfloat162float(ap[j]); \
                float yf = __bfloat162float(bp[j]); \
                ap[j] = __float2bfloat16(FLOAT_OP); \
            } \
            out4[i] = a; \
        } \
        const size_t tail_start = vec_numel * 8; \
        for (unsigned int i = tail_start + blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            __nv_bfloat16 x = lhs[i]; \
            __nv_bfloat16 y = rhs[i]; \
            out[i] = x FLOAT_OP y; \
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
            __nv_bfloat16 x = lhs[i]; \
            __nv_bfloat16 y = rhs[rhs_i]; \
            out[i] = x FLOAT_OP y; \
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
            __nv_bfloat16 x = lhs[lhs_i]; \
            __nv_bfloat16 y = rhs[i]; \
            out[i] = x FLOAT_OP y; \
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
            __nv_bfloat16 x = lhs[lhs_i]; \
            __nv_bfloat16 y = rhs[rhs_i]; \
            out[i] = x FLOAT_OP y; \
        } \
    } \
}
#endif
