// ROCm/HIP binary operation kernels
// These are embedded as strings and compiled at runtime

#ifndef __HIPCC__
#define __device__
#define __global__
#define __forceinline__
#endif

#include <stddef.h>
#include <stdint.h>

// Helper to check if tensor is contiguous
__device__ bool is_contiguous(
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides
) {
    size_t acc = 1;
    for (unsigned int d = 0; d < num_dims; d++) {
        unsigned int dim_idx = num_dims - 1 - d;
        if (dims[dim_idx] > 1 && acc != strides[dim_idx]) {
            return false;
        }
        acc *= dims[dim_idx];
    }
    return true;
}

// Binary operation macro - generates kernel for in-place operation
#define BINARY_OP(TYPENAME, FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *dims_and_strides, \
    const TYPENAME *lhs, \
    const TYPENAME *rhs, \
    TYPENAME *out \
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
}

// badd kernels
BINARY_OP(float, badd_f32, x + y)
BINARY_OP(double, badd_f64, x + y)
BINARY_OP(uint8_t, badd_u8, x + y)
BINARY_OP(uint32_t, badd_u32, x + y)
BINARY_OP(int64_t, badd_i64, x + y)

// bdiv kernels
BINARY_OP(float, bdiv_f32, x / y)
BINARY_OP(double, bdiv_f64, x / y)
BINARY_OP(uint8_t, bdiv_u8, x / y)
BINARY_OP(uint32_t, bdiv_u32, x / y)
BINARY_OP(int64_t, bdiv_i64, x / y)

// bmul kernels
BINARY_OP(float, bmul_f32, x * y)
BINARY_OP(double, bmul_f64, x * y)
BINARY_OP(uint8_t, bmul_u8, x * y)
BINARY_OP(uint32_t, bmul_u32, x * y)
BINARY_OP(int64_t, bmul_i64, x * y)

// bsub kernels
BINARY_OP(float, bsub_f32, x - y)
BINARY_OP(double, bsub_f64, x - y)
BINARY_OP(uint8_t, bsub_u8, x - y)
BINARY_OP(uint32_t, bsub_u32, x - y)
BINARY_OP(int64_t, bsub_i64, x - y)

// bminimum kernels
BINARY_OP(float, bminimum_f32, (x < y ? x : y))
BINARY_OP(double, bminimum_f64, (x < y ? x : y))
BINARY_OP(uint8_t, bminimum_u8, (x < y ? x : y))
BINARY_OP(uint32_t, bminimum_u32, (x < y ? x : y))
BINARY_OP(int64_t, bminimum_i64, (x < y ? x : y))

// bmaximum kernels
BINARY_OP(float, bmaximum_f32, (x > y ? x : y))
BINARY_OP(double, bmaximum_f64, (x > y ? x : y))
BINARY_OP(uint8_t, bmaximum_u8, (x > y ? x : y))
BINARY_OP(uint32_t, bmaximum_u32, (x > y ? x : y))
BINARY_OP(int64_t, bmaximum_i64, (x > y ? x : y))
