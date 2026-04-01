#ifndef __HIPCC__
#define __device__
#define __global__
#define __forceinline__
#else
#include <hip/hip_runtime.h>
#endif

#include <stddef.h>
#include <stdint.h>

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

__device__ unsigned int get_strided_index(
    unsigned int idx,
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides
) {
    unsigned int strided_i = 0;
    unsigned int tmp_i = idx;
    for (int d = num_dims - 1; d >= 0; d--) {
        unsigned int i_dim = tmp_i % dims[d];
        strided_i += i_dim * strides[d];
        tmp_i /= dims[d];
    }
    return strided_i;
}

#define CMP_OP(TYPENAME, FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *dims_and_strides, \
    const TYPENAME *lhs, \
    const TYPENAME *rhs, \
    uint8_t *out \
) { \
    const size_t *dims = dims_and_strides; \
    const size_t *lhs_strides = dims_and_strides + num_dims; \
    const size_t *rhs_strides = dims_and_strides + 2 * num_dims; \
    bool lhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, lhs_strides); \
    bool rhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, rhs_strides); \
    if (lhs_cont && rhs_cont) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            TYPENAME x = lhs[i]; \
            TYPENAME y = rhs[i]; \
            out[i] = FUNC ? 1 : 0; \
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
            out[i] = FUNC ? 1 : 0; \
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
            out[i] = FUNC ? 1 : 0; \
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
            out[i] = FUNC ? 1 : 0; \
        } \
    } \
}

CMP_OP(float, eq_f32, x == y)
CMP_OP(double, eq_f64, x == y)
CMP_OP(uint8_t, eq_u8, x == y)
CMP_OP(uint32_t, eq_u32, x == y)
CMP_OP(int64_t, eq_i64, x == y)

CMP_OP(float, ne_f32, x != y)
CMP_OP(double, ne_f64, x != y)
CMP_OP(uint8_t, ne_u8, x != y)
CMP_OP(uint32_t, ne_u32, x != y)
CMP_OP(int64_t, ne_i64, x != y)

CMP_OP(float, lt_f32, x < y)
CMP_OP(double, lt_f64, x < y)
CMP_OP(uint8_t, lt_u8, x < y)
CMP_OP(uint32_t, lt_u32, x < y)
CMP_OP(int64_t, lt_i64, x < y)

CMP_OP(float, le_f32, x <= y)
CMP_OP(double, le_f64, x <= y)
CMP_OP(uint8_t, le_u8, x <= y)
CMP_OP(uint32_t, le_u32, x <= y)
CMP_OP(int64_t, le_i64, x <= y)

CMP_OP(float, gt_f32, x > y)
CMP_OP(double, gt_f64, x > y)
CMP_OP(uint8_t, gt_u8, x > y)
CMP_OP(uint32_t, gt_u32, x > y)
CMP_OP(int64_t, gt_i64, x > y)

CMP_OP(float, ge_f32, x >= y)
CMP_OP(double, ge_f64, x >= y)
CMP_OP(uint8_t, ge_u8, x >= y)
CMP_OP(uint32_t, ge_u32, x >= y)
CMP_OP(int64_t, ge_i64, x >= y)
