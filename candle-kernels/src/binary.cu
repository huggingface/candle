#include "binary_op_macros.cuh"
#include<stdint.h>

#if __CUDA_ARCH__ >= 800
// Vectorized bf16 binary ops: 8 elements per float4 load in contiguous path
extern "C" __global__ void badd_bf16(
    const size_t numel, const size_t num_dims, const size_t *dims_and_strides,
    const __nv_bfloat16 *lhs, const __nv_bfloat16 *rhs, __nv_bfloat16 *out) {
    const size_t *dims = dims_and_strides;
    const size_t *lhs_strides = dims_and_strides + 1 * num_dims;
    const size_t *rhs_strides = dims_and_strides + 2 * num_dims;
    bool lhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, lhs_strides);
    bool rhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, rhs_strides);
    if (lhs_cont && rhs_cont) {
        const size_t vec_numel = numel / 8;
        const float4 *lhs4 = reinterpret_cast<const float4*>(lhs);
        const float4 *rhs4 = reinterpret_cast<const float4*>(rhs);
        float4 *out4 = reinterpret_cast<float4*>(out);
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_numel; i += blockDim.x * gridDim.x) {
            float4 a = lhs4[i];
            float4 b = rhs4[i];
            __nv_bfloat162 *a2 = reinterpret_cast<__nv_bfloat162*>(&a);
            __nv_bfloat162 *b2 = reinterpret_cast<__nv_bfloat162*>(&b);
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                a2[j] = __hadd2(a2[j], b2[j]);
            }
            out4[i] = a;
        }
        const size_t tail_start = vec_numel * 8;
        for (unsigned int i = tail_start + blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            out[i] = lhs[i] + rhs[i];
        }
    } else if (lhs_cont) {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned int tmp_i = i; unsigned int rhs_i = 0;
            for (int d = num_dims - 1; d >= 0; d--) { unsigned int i_dim = tmp_i % dims[d]; rhs_i += i_dim * rhs_strides[d]; tmp_i /= dims[d]; }
            out[i] = lhs[i] + rhs[rhs_i];
        }
    } else if (rhs_cont) {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned int tmp_i = i; unsigned int lhs_i = 0;
            for (int d = num_dims - 1; d >= 0; d--) { unsigned int i_dim = tmp_i % dims[d]; lhs_i += i_dim * lhs_strides[d]; tmp_i /= dims[d]; }
            out[i] = lhs[lhs_i] + rhs[i];
        }
    } else {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned int tmp_i = i; unsigned int lhs_i = 0, rhs_i = 0;
            for (int d = num_dims - 1; d >= 0; d--) { unsigned int i_dim = tmp_i % dims[d]; lhs_i += i_dim * lhs_strides[d]; rhs_i += i_dim * rhs_strides[d]; tmp_i /= dims[d]; }
            out[i] = lhs[lhs_i] + rhs[rhs_i];
        }
    }
}

extern "C" __global__ void bmul_bf16(
    const size_t numel, const size_t num_dims, const size_t *dims_and_strides,
    const __nv_bfloat16 *lhs, const __nv_bfloat16 *rhs, __nv_bfloat16 *out) {
    const size_t *dims = dims_and_strides;
    const size_t *lhs_strides = dims_and_strides + 1 * num_dims;
    const size_t *rhs_strides = dims_and_strides + 2 * num_dims;
    bool lhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, lhs_strides);
    bool rhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, rhs_strides);
    if (lhs_cont && rhs_cont) {
        const size_t vec_numel = numel / 8;
        const float4 *lhs4 = reinterpret_cast<const float4*>(lhs);
        const float4 *rhs4 = reinterpret_cast<const float4*>(rhs);
        float4 *out4 = reinterpret_cast<float4*>(out);
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_numel; i += blockDim.x * gridDim.x) {
            float4 a = lhs4[i];
            float4 b = rhs4[i];
            __nv_bfloat162 *a2 = reinterpret_cast<__nv_bfloat162*>(&a);
            __nv_bfloat162 *b2 = reinterpret_cast<__nv_bfloat162*>(&b);
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                a2[j] = __hmul2(a2[j], b2[j]);
            }
            out4[i] = a;
        }
        const size_t tail_start = vec_numel * 8;
        for (unsigned int i = tail_start + blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            out[i] = lhs[i] * rhs[i];
        }
    } else if (lhs_cont) {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned int tmp_i = i; unsigned int rhs_i = 0;
            for (int d = num_dims - 1; d >= 0; d--) { unsigned int i_dim = tmp_i % dims[d]; rhs_i += i_dim * rhs_strides[d]; tmp_i /= dims[d]; }
            out[i] = lhs[i] * rhs[rhs_i];
        }
    } else if (rhs_cont) {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned int tmp_i = i; unsigned int lhs_i = 0;
            for (int d = num_dims - 1; d >= 0; d--) { unsigned int i_dim = tmp_i % dims[d]; lhs_i += i_dim * lhs_strides[d]; tmp_i /= dims[d]; }
            out[i] = lhs[lhs_i] * rhs[i];
        }
    } else {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned int tmp_i = i; unsigned int lhs_i = 0, rhs_i = 0;
            for (int d = num_dims - 1; d >= 0; d--) { unsigned int i_dim = tmp_i % dims[d]; lhs_i += i_dim * lhs_strides[d]; rhs_i += i_dim * rhs_strides[d]; tmp_i /= dims[d]; }
            out[i] = lhs[lhs_i] * rhs[rhs_i];
        }
    }
}

BINARY_OP(__nv_bfloat16, bdiv_bf16, x / y)
BINARY_OP(__nv_bfloat16, bsub_bf16, x - y)
BINARY_OP(__nv_bfloat16, bmaximum_bf16, maxg(x, y))
BINARY_OP(__nv_bfloat16, bminimum_bf16, ming(x, y))
BINARY_OP_OUT(__nv_bfloat16, uint8_t, eq_bf16, x == y)
BINARY_OP_OUT(__nv_bfloat16, uint8_t, ne_bf16, x != y)
BINARY_OP_OUT(__nv_bfloat16, uint8_t, lt_bf16, x < y)
BINARY_OP_OUT(__nv_bfloat16, uint8_t, le_bf16, x <= y)
BINARY_OP_OUT(__nv_bfloat16, uint8_t, gt_bf16, x > y)
BINARY_OP_OUT(__nv_bfloat16, uint8_t, ge_bf16, x >= y)

#define F8E4M3_TO_FLOAT(x) __half2float(__nv_cvt_fp8_to_halfraw(x.__x, __NV_E4M3))

BINARY_OP(__nv_fp8_e4m3, badd_f8_e4m3, __nv_fp8_e4m3(F8E4M3_TO_FLOAT(x) + F8E4M3_TO_FLOAT(y)))
BINARY_OP(__nv_fp8_e4m3, bdiv_f8_e4m3, __nv_fp8_e4m3(F8E4M3_TO_FLOAT(x) / F8E4M3_TO_FLOAT(y)))
BINARY_OP(__nv_fp8_e4m3, bmul_f8_e4m3, __nv_fp8_e4m3(F8E4M3_TO_FLOAT(x) * F8E4M3_TO_FLOAT(y)))
BINARY_OP(__nv_fp8_e4m3, bsub_f8_e4m3, __nv_fp8_e4m3(F8E4M3_TO_FLOAT(x) - F8E4M3_TO_FLOAT(y)))
BINARY_OP(__nv_fp8_e4m3, bmaximum_f8_e4m3, maxg(x, y))
BINARY_OP(__nv_fp8_e4m3, bminimum_f8_e4m3, ming(x, y))
BINARY_OP_OUT(__nv_fp8_e4m3, uint8_t, eq_f8_e4m3, F8E4M3_TO_FLOAT(x) == F8E4M3_TO_FLOAT(y))
BINARY_OP_OUT(__nv_fp8_e4m3, uint8_t, ne_f8_e4m3, F8E4M3_TO_FLOAT(x) != F8E4M3_TO_FLOAT(y))
BINARY_OP_OUT(__nv_fp8_e4m3, uint8_t, lt_f8_e4m3, F8E4M3_TO_FLOAT(x) < F8E4M3_TO_FLOAT(y))
BINARY_OP_OUT(__nv_fp8_e4m3, uint8_t, le_f8_e4m3, F8E4M3_TO_FLOAT(x) <= F8E4M3_TO_FLOAT(y))
BINARY_OP_OUT(__nv_fp8_e4m3, uint8_t, gt_f8_e4m3, F8E4M3_TO_FLOAT(x) > F8E4M3_TO_FLOAT(y))
BINARY_OP_OUT(__nv_fp8_e4m3, uint8_t, ge_f8_e4m3, F8E4M3_TO_FLOAT(x) >= F8E4M3_TO_FLOAT(y))
#endif

#if __CUDA_ARCH__ >= 530
BINARY_OP(__half, badd_f16, x + y)
BINARY_OP(__half, bdiv_f16, x / y)
BINARY_OP(__half, bmul_f16, x * y)
BINARY_OP(__half, bsub_f16, x - y)
BINARY_OP(__half, bmaximum_f16, maxg(x, y))
BINARY_OP(__half, bminimum_f16, ming(x, y))
BINARY_OP_OUT(__half, uint8_t, eq_f16, x == y)
BINARY_OP_OUT(__half, uint8_t, ne_f16, x != y)
BINARY_OP_OUT(__half, uint8_t, lt_f16, x < y)
BINARY_OP_OUT(__half, uint8_t, le_f16, x <= y)
BINARY_OP_OUT(__half, uint8_t, gt_f16, x > y)
BINARY_OP_OUT(__half, uint8_t, ge_f16, x >= y)
#endif

// Vectorized f32 binary ops: 4 elements per float4 load
extern "C" __global__ void badd_f32(
    const size_t numel, const size_t num_dims, const size_t *dims_and_strides,
    const float *lhs, const float *rhs, float *out) {
    const size_t *dims = dims_and_strides;
    const size_t *lhs_strides = dims_and_strides + 1 * num_dims;
    const size_t *rhs_strides = dims_and_strides + 2 * num_dims;
    bool lhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, lhs_strides);
    bool rhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, rhs_strides);
    if (lhs_cont && rhs_cont) {
        const size_t vec_numel = numel / 4;
        const float4 *lhs4 = reinterpret_cast<const float4*>(lhs);
        const float4 *rhs4 = reinterpret_cast<const float4*>(rhs);
        float4 *out4 = reinterpret_cast<float4*>(out);
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_numel; i += blockDim.x * gridDim.x) {
            float4 a = lhs4[i], b = rhs4[i];
            a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
            out4[i] = a;
        }
        const size_t tail_start = vec_numel * 4;
        for (unsigned int i = tail_start + blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x)
            out[i] = lhs[i] + rhs[i];
    } else if (lhs_cont) {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned int tmp_i = i; unsigned int rhs_i = 0;
            for (int d = num_dims - 1; d >= 0; d--) { unsigned int i_dim = tmp_i % dims[d]; rhs_i += i_dim * rhs_strides[d]; tmp_i /= dims[d]; }
            out[i] = lhs[i] + rhs[rhs_i];
        }
    } else if (rhs_cont) {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned int tmp_i = i; unsigned int lhs_i = 0;
            for (int d = num_dims - 1; d >= 0; d--) { unsigned int i_dim = tmp_i % dims[d]; lhs_i += i_dim * lhs_strides[d]; tmp_i /= dims[d]; }
            out[i] = lhs[lhs_i] + rhs[i];
        }
    } else {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned int tmp_i = i; unsigned int lhs_i = 0, rhs_i = 0;
            for (int d = num_dims - 1; d >= 0; d--) { unsigned int i_dim = tmp_i % dims[d]; lhs_i += i_dim * lhs_strides[d]; rhs_i += i_dim * rhs_strides[d]; tmp_i /= dims[d]; }
            out[i] = lhs[lhs_i] + rhs[rhs_i];
        }
    }
}

extern "C" __global__ void bdiv_f32(
    const size_t numel, const size_t num_dims, const size_t *dims_and_strides,
    const float *lhs, const float *rhs, float *out) {
    const size_t *dims = dims_and_strides;
    const size_t *lhs_strides = dims_and_strides + 1 * num_dims;
    const size_t *rhs_strides = dims_and_strides + 2 * num_dims;
    bool lhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, lhs_strides);
    bool rhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, rhs_strides);
    if (lhs_cont && rhs_cont) {
        const size_t vec_numel = numel / 4;
        const float4 *lhs4 = reinterpret_cast<const float4*>(lhs);
        const float4 *rhs4 = reinterpret_cast<const float4*>(rhs);
        float4 *out4 = reinterpret_cast<float4*>(out);
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_numel; i += blockDim.x * gridDim.x) {
            float4 a = lhs4[i], b = rhs4[i];
            a.x /= b.x; a.y /= b.y; a.z /= b.z; a.w /= b.w;
            out4[i] = a;
        }
        const size_t tail_start = vec_numel * 4;
        for (unsigned int i = tail_start + blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x)
            out[i] = lhs[i] / rhs[i];
    } else if (lhs_cont) {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned int tmp_i = i; unsigned int rhs_i = 0;
            for (int d = num_dims - 1; d >= 0; d--) { unsigned int i_dim = tmp_i % dims[d]; rhs_i += i_dim * rhs_strides[d]; tmp_i /= dims[d]; }
            out[i] = lhs[i] / rhs[rhs_i];
        }
    } else if (rhs_cont) {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned int tmp_i = i; unsigned int lhs_i = 0;
            for (int d = num_dims - 1; d >= 0; d--) { unsigned int i_dim = tmp_i % dims[d]; lhs_i += i_dim * lhs_strides[d]; tmp_i /= dims[d]; }
            out[i] = lhs[lhs_i] / rhs[i];
        }
    } else {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned int tmp_i = i; unsigned int lhs_i = 0, rhs_i = 0;
            for (int d = num_dims - 1; d >= 0; d--) { unsigned int i_dim = tmp_i % dims[d]; lhs_i += i_dim * lhs_strides[d]; rhs_i += i_dim * rhs_strides[d]; tmp_i /= dims[d]; }
            out[i] = lhs[lhs_i] / rhs[rhs_i];
        }
    }
}

extern "C" __global__ void bmul_f32(
    const size_t numel, const size_t num_dims, const size_t *dims_and_strides,
    const float *lhs, const float *rhs, float *out) {
    const size_t *dims = dims_and_strides;
    const size_t *lhs_strides = dims_and_strides + 1 * num_dims;
    const size_t *rhs_strides = dims_and_strides + 2 * num_dims;
    bool lhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, lhs_strides);
    bool rhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, rhs_strides);
    if (lhs_cont && rhs_cont) {
        const size_t vec_numel = numel / 4;
        const float4 *lhs4 = reinterpret_cast<const float4*>(lhs);
        const float4 *rhs4 = reinterpret_cast<const float4*>(rhs);
        float4 *out4 = reinterpret_cast<float4*>(out);
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_numel; i += blockDim.x * gridDim.x) {
            float4 a = lhs4[i], b = rhs4[i];
            a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w;
            out4[i] = a;
        }
        const size_t tail_start = vec_numel * 4;
        for (unsigned int i = tail_start + blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x)
            out[i] = lhs[i] * rhs[i];
    } else if (lhs_cont) {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned int tmp_i = i; unsigned int rhs_i = 0;
            for (int d = num_dims - 1; d >= 0; d--) { unsigned int i_dim = tmp_i % dims[d]; rhs_i += i_dim * rhs_strides[d]; tmp_i /= dims[d]; }
            out[i] = lhs[i] * rhs[rhs_i];
        }
    } else if (rhs_cont) {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned int tmp_i = i; unsigned int lhs_i = 0;
            for (int d = num_dims - 1; d >= 0; d--) { unsigned int i_dim = tmp_i % dims[d]; lhs_i += i_dim * lhs_strides[d]; tmp_i /= dims[d]; }
            out[i] = lhs[lhs_i] * rhs[i];
        }
    } else {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned int tmp_i = i; unsigned int lhs_i = 0, rhs_i = 0;
            for (int d = num_dims - 1; d >= 0; d--) { unsigned int i_dim = tmp_i % dims[d]; lhs_i += i_dim * lhs_strides[d]; rhs_i += i_dim * rhs_strides[d]; tmp_i /= dims[d]; }
            out[i] = lhs[lhs_i] * rhs[rhs_i];
        }
    }
}

BINARY_OP(double, badd_f64, x + y);
BINARY_OP(uint8_t, badd_u8, x + y);
BINARY_OP(uint32_t, badd_u32, x + y);
BINARY_OP(int64_t, badd_i64, x + y);
BINARY_OP(double, bdiv_f64, x / y);
BINARY_OP(uint8_t, bdiv_u8, x / y);
BINARY_OP(uint32_t, bdiv_u32, x / y);
BINARY_OP(int64_t, bdiv_i64, x / y);
BINARY_OP(double, bmul_f64, x * y);
BINARY_OP(uint8_t, bmul_u8, x * y);
BINARY_OP(uint32_t, bmul_u32, x * y);
BINARY_OP(int64_t, bmul_i64, x * y);
BINARY_OP(float, bsub_f32, x - y)
BINARY_OP(double, bsub_f64, x - y);
BINARY_OP(uint8_t, bsub_u8, x - y);
BINARY_OP(uint32_t, bsub_u32, x - y);
BINARY_OP(int64_t, bsub_i64, x - y);
BINARY_OP(float, bminimum_f32, ming(x, y));
BINARY_OP(double, bminimum_f64, ming(x, y));
BINARY_OP(uint8_t, bminimum_u8, ming(x, y));
BINARY_OP(uint32_t, bminimum_u32, ming(x, y));
BINARY_OP(int64_t, bminimum_i64, ming(x, y));
BINARY_OP(float, bmaximum_f32, maxg(x, y));
BINARY_OP(double, bmaximum_f64, maxg(x, y));
BINARY_OP(uint8_t, bmaximum_u8, maxg(x, y));
BINARY_OP(uint32_t, bmaximum_u32, maxg(x, y));
BINARY_OP(int64_t, bmaximum_i64, maxg(x, y));

BINARY_OP_OUT(float, uint8_t, eq_f32, x == y)
BINARY_OP_OUT(double, uint8_t, eq_f64, x == y)
BINARY_OP_OUT(uint8_t, uint8_t, eq_u8, x == y)
BINARY_OP_OUT(uint32_t, uint8_t, eq_u32, x == y)
BINARY_OP_OUT(int64_t, uint8_t, eq_i64, x == y)

BINARY_OP_OUT(float, uint8_t, ne_f32, x != y)
BINARY_OP_OUT(double, uint8_t, ne_f64, x != y)
BINARY_OP_OUT(uint8_t, uint8_t, ne_u8, x != y)
BINARY_OP_OUT(uint32_t, uint8_t, ne_u32, x != y)
BINARY_OP_OUT(int64_t, uint8_t, ne_i64, x != y)

BINARY_OP_OUT(float, uint8_t, lt_f32, x < y)
BINARY_OP_OUT(double, uint8_t, lt_f64, x < y)
BINARY_OP_OUT(uint8_t, uint8_t, lt_u8, x < y)
BINARY_OP_OUT(uint32_t, uint8_t, lt_u32, x < y)
BINARY_OP_OUT(int64_t, uint8_t, lt_i64, x < y)

BINARY_OP_OUT(float, uint8_t, le_f32, x <= y)
BINARY_OP_OUT(double, uint8_t, le_f64, x <= y)
BINARY_OP_OUT(uint8_t, uint8_t, le_u8, x <= y)
BINARY_OP_OUT(uint32_t, uint8_t, le_u32, x <= y)
BINARY_OP_OUT(int64_t, uint8_t, le_i64, x <= y)

BINARY_OP_OUT(float, uint8_t, gt_f32, x > y)
BINARY_OP_OUT(double, uint8_t, gt_f64, x > y)
BINARY_OP_OUT(uint8_t, uint8_t, gt_u8, x > y)
BINARY_OP_OUT(uint32_t, uint8_t, gt_u32, x > y)
BINARY_OP_OUT(int64_t, uint8_t, gt_i64, x > y)

BINARY_OP_OUT(float, uint8_t, ge_f32, x >= y)
BINARY_OP_OUT(double, uint8_t, ge_f64, x >= y)
BINARY_OP_OUT(uint8_t, uint8_t, ge_u8, x >= y)
BINARY_OP_OUT(uint32_t, uint8_t, ge_u32, x >= y)
BINARY_OP_OUT(int64_t, uint8_t, ge_i64, x >= y)
