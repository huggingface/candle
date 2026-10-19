// Adapted from https://github.com/ggerganov/llama.cpp/blob/master/ggml-cuda/argsort.cu
#define SORT_ORDER_ASC 1
#define SORT_ORDER_DESC 0
#include "cuda_utils.cuh"
#include<stdint.h>

static inline __device__ void topk_insert(
    float v,
    uint32_t i,
    float * vals,
    uint32_t * idx,
    const int k
) {
    if (v <= vals[k - 1]) return;
    int p = k - 1;
    while (p > 0 && v > vals[p - 1]) {
        vals[p] = vals[p - 1];
        idx[p] = idx[p - 1];
        --p;
    }
    vals[p] = v;
    idx[p] = i;
}

extern "C" __global__ void topk_stage1_f32(
    const float * x,
    const uint32_t n,
    const uint32_t k,
    const uint32_t items_per_block,
    float * out_vals,
    uint32_t * out_idx
) {
    const uint32_t start = blockIdx.x * items_per_block;
    const uint32_t end = min(n, start + items_per_block);
    if (start >= end) return;

    float vals[64];
    uint32_t idx[64];
#pragma unroll
    for (int j = 0; j < 64; ++j) {
        vals[j] = -INFINITY;
        idx[j] = 0;
    }

    for (uint32_t i = start + threadIdx.x; i < end; i += blockDim.x) {
        float v = x[i];
        topk_insert(v, i, vals, idx, (int)k);
    }

    extern __shared__ uint8_t smem[];
    float * block_vals = (float *)smem;
    uint32_t * block_idx = (uint32_t *)(block_vals + (uint32_t)blockDim.x * k);

    const uint32_t base = (uint32_t)threadIdx.x * k;
    for (uint32_t j = 0; j < k; ++j) {
        block_vals[base + j] = vals[j];
        block_idx[base + j] = idx[j];
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        float bvals[64];
        uint32_t bidx[64];
#pragma unroll
        for (int j = 0; j < 64; ++j) {
            bvals[j] = -INFINITY;
            bidx[j] = 0;
        }
        for (uint32_t t = 0; t < (uint32_t)blockDim.x; ++t) {
            const uint32_t tb = t * k;
            for (uint32_t j = 0; j < k; ++j) {
                topk_insert(block_vals[tb + j], block_idx[tb + j], bvals, bidx, (int)k);
            }
        }
        const uint32_t out_base = blockIdx.x * k;
        for (uint32_t j = 0; j < k; ++j) {
            out_vals[out_base + j] = bvals[j];
            out_idx[out_base + j] = bidx[j];
        }
    }
}

extern "C" __global__ void topk_stage2_f32(
    const float * in_vals,
    const uint32_t * in_idx,
    const uint32_t m,
    const uint32_t k,
    uint32_t * out_idx
) {
    float vals[64];
    uint32_t idx[64];
    #pragma unroll
    for (int j = 0; j < 64; ++j) {
        vals[j] = -INFINITY;
        idx[j] = 0;
    }

    for (uint32_t i = threadIdx.x; i < m; i += blockDim.x) {
        float v = in_vals[i];
        uint32_t id = in_idx[i];
        topk_insert(v, id, vals, idx, (int)k);
    }

    extern __shared__ uint8_t smem2[];
    float * block_vals = (float *)smem2;
    uint32_t * block_idx = (uint32_t *)(block_vals + (uint32_t)blockDim.x * k);

    const uint32_t base = (uint32_t)threadIdx.x * k;
    for (uint32_t j = 0; j < k; ++j) {
        block_vals[base + j] = vals[j];
        block_idx[base + j] = idx[j];
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        float bvals[64];
        uint32_t bidx[64];
#pragma unroll
        for (int j = 0; j < 64; ++j) {
            bvals[j] = -INFINITY;
            bidx[j] = 0;
        }
        for (uint32_t t = 0; t < (uint32_t)blockDim.x; ++t) {
            const uint32_t tb = t * k;
            for (uint32_t j = 0; j < k; ++j) {
                topk_insert(block_vals[tb + j], block_idx[tb + j], bvals, bidx, (int)k);
            }
        }
        for (uint32_t j = 0; j < k; ++j) {
            out_idx[j] = bidx[j];
        }
    }
}

template<typename T>
static inline __device__ void ggml_cuda_swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template<int order, typename T>
static __device__ void k_argsort(const T * x, uint32_t * dst, const int ncols, int ncols_pad) {
    // bitonic sort
    int row = blockIdx.x;

    const T * x_row = x + row * ncols;
    extern __shared__ int dst_row[];

    // initialize indices - each thread handles multiple elements if ncols_pad > blockDim.x
    for (int col = threadIdx.x; col < ncols_pad; col += blockDim.x) {
        dst_row[col] = col;
    }

    __syncthreads();

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int col = threadIdx.x; col < ncols_pad; col += blockDim.x) {
                int ixj = col ^ j;
                if (ixj > col) {
                    if ((col & k) == 0) {
                        if (dst_row[col] >= ncols ||
                            (dst_row[ixj] < ncols && (order == SORT_ORDER_ASC ?
                                x_row[dst_row[col]] > x_row[dst_row[ixj]] :
                                x_row[dst_row[col]] < x_row[dst_row[ixj]]))
                        ) {
                            ggml_cuda_swap(dst_row[col], dst_row[ixj]);
                        }
                    } else {
                        if (dst_row[ixj] >= ncols ||
                            (dst_row[col] < ncols && (order == SORT_ORDER_ASC ?
                                x_row[dst_row[col]] < x_row[dst_row[ixj]] :
                                x_row[dst_row[col]] > x_row[dst_row[ixj]]))
                        ) {
                            ggml_cuda_swap(dst_row[col], dst_row[ixj]);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    // copy the result to dst without the padding
    for (int col = threadIdx.x; col < ncols; col += blockDim.x) {
        dst[row * ncols + col] = dst_row[col];
    }
}

#define ASORT_OP(TYPENAME, RUST_NAME) \
extern "C" __global__ void asort_asc_##RUST_NAME(  \
    const TYPENAME * x, uint32_t * dst, const int ncols, int ncols_pad \
) { \
    k_argsort<SORT_ORDER_ASC>(x, dst, ncols, ncols_pad); \
} \
extern "C" __global__ void asort_desc_##RUST_NAME(  \
    const TYPENAME * x, uint32_t * dst, const int ncols, int ncols_pad \
) { \
    k_argsort<SORT_ORDER_DESC>(x, dst, ncols, ncols_pad); \
} \
 
#if __CUDA_ARCH__ >= 800
ASORT_OP(__nv_bfloat16, bf16)

// NOTE: No sort ops for f8
// ASORT_OP(__nv_fp8_e4m3, fp8_e4m3)
#endif

#if __CUDA_ARCH__ >= 530
ASORT_OP(__half, f16)
#endif

ASORT_OP(float, f32)
ASORT_OP(double, f64)
ASORT_OP(uint8_t, u8)
ASORT_OP(uint32_t, u32)
ASORT_OP(int64_t, i64)
