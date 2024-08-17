// Adapted from https://github.com/ggerganov/llama.cpp/blob/master/ggml-cuda/argsort.cu
#define SORT_ORDER_ASC 1
#define SORT_ORDER_DESC 0
#include "cuda_utils.cuh"
#include<stdint.h>

template<typename T>
static inline __device__ void ggml_cuda_swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template<int order, typename T>
static __device__ void k_argsort(const T * x, uint32_t * dst, const int ncols, int ncols_pad) {
    // bitonic sort
    int col = threadIdx.x;
    int row = blockIdx.y;

    if (col >= ncols_pad) {
        return;
    }

    const T * x_row = x + row * ncols;
    extern __shared__ int dst_row[];

    // initialize indices
    dst_row[col] = col;

    __syncthreads();

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
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
            __syncthreads();
        }
    }

    // copy the result to dst without the padding
    if (col < ncols) {
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
#endif

#if __CUDA_ARCH__ >= 530
ASORT_OP(__half, f16)
#endif

ASORT_OP(float, f32)
ASORT_OP(double, f64)
ASORT_OP(uint8_t, u8)
ASORT_OP(uint32_t, u32)
ASORT_OP(int64_t, i64)

template <bool Ascending, typename T>
__device__ bool compare(const T& a, const T& b) {
    return Ascending ? (a > b) : (a < b);
}

template <bool Ascending, typename T>
__device__ void argsort_no_smem(const T* data, uint32_t * indices, const int n, const int m) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n) {
        // Each thread sorts one row
        for (int i = 0; i < m; i++) {
            indices[row * m + i] = i;
        }

        // Simple bubble sort
        for (int i = 0; i < m - 1; i++) {
            for (int j = 0; j < m - 1 - i; j++) {
                int idx1 = row * m + j;
                int idx2 = row * m + j + 1;
                
                if (compare<Ascending>(data[row * m + indices[idx1]], data[row * m + indices[idx2]])) {
                    // Swap indices
                    int temp = indices[idx1];
                    indices[idx1] = indices[idx2];
                    indices[idx2] = temp;
                }
            }
        }
    }
}

#define ASORT_NOSMEM_OP(TYPENAME, RUST_NAME) \
extern "C" __global__ void asort_asc_no_smem_##RUST_NAME(  \
    const TYPENAME* data, uint32_t * indices, const int n, const int m \
) { \
    argsort_no_smem<true>(data, indices, n, m); \
} \
extern "C" __global__ void asort_desc_no_smem_##RUST_NAME(  \
    const TYPENAME* data, uint32_t * indices, const int n, const int m \
) { \
    argsort_no_smem<false>(data, indices, n, m); \
} \
 
#if __CUDA_ARCH__ >= 800
ASORT_NOSMEM_OP(__nv_bfloat16, bf16)
#endif

#if __CUDA_ARCH__ >= 530
ASORT_NOSMEM_OP(__half, f16)
#endif

ASORT_NOSMEM_OP(float, f32)
ASORT_NOSMEM_OP(double, f64)
ASORT_NOSMEM_OP(uint8_t, u8)
ASORT_NOSMEM_OP(uint32_t, u32)
ASORT_NOSMEM_OP(int64_t, i64)
