// Translated from cuda_utils.cuh by TEAM-488 (Phase 2)
// HIP utility functions for kernel operations

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <stdint.h>
#include <cmath>

// Check if tensor layout is contiguous
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

// Convert flat index to strided index
__device__ unsigned int get_strided_index(
    unsigned int idx,
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides
) {
    unsigned int strided_i = 0;
    for (unsigned int d = 0; d < num_dims; d++) {
        unsigned int dim_idx = num_dims - 1 - d;
        strided_i += (idx % dims[dim_idx]) * strides[dim_idx];
        idx /= dims[dim_idx];
    }
    return strided_i;
}

// Restride an index from one layout to another
__device__ unsigned int restrided(
    const unsigned int strided_i,
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides,
    const size_t *new_strides
) {
    unsigned int idx = 0;
    for (int d = 0; d < num_dims; d++) {
        idx += (strides[d] == 0 ? 0 : (strided_i / strides[d]) % dims[d]) * new_strides[d];
    }
    return idx;
}

// Round up to next power of 2 (for reductions)
__device__ __forceinline__ unsigned int next_power_of_two(unsigned int v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v++;
    return v;
}

// Chunk sum for reductions
template<typename T>
__device__ void chunk_sum(
    const size_t chunk_len,
    const T data,
    T* out
) {
    __shared__ T buf[1024];

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int block_i = threadIdx.x;

    buf[block_i] = data;
    unsigned int chunk_i = i / chunk_len;
    unsigned int chunk_start = chunk_len * chunk_i;
    unsigned int chunk_end = chunk_len * (chunk_i + 1);

    chunk_end = (numel < chunk_end) ? numel : chunk_end;
    size_t chunk_size = chunk_end - chunk_start;
    chunk_size = (chunk_size < blockDim.x) ? chunk_size : blockDim.x;
    chunk_size = next_power_of_two(chunk_size);

    __syncthreads();

    for (unsigned int s = chunk_size / 2; s > 0; s >>= 1) {
        if (block_i < s && block_i + s < chunk_size) {
            buf[block_i] += buf[block_i + s];
        }
        __syncthreads();
    }

    if (block_i == 0) {
        out[chunk_i] = buf[0];
    }
}
