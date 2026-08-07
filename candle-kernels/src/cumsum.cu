#include <stdint.h>

// One CUDA block scans one row. Each thread first scans a contiguous chunk,
// then the block scans chunk sums and adds the preceding chunks as an offset.
template<typename T>
static __device__ void k_cumsum_last_dim(
    const T *src,
    T *dst,
    const int ncols,
    const int nthreads
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int chunk = (ncols + nthreads - 1) / nthreads;
    const int start = tid * chunk;
    const int end = min(start + chunk, ncols);
    const int row_offset = row * ncols;

    extern __shared__ char shared[];
    T *sums = reinterpret_cast<T *>(shared);
    T acc = T(0);
    for (int col = start; col < end; ++col) {
        acc += src[row_offset + col];
        dst[row_offset + col] = acc;
    }
    sums[tid] = acc;
    __syncthreads();

    for (int offset = 1; offset < nthreads; offset <<= 1) {
        T value = T(0);
        if (tid >= offset) {
            value = sums[tid - offset];
        }
        __syncthreads();
        sums[tid] += value;
        __syncthreads();
    }

    const T chunk_offset = tid == 0 ? T(0) : sums[tid - 1];
    for (int col = start; col < end; ++col) {
        dst[row_offset + col] += chunk_offset;
    }
}

// Multi-block cumsum uses three kernels. This first pass scans a tile within a
// row and stores one tile total per block; a later pass scans those totals.
template<typename T>
static __device__ void k_cumsum_tile(
    const T *src,
    T *dst,
    T *block_sums,
    const int ncols,
    const int tile_size,
    const int ntiles,
    const int nthreads
) {
    const int row = blockIdx.x;
    const int tile = blockIdx.y;
    const int tid = threadIdx.x;
    const int tile_start = tile * tile_size;
    const int tile_cols = min(tile_size, ncols - tile_start);
    const int chunk = (tile_cols + nthreads - 1) / nthreads;
    const int start = tile_start + tid * chunk;
    const int end = min(start + chunk, tile_start + tile_cols);
    const int row_offset = row * ncols;

    extern __shared__ char shared[];
    T *sums = reinterpret_cast<T *>(shared);
    T acc = T(0);
    for (int col = start; col < end; ++col) {
        acc += src[row_offset + col];
        dst[row_offset + col] = acc;
    }
    sums[tid] = acc;
    __syncthreads();

    for (int offset = 1; offset < nthreads; offset <<= 1) {
        T value = T(0);
        if (tid >= offset) {
            value = sums[tid - offset];
        }
        __syncthreads();
        sums[tid] += value;
        __syncthreads();
    }

    const T chunk_offset = tid == 0 ? T(0) : sums[tid - 1];
    for (int col = start; col < end; ++col) {
        dst[row_offset + col] += chunk_offset;
    }
    if (tid == nthreads - 1) {
        block_sums[row * ntiles + tile] = sums[tid];
    }
}

// Adds the sum of preceding tiles to each element in a tile. The offsets tensor
// is the scanned version of block_sums from k_cumsum_tile.
template<typename T>
static __device__ void k_cumsum_add_offsets(
    T *dst,
    const T *offsets,
    const int ncols,
    const int tile_size,
    const int ntiles
) {
    const int row = blockIdx.x;
    const int tile = blockIdx.y;
    if (tile == 0) {
        return;
    }

    const int tid = threadIdx.x;
    const int tile_start = tile * tile_size;
    const int tile_end = min(tile_start + tile_size, ncols);
    const int row_offset = row * ncols;
    const T offset = offsets[row * ntiles + tile - 1];
    for (int col = tile_start + tid; col < tile_end; col += blockDim.x) {
        dst[row_offset + col] += offset;
    }
}

#define CUMSUM_OP(TYPENAME, RUST_NAME) \
extern "C" __global__ void cumsum_last_dim_##RUST_NAME( \
    const TYPENAME *src, TYPENAME *dst, const int ncols, const int nthreads \
) { \
    k_cumsum_last_dim<TYPENAME>(src, dst, ncols, nthreads); \
} \
extern "C" __global__ void cumsum_tile_##RUST_NAME( \
    const TYPENAME *src, TYPENAME *dst, TYPENAME *block_sums, \
    const int ncols, const int tile_size, const int ntiles, const int nthreads \
) { \
    k_cumsum_tile<TYPENAME>(src, dst, block_sums, ncols, tile_size, ntiles, nthreads); \
} \
extern "C" __global__ void cumsum_add_offsets_##RUST_NAME( \
    TYPENAME *dst, const TYPENAME *offsets, \
    const int ncols, const int tile_size, const int ntiles \
) { \
    k_cumsum_add_offsets<TYPENAME>(dst, offsets, ncols, tile_size, ntiles); \
}

CUMSUM_OP(float, f32)
CUMSUM_OP(double, f64)
CUMSUM_OP(uint32_t, u32)
CUMSUM_OP(int64_t, i64)
