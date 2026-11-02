#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/warp/warp_scan.cuh>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stddef.h>
#include <stdint.h>

// One CUDA block scans one row. Each thread first scans a contiguous chunk,
// then the block scans chunk sums and adds the preceding chunks as an offset.
template<typename T>
static __device__ void k_cumsum_last_dim(
    const T *src,
    T *dst,
    const size_t ncols,
    const int nthreads
) {
    const size_t row = blockIdx.x;
    const int tid = threadIdx.x;
    const size_t chunk = (ncols - 1) / nthreads + 1;
    const size_t start = tid * chunk;
    const size_t end = start < ncols ? start + min(chunk, ncols - start) : ncols;
    const size_t row_offset = row * ncols;

    extern __shared__ char shared[];
    T *sums = reinterpret_cast<T *>(shared);
    T acc = T(0);
    for (size_t col = start; col < end; ++col) {
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
    for (size_t col = start; col < end; ++col) {
        dst[row_offset + col] += chunk_offset;
    }
}

// Specialize the normal launch separately so its register allocation does
// not include loop state. Only the grid-stride entry reuses CUB shared storage.
template<bool GridStride>
static __device__ __forceinline__ size_t cumsum_next_task(size_t task, size_t ntasks) {
    if constexpr (!GridStride) return ntasks;
    if (ntasks - task <= size_t(gridDim.x)) return ntasks;
    __syncthreads();
    return task + size_t(gridDim.x);
}

template<bool GridStride>
static __device__ __forceinline__ size_t cumsum_task_quotient(size_t task, size_t divisor, uint32_t reciprocal) {
    if constexpr (GridStride) {
        return task / divisor;
    } else {
        // The Rust launcher uses this entry only when all tasks fit in grid.x,
        // hence both task and divisor fit in 31 bits. Addresses remain size_t.
        const uint32_t d = uint32_t(divisor);
        if (d == 1) return task;
        // ceil(2^32 / divisor) overestimates this 31-bit quotient by at most
        // one. The product fits u32 because task and divisor are <= INT_MAX.
        const uint32_t quotient = __umulhi(uint32_t(task), reciprocal);
        return quotient - uint32_t(quotient * d > uint32_t(task));
    }
}

// Single-row entries below compile row=0 and tile=blockIdx.x into the same
// helpers. Generic and grid-stride entries keep their existing task mapping.
// First of three passes. The Rust launcher fixes 256 threads and 4096 elements
// per tile. CUB keeps local prefixes in registers and coalesces global I/O.
template<typename T, typename Input = T>
static __device__ void k_cumsum_tile(
    const Input *src,
    T *dst,
    T *block_sums,
    const size_t ncols,
    const size_t tile_size,
    const size_t ntiles,
    const size_t row,
    const size_t tile
) {
    constexpr int threads = 256;
    constexpr int items_per_thread = 16;
    using Load = cub::BlockLoad<T, threads, items_per_thread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using Scan = cub::BlockScan<T, threads, cub::BLOCK_SCAN_WARP_SCANS>;
    using TailScan = cub::WarpScan<T, 32>;
    using Store = cub::BlockStore<T, threads, items_per_thread, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    __shared__ union {
        typename Load::TempStorage load;
        typename Scan::TempStorage scan;
        typename Store::TempStorage store;
        typename TailScan::TempStorage tail;
    } storage;
    const size_t tile_start = tile * tile_size;
    const size_t row_offset = row * ncols;
    const int valid = static_cast<int>(min(tile_size, ncols - tile_start));
    // Uniform across the block; finish before entering any block collective.
    if (valid == 1) {
        if (threadIdx.x == 0) {
            const T value = src[row_offset + tile_start];
            dst[row_offset + tile_start] = value;
            block_sums[row * ntiles + tile] = value;
        }
        return;
    }
    if (valid <= 32) {
        if (threadIdx.x < 32) {
            const int lane = threadIdx.x;
            const T value = lane < valid ? T(src[row_offset + tile_start + lane]) : T(0);
            T prefix;
            TailScan(storage.tail).InclusiveSum(value, prefix);
            if (lane < valid) dst[row_offset + tile_start + lane] = prefix;
            if (lane == valid - 1) block_sums[row * ntiles + tile] = prefix;
        }
        return;
    }
    T items[items_per_thread];
    Load(storage.load).Load(src + row_offset + tile_start, items, valid, T(0));
    // The union cannot be reused until every warp has finished with it.
    __syncthreads();
    T tile_sum;
    Scan(storage.scan).InclusiveSum(items, items, tile_sum);
    __syncthreads();
    Store(storage.store).Store(dst + row_offset + tile_start, items, valid);
    if (threadIdx.x == 0) {
        block_sums[row * ntiles + tile] = tile_sum;
    }
}

// Adds the sum of preceding tiles to each element in a tile. The offsets tensor
// is the scanned version of block_sums from k_cumsum_tile.
template<typename T>
static __device__ void k_cumsum_add_offsets(
    T *dst,
    const T *offsets,
    const size_t ncols,
    const size_t tile_size,
    const size_t ntiles,
    const size_t row,
    const size_t tile
) {
    if (tile == 0) {
        return;
    }

    const int tid = threadIdx.x;
    const size_t tile_start = tile * tile_size;
    const size_t tile_end = min(tile_start + tile_size, ncols);
    const size_t row_offset = row * ncols;
    const T offset = offsets[row * ntiles + tile - 1];
    for (size_t col = tile_start + tid; col < tile_end; col += blockDim.x) {
        dst[row_offset + col] += offset;
    }
}

#define CUMSUM_TILE_OPS(TYPENAME, RUST_NAME, SUFFIX, GRID_STRIDE, SINGLE_ROW, TWO_TILES) \
extern "C" __global__ void cumsum_tile##SUFFIX##_##RUST_NAME( \
    const TYPENAME *src, TYPENAME *dst, TYPENAME *block_sums, \
    const size_t ncols, const size_t tile_size, const size_t ntiles, const int nthreads, const size_t ntasks, const uint32_t nrows, const uint32_t row_reciprocal \
) { \
    size_t task = blockIdx.x; \
    if (GRID_STRIDE && task >= ntasks) return; \
    do { \
        const size_t tile = TWO_TILES ? (task & 1) : (SINGLE_ROW ? task : cumsum_task_quotient<GRID_STRIDE>(task, nrows, row_reciprocal)); \
        const size_t row = TWO_TILES ? (task >> 1) : (SINGLE_ROW ? 0 : (GRID_STRIDE ? task - tile * nrows : uint32_t(task) - uint32_t(tile) * nrows)); \
        k_cumsum_tile<TYPENAME>(src, dst, block_sums, ncols, tile_size, ntiles, row, tile); \
        if constexpr (!GRID_STRIDE) break; \
        task = cumsum_next_task<GRID_STRIDE>(task, ntasks); \
    } while (task < ntasks); \
} \
extern "C" __global__ void cumsum_add_offsets##SUFFIX##_##RUST_NAME( \
    TYPENAME *dst, const TYPENAME *offsets, \
    const size_t ncols, const size_t tile_size, const size_t ntiles, const size_t ntasks, const uint32_t nrows, const uint32_t row_reciprocal \
) { \
    size_t task = blockIdx.x; \
    if (GRID_STRIDE && task >= ntasks) return; \
    do { \
        const size_t tile = TWO_TILES ? (task & 1) : (SINGLE_ROW ? task : cumsum_task_quotient<GRID_STRIDE>(task, nrows, row_reciprocal)); \
        const size_t row = TWO_TILES ? (task >> 1) : (SINGLE_ROW ? 0 : (GRID_STRIDE ? task - tile * nrows : uint32_t(task) - uint32_t(tile) * nrows)); \
        k_cumsum_add_offsets<TYPENAME>(dst, offsets, ncols, tile_size, ntiles, row, tile); \
        if constexpr (!GRID_STRIDE) break; \
        task = cumsum_next_task<GRID_STRIDE>(task, ntasks); \
    } while (task < ntasks); \
}

#define CUMSUM_OP(TYPENAME, RUST_NAME) \
extern "C" __global__ void cumsum_last_dim_##RUST_NAME( \
    const TYPENAME *src, TYPENAME *dst, const size_t ncols, const int nthreads \
) { \
    k_cumsum_last_dim<TYPENAME>(src, dst, ncols, nthreads); \
} \
CUMSUM_TILE_OPS(TYPENAME, RUST_NAME, , false, false, false) \
CUMSUM_TILE_OPS(TYPENAME, RUST_NAME, _single_row, false, true, false) \
CUMSUM_TILE_OPS(TYPENAME, RUST_NAME, _two_tiles, false, false, true) \
CUMSUM_TILE_OPS(TYPENAME, RUST_NAME, _grid_stride, true, false, false)

CUMSUM_OP(float, f32)
CUMSUM_OP(double, f64)
CUMSUM_OP(uint32_t, u32)
// The i64 ABI keeps its original bits. Unsigned arithmetic defines modulo-2^64
// addition in every pass (including CUB), without C++ signed-overflow UB.
CUMSUM_OP(uint64_t, i64)

// Low-precision values are rounded only when writing the final output.
template<typename T> static __device__ T cumsum_from_float(float value);
template<> __device__ __half cumsum_from_float(float value) {
    return __float2half_rn(value);
}
template<> __device__ __nv_bfloat16 cumsum_from_float(float value) {
    return __float2bfloat16_rn(value);
}

template<typename T>
static __device__ void k_cumsum_last_dim_lowp(
    const T *src, T *dst, const size_t ncols, const int nthreads
) {
    const size_t row_offset = size_t(blockIdx.x) * ncols;
    const int tid = threadIdx.x;
    const size_t chunk = (ncols - 1) / nthreads + 1;
    const size_t start = tid * chunk;
    const size_t end = start < ncols ? start + min(chunk, ncols - start) : ncols;
    extern __shared__ char shared[];
    float *sums = reinterpret_cast<float *>(shared);
    float prefixes[4];
    float acc = 0.f;
    // Constant array indices let the compiler keep short-row prefixes in
    // registers. Larger chunks reread input.
    if (chunk <= 4) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (start + i < end) {
                acc += float(src[row_offset + start + i]);
                prefixes[i] = acc;
            }
        }
    } else {
        for (size_t col = start; col < end; ++col) acc += float(src[row_offset + col]);
    }
    sums[tid] = acc;
    __syncthreads();
    for (int offset = 1; offset < nthreads; offset <<= 1) {
        const float value = tid >= offset ? sums[tid - offset] : 0.f;
        __syncthreads();
        sums[tid] += value;
        __syncthreads();
    }
    const float offset = tid == 0 ? 0.f : sums[tid - 1];
    if (chunk <= 4) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (start + i < end) {
                dst[row_offset + start + i] = cumsum_from_float<T>(prefixes[i] + offset);
            }
        }
    } else {
        acc = 0.f;
        for (size_t col = start; col < end; ++col) {
            acc += float(src[row_offset + col]);
            dst[row_offset + col] = cumsum_from_float<T>(acc + offset);
        }
    }
}

// Three-pass low-precision scan without a global f32 prefix buffer. Both
// tile passes use the same f32 scan tree; only final outputs are rounded.
template<typename T, bool Write>
static __device__ void k_cumsum_lowp_tile(const T *src, T *dst, float *totals,
                           const float *offsets, size_t cols, size_t tiles,
                           size_t row, size_t tile) {
    using Load = cub::BlockLoad<float, 256, 16, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using Scan = cub::BlockScan<float, 256, cub::BLOCK_SCAN_WARP_SCANS>;
    using Tail = cub::WarpScan<float, 32>;
    using Store = cub::BlockStore<T, 256, 16, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    __shared__ union {
        typename Load::TempStorage load;
        typename Scan::TempStorage scan;
        typename Tail::TempStorage tail;
        typename Store::TempStorage store;
    } storage;
    const size_t base = row * cols + tile * 4096;
    const int valid = int(min(size_t(4096), cols - tile * 4096));
    const float offset = Write && tile != 0 ? offsets[row * tiles + tile - 1] : 0.f;
    if (valid == 1) {
        if (threadIdx.x == 0) {
            const float value = float(src[base]);
            if constexpr (Write) dst[base] = cumsum_from_float<T>(tile == 0 ? value : value + offset);
            else totals[row * tiles + tile] = value;
        }
        return;
    }
    if (valid <= 32) {
        if (threadIdx.x < 32) {
            const int lane = threadIdx.x;
            const float value = lane < valid ? float(src[base + lane]) : 0.f;
            float prefix;
            Tail(storage.tail).InclusiveSum(value, prefix);
            if constexpr (Write) {
                if (lane < valid) dst[base + lane] = cumsum_from_float<T>(tile == 0 ? prefix : prefix + offset);
            } else if (lane == valid - 1) totals[row * tiles + tile] = prefix;
        }
        return;
    }
    float items[16], total;
    Load(storage.load).Load(src + base, items, valid, 0.f);
    __syncthreads();
    Scan(storage.scan).InclusiveSum(items, items, total);
    if constexpr (Write) {
        T output[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) output[i] = cumsum_from_float<T>(tile == 0 ? items[i] : items[i] + offset);
        __syncthreads();
        Store(storage.store).Store(dst + base, output, valid);
    } else if (threadIdx.x == 0) totals[row * tiles + tile] = total;
}

#define CUMSUM_LOWP_TILE_OPS(TYPENAME, RUST_NAME, SUFFIX, GRID_STRIDE, SINGLE_ROW, TWO_TILES) \
extern "C" __global__ void cumsum_tile_totals##SUFFIX##_##RUST_NAME( \
    const TYPENAME *src, float *totals, const size_t ncols, const size_t ntiles, const size_t ntasks, const uint32_t nrows, const uint32_t row_reciprocal \
) { \
    size_t task = blockIdx.x; \
    if (GRID_STRIDE && task >= ntasks) return; \
    do { \
        const size_t tile = TWO_TILES ? (task & 1) : (SINGLE_ROW ? task : cumsum_task_quotient<GRID_STRIDE>(task, nrows, row_reciprocal)); \
        const size_t row = TWO_TILES ? (task >> 1) : (SINGLE_ROW ? 0 : (GRID_STRIDE ? task - tile * nrows : uint32_t(task) - uint32_t(tile) * nrows)); \
        k_cumsum_lowp_tile<TYPENAME, false>(src, nullptr, totals, nullptr, ncols, ntiles, row, tile); \
        if constexpr (!GRID_STRIDE) break; \
        task = cumsum_next_task<GRID_STRIDE>(task, ntasks); \
    } while (task < ntasks); \
} \
extern "C" __global__ void cumsum_rescan##SUFFIX##_##RUST_NAME( \
    const TYPENAME *src, TYPENAME *dst, const float *offsets, \
    const size_t ncols, const size_t ntiles, const size_t ntasks, const uint32_t nrows, const uint32_t row_reciprocal \
) { \
    size_t task = blockIdx.x; \
    if (GRID_STRIDE && task >= ntasks) return; \
    do { \
        const size_t tile = TWO_TILES ? (task & 1) : (SINGLE_ROW ? task : cumsum_task_quotient<GRID_STRIDE>(task, nrows, row_reciprocal)); \
        const size_t row = TWO_TILES ? (task >> 1) : (SINGLE_ROW ? 0 : (GRID_STRIDE ? task - tile * nrows : uint32_t(task) - uint32_t(tile) * nrows)); \
        k_cumsum_lowp_tile<TYPENAME, true>(src, dst, nullptr, offsets, ncols, ntiles, row, tile); \
        if constexpr (!GRID_STRIDE) break; \
        task = cumsum_next_task<GRID_STRIDE>(task, ntasks); \
    } while (task < ntasks); \
}

#define CUMSUM_LOWP_OP(TYPENAME, RUST_NAME) \
extern "C" __global__ void cumsum_last_dim_##RUST_NAME( \
    const TYPENAME *src, TYPENAME *dst, const size_t ncols, const int nthreads \
) { \
    k_cumsum_last_dim_lowp(src, dst, ncols, nthreads); \
} \
CUMSUM_LOWP_TILE_OPS(TYPENAME, RUST_NAME, , false, false, false) \
CUMSUM_LOWP_TILE_OPS(TYPENAME, RUST_NAME, _single_row, false, true, false) \
CUMSUM_LOWP_TILE_OPS(TYPENAME, RUST_NAME, _two_tiles, false, false, true) \
CUMSUM_LOWP_TILE_OPS(TYPENAME, RUST_NAME, _grid_stride, true, false, false)

CUMSUM_LOWP_OP(__half, f16)
CUMSUM_LOWP_OP(__nv_bfloat16, bf16)
