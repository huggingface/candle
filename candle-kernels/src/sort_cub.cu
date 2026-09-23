#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace {

__global__ void init_argsort_indices(uint32_t *indices, int num_items,
                                     int ncols) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t item = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
       item < static_cast<int64_t>(num_items);
       item += stride) {
    indices[item] = static_cast<uint32_t>(item % static_cast<int64_t>(ncols));
  }
}

__global__ void init_argsort_offsets(int *offsets, int nrows, int ncols) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
       row <= static_cast<int64_t>(nrows);
       row += stride) {
    offsets[row] = static_cast<int>(row * ncols);
  }
}

cudaError_t segmented_argsort_f32(
    void *temp_storage, size_t &temp_storage_bytes, const float *keys_in,
    float *keys_out, const uint32_t *indices_in, uint32_t *indices_out,
    const int *offsets, int num_items, int nrows, bool descending,
    cudaStream_t stream) {
  const int *end_offsets = offsets == nullptr ? nullptr : offsets + 1;
  if (descending) {
    return cub::DeviceSegmentedRadixSort::SortPairsDescending(
        temp_storage, temp_storage_bytes, keys_in, keys_out, indices_in,
        indices_out, num_items, nrows, offsets, end_offsets, 0,
        static_cast<int>(sizeof(float) * 8), stream);
  }
  return cub::DeviceSegmentedRadixSort::SortPairs(
      temp_storage, temp_storage_bytes, keys_in, keys_out, indices_in,
      indices_out, num_items, nrows, offsets, end_offsets, 0,
      static_cast<int>(sizeof(float) * 8), stream);
}

} // namespace

extern "C" int candle_argsort_f32(
    const float *keys_in, float *keys_out, uint32_t *indices_in,
    uint32_t *indices_out, int *offsets, void *temp_storage,
    size_t *temp_storage_bytes, int nrows, int ncols, int descending,
    cudaStream_t stream) {
  if (temp_storage_bytes == nullptr || nrows <= 0 || ncols <= 0 ||
      nrows > std::numeric_limits<int>::max() / ncols) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const int num_items = nrows * ncols;
  if (temp_storage != nullptr) {
    if (keys_in == nullptr || keys_out == nullptr || indices_in == nullptr ||
        indices_out == nullptr || offsets == nullptr) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
    constexpr int block_size = 256;
    const int required_blocks = (num_items - 1) / block_size + 1;
    const int blocks = required_blocks < 65535 ? required_blocks : 65535;
    init_argsort_indices<<<blocks, block_size, 0, stream>>>(indices_in,
                                                            num_items, ncols);
    const cudaError_t init_status = cudaGetLastError();
    if (init_status != cudaSuccess) {
      return static_cast<int>(init_status);
    }
    const int required_offset_blocks = nrows / block_size + 1;
    init_argsort_offsets<<<required_offset_blocks, block_size, 0, stream>>>(
        offsets, nrows, ncols);
    const cudaError_t offset_status = cudaGetLastError();
    if (offset_status != cudaSuccess) {
      return static_cast<int>(offset_status);
    }
  }

  const cudaError_t sort_status = segmented_argsort_f32(
      temp_storage, *temp_storage_bytes, keys_in, keys_out, indices_in,
      indices_out, offsets, num_items, nrows, descending != 0, stream);
  return static_cast<int>(sort_status);
}
