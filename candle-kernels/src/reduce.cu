// TODO: Use a proper distributed reduction rather than atomicAdd.
// https://people.maths.ox.ac.uk/gilesm/cuda/prac4/reduction.pdf
#include "cuda_utils.cuh"
#include<stdint.h>

const int BLOCK_SIZE = 1024;

// TODO: Maybe add some fast_sum_f16_f32 variant that not only accumulate in f32 but
// also expect a f32 output so that this can be used for normalization e.g. in softmax.

// Fast reduce sum kernel, this assumes that the dimensions to loop over are at
// the end, each block is responsible for populating one value in the output array.
// There are at most 1024 threads per block.
template <typename T>
__device__ void fast_sum(
    const size_t src_numel,
    const size_t el_to_sum_per_block,
    const size_t num_dims, 
    const size_t *info,
    const T *src,
    T *dst
) {
  const size_t *dims = info;
  const size_t *strides = info + num_dims;

  __shared__ T shr[BLOCK_SIZE];
  size_t tid = threadIdx.x;
  size_t dst_id = blockIdx.x;

  shr[tid] = 0.0;
  // Elements summed in this block range from dst_id * el_to_sum_per_block
  // to (dst_id + 1) * el_to_sum_per_block.
  size_t start_idx = dst_id * el_to_sum_per_block;
  size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);
  size_t idx = start_idx + tid;

  while (idx < stop_idx) {
    // TODO: Fast version for the contiguous case.
    size_t strided_i = get_strided_index(idx, num_dims, dims, strides);
    shr[tid] += src[strided_i];
    idx += blockDim.x;
  }

  // Parallel reduction, see the slides:
  // https://www.olcf.ornl.gov/wp-content/uploads/2019/12/05_Atomics_Reductions_Warp_Shuffle.pdf
  // https://stackoverflow.com/questions/66078814/is-cuda-atomicadd-operation-faster-than-launch-another-kernel-when-we-do-reduce
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    if (tid < s) shr[tid] += shr[tid + s];
  }

  if (tid == 0) atomicAdd(dst + dst_id, shr[0]);
}

#define FAST_SUM_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t src_numel, \
    const size_t el_to_sum_per_block, \
    const size_t num_dims,  \
    const size_t *info, \
    const TYPENAME *src, \
    TYPENAME *dst \
) {  \
  fast_sum(src_numel, el_to_sum_per_block, num_dims, info, src, dst); \
} \

#define SUM_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t numel,  \
    const size_t num_dims, \
    const size_t num_sum_dims, \
    const size_t *info, \
    const TYPENAME *inp, \
    TYPENAME *out \
) {  \
    const size_t *dims = info; \
    const size_t *strides = info + num_dims; \
    const size_t *sum_dims_l = info + 2*num_dims; \
    const size_t *sum_dims_s = info + 2*num_dims + num_sum_dims; \
    if (is_contiguous(num_dims, dims, strides)) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            size_t dst_index = i; \
            for (unsigned int nd = 0; nd < num_sum_dims; ++nd) { \
              size_t stride = sum_dims_s[nd]; \
              size_t pre = dst_index / stride; \
              size_t post = dst_index % stride; \
              dst_index = (pre / sum_dims_l[nd]) * stride + post; \
            } \
            atomicAdd(out + dst_index, inp[i]); \
        } \
    } \
    else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides); \
            size_t dst_index = i; \
            for (unsigned int nd = 0; nd < num_sum_dims; ++nd) { \
              size_t stride = sum_dims_s[nd]; \
              size_t pre = dst_index / stride; \
              size_t post = dst_index % stride; \
              dst_index = (pre / sum_dims_l[nd]) * stride + post; \
            } \
            atomicAdd(out + dst_index, inp[strided_i]); \
        } \
    } \
} \

#if __CUDA_ARCH__ >= 800
SUM_OP(__nv_bfloat16, sum_bf16)
FAST_SUM_OP(__nv_bfloat16, fast_sum_bf16)
#endif

#if __CUDA_ARCH__ >= 530
SUM_OP(__half, sum_f16)
FAST_SUM_OP(__half, fast_sum_f16)
#endif

SUM_OP(float, sum_f32)
SUM_OP(double, sum_f64)
SUM_OP(uint32_t, sum_u32)

FAST_SUM_OP(float, fast_sum_f32)
FAST_SUM_OP(double, fast_sum_f64)
FAST_SUM_OP(uint32_t, fast_sum_u32)
