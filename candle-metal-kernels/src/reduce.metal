#include <metal_stdlib>
using namespace metal;

METAL_FUNC uint get_strided_index(
    uint idx,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides
) {
    uint strided_i = 0;
    for (uint d = 0; d < num_dims; d++) {
        uint dim_idx = num_dims - 1 - d;
        strided_i += (idx % dims[dim_idx]) * strides[dim_idx];
        idx /= dims[dim_idx];
    }
    return strided_i;
}

constant int THREADGROUP_SIZE = 256;

# define REDUCE(FN, NAME, TYPENAME) \
kernel void NAME( \
    constant size_t &src_numel, \
    constant size_t &el_to_sum_per_block, \
    device const TYPENAME *src,  \
    device TYPENAME *dst, \
    uint id [[ thread_position_in_grid ]], \
    uint tid [[ thread_index_in_threadgroup ]], \
    uint dst_id [[ threadgroup_position_in_grid ]], \
    uint blockDim [[ threads_per_threadgroup ]] \
) { \
     \
   threadgroup float shared_memory[THREADGROUP_SIZE]; \
      \
   shared_memory[tid] = 0; \
   /* \
   // Elements summed in this block range from dst_id * el_to_sum_per_block  \
   // to (dst_id + 1) * el_to_sum_per_block. \
   */ \
   size_t start_idx = dst_id * el_to_sum_per_block; \
   size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel); \
   size_t idx = start_idx + tid; \
   while (idx < stop_idx) { \
     /* \
     // TODO: Fast version for the contiguous case. \
     // size_t strided_i = get_strided_index(idx, num_dims, dims, strides); \
     */ \
     TYPENAME x = shared_memory[tid]; \
     TYPENAME y = src[idx]; \
     shared_memory[tid] = FN; \
     idx += blockDim; \
   } \
      \
   threadgroup_barrier(mem_flags::mem_none); \
    \
   /* \
   // reduction in shared memory \
   */ \
   for (uint s = blockDim / 2; s > 0; s >>= 1) { \
       if (tid < s) { \
           TYPENAME x = shared_memory[tid]; \
           TYPENAME y = shared_memory[tid + s]; \
           shared_memory[tid] = FN; \
       } \
       threadgroup_barrier(mem_flags::mem_none); \
   } \
    \
   dst[dst_id] = shared_memory[0]; \
} \

kernel void softmax_float(
    constant size_t &src_numel,
    constant size_t &el_to_sum_per_block,
    device const float *src, 
    device float *dst,
    uint id [[ thread_position_in_grid ]],
    uint tid [[ thread_index_in_threadgroup ]],
    uint dst_id [[ threadgroup_position_in_grid ]],
    uint blockDim [[ threads_per_threadgroup ]]
) {
    
   threadgroup float shared_memory[THREADGROUP_SIZE];
     
   shared_memory[tid] = -INFINITY;
   // Elements summed in this block range from dst_id * el_to_sum_per_block
   // to (dst_id + 1) * el_to_sum_per_block.
   size_t start_idx = dst_id * el_to_sum_per_block;
   size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);
   size_t idx = start_idx + tid;

   while (idx < stop_idx) {
     // TODO: Fast version for the contiguous case.
     shared_memory[tid] = max(shared_memory[tid], src[idx]);
     idx += blockDim;
   }
     
   threadgroup_barrier(mem_flags::mem_none);
   
   // reduction in shared memory
   for (uint s = blockDim / 2; s > 0; s >>= 1) {
       if (tid < s) {
           shared_memory[tid] = max(shared_memory[tid], shared_memory[tid + s]);
       }
       threadgroup_barrier(mem_flags::mem_none);
   }
   
   float max = shared_memory[0];

   shared_memory[tid] = 0;

   // Restart
   idx = start_idx + tid;
   while (idx < stop_idx) {
     // TODO: Fast version for the contiguous case.
     const float val = exp(src[idx] - max);
     dst[idx] = val; 
     shared_memory[tid] += val;
     idx += blockDim;
   }
   // reduction in shared memory
   for (uint s = blockDim / 2; s > 0; s >>= 1) {
       if (tid < s) {
           shared_memory[tid] += shared_memory[tid + s];
       }
       threadgroup_barrier(mem_flags::mem_none);
   }

   const float inv_acc = 1/shared_memory[0];
   idx = start_idx + tid;
   while (idx < stop_idx) {
     dst[idx] *= inv_acc; 
     idx += blockDim;
   }
}


REDUCE(x + y, fast_sum_float, float)
REDUCE(x * y, fast_mul_float, float)
REDUCE(max(x, y), fast_max_float, float)
