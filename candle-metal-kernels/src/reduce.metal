#include <metal_stdlib>
using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))

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

constant int THREADGROUP_SIZE = 2048;

# define REDUCE(FN, NAME, T) \
kernel void NAME( \
    constant size_t &src_numel, \
    constant size_t &el_to_sum_per_block, \
    device const T *src,  \
    device T *dst, \
    uint id [[ thread_position_in_grid ]], \
    uint tid [[ thread_index_in_threadgroup ]], \
    uint dst_id [[ threadgroup_position_in_grid ]], \
    uint block_dim [[ threads_per_threadgroup ]] \
) { \
     \
   threadgroup T shared_memory[THREADGROUP_SIZE]; \
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
     T x = shared_memory[tid]; \
     T y = src[idx]; \
     shared_memory[tid] = FN; \
     idx += block_dim; \
   } \
      \
   threadgroup_barrier(mem_flags::mem_none); \
    \
   /* \
   // reduction in shared memory \
   */ \
   for (uint s = block_dim / 2; s > 0; s >>= 1) { \
       if (tid < s) { \
           T x = shared_memory[tid]; \
           T y = shared_memory[tid + s]; \
           shared_memory[tid] = FN; \
       } \
       threadgroup_barrier(mem_flags::mem_none); \
   } \
    \
   threadgroup_barrier(mem_flags::mem_none); \
   dst[dst_id] = shared_memory[0]; \
} \


REDUCE(x + y, fast_sum_float, float)
REDUCE(x * y, fast_mul_float, float)
REDUCE(max(x, y), fast_max_float, float)

#define SOFTMAX(NAME, T)                                                          \
kernel void NAME(                                                                 \
    constant size_t &src_numel,                                                   \
    constant size_t &el_to_sum_per_block,                                         \
    device const T *src,                                                          \
    device T *dst,                                                                \
                                                                                  \
    uint id [[ thread_position_in_grid ]],                                        \
    uint tid [[ thread_index_in_threadgroup ]],                                   \
    uint dst_id [[ threadgroup_position_in_grid ]],                               \
    uint block_dim [[ threads_per_threadgroup ]]                                  \
) {                                                                               \
    threadgroup float shared_memory[THREADGROUP_SIZE];                                \
    shared_memory[tid] = -INFINITY;                                            \
    size_t start_idx = dst_id * el_to_sum_per_block;                              \
    size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);            \
    size_t idx = start_idx + tid;                                                 \
                                                                                  \
    threadgroup_barrier(mem_flags::mem_threadgroup);                              \
                                                                                  \
    float tmp = 0; \
    while (idx < stop_idx) {                                                      \
        tmp = MAX(tmp, src[idx]);                   \
        idx += block_dim;                                                         \
    }                                                                             \
    shared_memory[tid] = tmp; \
                                                                                  \
    threadgroup_barrier(mem_flags::mem_threadgroup);                              \
                                                                                  \
    for (uint s = block_dim / 2; s > 0; s >>= 1) {                                \
        if (tid < s) {                                                            \
            shared_memory[tid] = MAX(shared_memory[tid], shared_memory[tid + s]); \
        }                                                                         \
    }                                                                             \
                                                                                  \
    threadgroup_barrier(mem_flags::mem_threadgroup);                              \
                                                                                  \
    float _max = shared_memory[0];                                                    \
                                                                                  \
    threadgroup_barrier(mem_flags::mem_threadgroup);                              \
    shared_memory[tid] = 0;                                                       \
                                                                                  \
    idx = start_idx + tid;                                                        \
    while (idx < stop_idx) {                                                      \
        const float val = exp(float(src[idx]) - _max);                                    \
        dst[idx] = T(val);                                                           \
        shared_memory[tid] += val;                                                \
        idx += block_dim;                                                         \
    }                                                                             \
    for (uint s = block_dim / 2; s > 0; s >>= 1) {                                \
        if (tid < s) {                                                            \
            shared_memory[tid] += shared_memory[tid + s];                         \
        }                                                                         \
    }                                                                             \
                                                                                  \
    const T inv_acc = T(1.0/shared_memory[0]);                                         \
    idx = start_idx + tid;                                                        \
    while (idx < stop_idx) {                                                      \
        dst[idx] *= inv_acc;                                                      \
        idx += block_dim;                                                         \
    }                                                                             \
}                                                                                 \

SOFTMAX(softmax_float, float)
SOFTMAX(softmax_half, half)
#if __METAL_VERSION__ >= 310
SOFTMAX(softmax_bfloat, bfloat)
#endif
