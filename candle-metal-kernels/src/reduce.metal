#include <metal_stdlib>
using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))

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


#define ARGMIN(NAME, T, MAXVALUE) \
kernel void NAME( \
    constant size_t &num_dims, \
    constant size_t *dims, \
    constant size_t *strides, \
    constant size_t &el_to_sum_per_block, \
    device const T *src, \
    device uint *dst,  \
    uint id [[ thread_position_in_grid ]],  \
    uint tid [[ thread_index_in_threadgroup ]],  \
    uint dst_id [[ threadgroup_position_in_grid ]],  \
    uint block_dim [[ threads_per_threadgroup ]]  \
) {  \
      \
   threadgroup T shared_memory[THREADGROUP_SIZE];  \
   threadgroup uint shared_indices[THREADGROUP_SIZE];  \
       \
   shared_memory[tid] = MAXVALUE;  \
   shared_indices[tid] = 0xFFFFFFFF; \
   bool notset = true; \
   /*  \
   // Elements summed in this block range from dst_id * el_to_sum_per_block   \
   // to (dst_id + 1) * el_to_sum_per_block.  \
   */  \
   size_t start_idx = dst_id * el_to_sum_per_block;  \
   size_t stop_idx = start_idx + el_to_sum_per_block;  \
   size_t idx = start_idx + tid;  \
   while (idx < stop_idx) {  \
     /*  \
     // TODO: Fast version for the contiguous case.  \
     */  \
     size_t strided_i = get_strided_index(idx, num_dims, dims, strides);  \
     if (notset || src[strided_i] < shared_memory[tid]) {  \
         shared_memory[tid] = src[strided_i];  \
          /* Assume that the reduction takes place over the last dimension which is contiguous. */ \
          shared_indices[tid] = idx % dims[num_dims - 1]; \
          notset = false; \
     }  \
     idx += block_dim;  \
   }  \
       \
   threadgroup_barrier(mem_flags::mem_none);  \
     \
   /*  \
   // reduction in shared memory  \
   */  \
   for (uint s = block_dim / 2; s > 0; s >>= 1) {  \
       if (tid < s && shared_memory[tid + s] < shared_memory[tid]) {  \
           shared_indices[tid] = shared_indices[tid + s];  \
           shared_memory[tid] = shared_memory[tid + s];  \
       }  \
       threadgroup_barrier(mem_flags::mem_none);  \
   }  \
     \
     if (tid == 0){ \
       dst[dst_id] = shared_indices[0];  \
     } \
} \


#define ARGMAX(NAME, T, MINVALUE) \
kernel void NAME( \
    constant size_t &num_dims, \
    constant size_t *dims, \
    constant size_t *strides, \
    constant size_t &el_to_sum_per_block, \
    device const T *src, \
    device uint *dst,  \
    uint id [[ thread_position_in_grid ]],  \
    uint tid [[ thread_index_in_threadgroup ]],  \
    uint dst_id [[ threadgroup_position_in_grid ]],  \
    uint block_dim [[ threads_per_threadgroup ]]  \
) {  \
      \
   threadgroup T shared_memory[THREADGROUP_SIZE];  \
   threadgroup uint shared_indices[THREADGROUP_SIZE];  \
       \
   shared_memory[tid] = MINVALUE;  \
   shared_indices[tid] = 0xFFFFFFFF; \
   /*  \
   // Elements summed in this block range from dst_id * el_to_sum_per_block   \
   // to (dst_id + 1) * el_to_sum_per_block.  \
   */  \
   size_t start_idx = dst_id * el_to_sum_per_block;  \
   size_t stop_idx = start_idx + el_to_sum_per_block;  \
   size_t idx = start_idx + tid;  \
   bool notset = true; \
   while (idx < stop_idx) {  \
     /*  \
     // TODO: Fast version for the contiguous case.  \
     */  \
     size_t strided_i = get_strided_index(idx, num_dims, dims, strides);  \
     if (notset || shared_memory[tid] < src[strided_i]) {  \
         shared_memory[tid] = src[strided_i];  \
         shared_indices[tid] = idx % dims[num_dims - 1]; \
         notset = false; \
     }  \
     idx += block_dim;  \
   }  \
       \
   threadgroup_barrier(mem_flags::mem_none);  \
     \
   /*  \
   // reduction in shared memory  \
   */  \
   for (uint s = block_dim / 2; s > 0; s >>= 1) {  \
       if (tid < s && shared_memory[tid + s] > shared_memory[tid]) {  \
           shared_indices[tid] = shared_indices[tid + s];  \
           shared_memory[tid] = shared_memory[tid + s];  \
       }  \
       threadgroup_barrier(mem_flags::mem_none);  \
   }  \
     \
   if (tid == 0){ \
       dst[dst_id] = shared_indices[0];  \
   } \
} \

#define REDUCE(FN, NAME, T, START) \
kernel void NAME( \
    constant size_t &num_dims, \
    constant size_t *dims, \
    constant size_t *strides, \
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
   shared_memory[tid] = START; \
   /* \
   // Elements summed in this block range from dst_id * el_to_sum_per_block  \
   // to (dst_id + 1) * el_to_sum_per_block. \
   */ \
   size_t start_idx = dst_id * el_to_sum_per_block; \
   size_t stop_idx = start_idx + el_to_sum_per_block; \
   size_t idx = start_idx + tid; \
   while (idx < stop_idx) { \
     /* \
     // TODO: Fast version for the contiguous case. \
     */ \
     size_t strided_i = get_strided_index(idx, num_dims, dims, strides); \
     T x = shared_memory[tid]; \
     T y = src[strided_i]; \
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
   dst[dst_id] = shared_memory[0]; \
} \


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
                                                                                  \
    float tmp = -INFINITY; \
    while (idx < stop_idx) {                                                      \
        tmp = MAX(tmp, float(src[idx]));                   \
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
        threadgroup_barrier(mem_flags::mem_threadgroup);                              \
    }                                                                             \
                                                                                  \
    /* wait for shared_memory[0] to be filled */ \
    threadgroup_barrier(mem_flags::mem_threadgroup);                              \
                                                                                  \
    float _max = shared_memory[0];                                                    \
                                                                                  \
    /* prevent tid=0 from overwriting _max before other threads have written it */ \
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
    threadgroup_barrier(mem_flags::mem_threadgroup);                              \
    for (uint s = block_dim / 2; s > 0; s >>= 1) {                                \
        if (tid < s) {                                                            \
            shared_memory[tid] += shared_memory[tid + s];                         \
        }                                                                         \
        threadgroup_barrier(mem_flags::mem_threadgroup);                              \
    }                                                                             \
                                                                                  \
    const T inv_acc = T(1.0/shared_memory[0]);                                         \
    idx = start_idx + tid;                                                        \
    while (idx < stop_idx) {                                                      \
        dst[idx] *= inv_acc;                                                      \
        idx += block_dim;                                                         \
    }                                                                             \
}                                                                                 \

#define RMSNORM(NAME, T)                                                          \
kernel void NAME(                                                                 \
    constant size_t &src_numel,                                                   \
    constant size_t &el_to_sum_per_block,                                         \
    device const T *src,                                                          \
    device T *dst,                                                                \
    device const T *alpha,                                                        \
    constant float &eps,                                                          \
                                                                                  \
    uint id [[ thread_position_in_grid ]],                                        \
    uint tid [[ thread_index_in_threadgroup ]],                                   \
    uint dst_id [[ threadgroup_position_in_grid ]],                               \
    uint block_dim [[ threads_per_threadgroup ]]                                  \
) {                                                                               \
    threadgroup float shared_memory[THREADGROUP_SIZE];                            \
    shared_memory[tid] = 0;                                                       \
    size_t start_idx = dst_id * el_to_sum_per_block;                              \
    size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);            \
    size_t idx = start_idx + tid;                                                 \
                                                                                  \
                                                                                  \
    float tmp = 0;                                                                \
    while (idx < stop_idx) {                                                      \
        tmp = tmp + float(src[idx]) * float(src[idx]);                            \
        idx += block_dim;                                                         \
    }                                                                             \
    shared_memory[tid] = tmp;                                                     \
                                                                                  \
    threadgroup_barrier(mem_flags::mem_threadgroup);                              \
                                                                                  \
    for (uint s = block_dim / 2; s > 0; s >>= 1) {                                \
        if (tid < s) {                                                            \
            shared_memory[tid] = shared_memory[tid] + shared_memory[tid + s];     \
        }                                                                         \
        threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    }                                                                             \
                                                                                  \
    /* wait for shared_memory[0] to be filled */ \
    threadgroup_barrier(mem_flags::mem_threadgroup);                              \
                                                                                  \
    float norm = sqrt(shared_memory[0] / float(el_to_sum_per_block) + eps);       \
    float inv_norm = 1.0f / norm;                                                 \
    idx = start_idx + tid;                                                        \
    while (idx < stop_idx) {                                                      \
        float val = float(src[idx]) * inv_norm;                                   \
        if (alpha != nullptr) {                                                   \
            val *= float(alpha[idx - start_idx]);                                 \
        }                                                                         \
        dst[idx] = T(val);                                                        \
        idx += block_dim;                                                         \
    }                                                                             \
}                                                                                 \

#define ROPEI(FN_NAME, TYPENAME) \
kernel void FN_NAME( \
    constant size_t &bh, \
    constant size_t &td, \
    device const TYPENAME *src,  \
    device const TYPENAME *cos,  \
    device const TYPENAME *sin,  \
    device TYPENAME *dst, \
    uint tid [[ thread_position_in_grid ]] \
) { \
    if (2 * tid >= bh * td) { \
        return; \
    } \
    size_t rope_idx = tid % (td / 2); \
    TYPENAME c = cos[rope_idx]; \
    TYPENAME s = sin[rope_idx]; \
    dst[2 * tid] = src[2 * tid] * c - src[2 * tid + 1] * s; \
    dst[2 * tid + 1] = src[2 * tid] * s + src[2 * tid + 1] * c; \
}\

REDUCE(x + y, fast_sum_f32_strided, float, 0)
REDUCE(x + y, fast_sum_u32_strided, uint, 0)
REDUCE(x + y, fast_sum_f16_strided, half, 0)
REDUCE(x + y, fast_sum_u8_strided, uint8_t, 0)
REDUCE(x * y, fast_mul_f32_strided, float, 1)
REDUCE(x * y, fast_mul_u32_strided, uint, 1)
REDUCE(x * y, fast_mul_f16_strided, half, 1)
REDUCE(MAX(x, y), fast_max_f32_strided, float, -HUGE_VALF)
REDUCE(MAX(x, y), fast_max_u32_strided, uint, 0)
REDUCE(MAX(x, y), fast_max_f16_strided, half, -HUGE_VALH)
REDUCE(MAX(x, y), fast_max_u8_strided, uint8_t, 0)
REDUCE(MIN(x, y), fast_min_f32_strided, float, HUGE_VALF)
REDUCE(MIN(x, y), fast_min_u32_strided, uint, 0xFFFFFFFF)
REDUCE(MIN(x, y), fast_min_f16_strided, half, HUGE_VALH)
REDUCE(MIN(x, y), fast_min_u8_strided, uint8_t, 0xFF)
ARGMIN(fast_argmin_f32_strided, float, HUGE_VALF)
ARGMIN(fast_argmin_f16_strided, half, HUGE_VALH)
ARGMIN(fast_argmin_u32_strided, uint, 0xFFFFFFFF)
ARGMIN(fast_argmin_u8_strided, uint8_t, 0xFF)
ARGMAX(fast_argmax_f32_strided, float, -HUGE_VALF)
ARGMAX(fast_argmax_f16_strided, half, -HUGE_VALH)
ARGMAX(fast_argmax_u32_strided, uint, 0)
ARGMAX(fast_argmax_u8_strided, uint8_t, 0)

SOFTMAX(softmax_f32, float)
SOFTMAX(softmax_f16, half)
RMSNORM(rmsnorm_f32, float)
RMSNORM(rmsnorm_f16, half)
ROPEI(rope_i_f32, float)
ROPEI(rope_i_f16, half)

#if __METAL_VERSION__ >= 220
REDUCE(x + y, fast_sum_i64_strided, int64_t, 0)
REDUCE(MIN(x, y), fast_min_i64_strided, int64_t, INT_MAX)
REDUCE(MAX(x, y), fast_max_i64_strided, int64_t, INT_MIN)
ARGMIN(fast_argmin_i64_strided, int64_t, INT_MAX)
ARGMAX(fast_argmax_i64_strided, int64_t, INT_MIN)
#endif

#if defined(__HAVE_BFLOAT__)
REDUCE(x + y, fast_sum_bf16, bfloat, 0)
REDUCE(x * y, fast_mul_bf16, bfloat, 1)
REDUCE(MAX(x, y), fast_max_bf16, bfloat, -HUGE_VALBF)
REDUCE(MIN(x, y), fast_min_bf16, bfloat, HUGE_VALBF)
ARGMIN(fast_argmin_bf16, bfloat, HUGE_VALBF)
ARGMAX(fast_argmax_bf16, bfloat, -HUGE_VALBF)
SOFTMAX(softmax_bf16, bfloat)
RMSNORM(rmsnorm_bf16, bfloat)
ROPEI(rope_i_bf16, bfloat)
#endif
