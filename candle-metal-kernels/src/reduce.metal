#include <metal_stdlib>
#include <metal_limits>
using namespace metal;

METAL_FUNC uint get_strided_index(
    uint idx,
    constant const size_t &num_dims,
    constant const size_t *dims,
    constant const size_t *strides
) {
    uint strided_i = 0;
    #pragma clang loop unroll(full)
    for (uint d = 0; d < num_dims; d++) {
        uint dim_idx = num_dims - 1 - d;
        strided_i += (idx % dims[dim_idx]) * strides[dim_idx];
        idx /= dims[dim_idx];
    }
    return strided_i;
}

#define impl_reduction_op(name, op, init_val)                                       \
template<typename T>                                                                \
struct name {                                                                       \
                                                                                    \
    static constexpr constant T init = init_val;                                    \
                                                                                    \
    METAL_FUNC T operator()(thread const T &a, thread const T &b) const {           \
        return op;                                                                  \
    }                                                                               \
                                                                                    \
    METAL_FUNC T operator()(threadgroup const T &a, threadgroup const T &b) const { \
        return op;                                                                  \
    }                                                                               \
                                                                                    \
    METAL_FUNC T operator()(device const T &a, device const T &b) const {           \
        return op;                                                                  \
    }                                                                               \
                                                                                    \
    METAL_FUNC T operator()(T a, T b) {                                             \
        return op;                                                                  \
    }                                                                               \
}                                                                                   \

impl_reduction_op(Sum, a + b, 0);
impl_reduction_op(Mul, a * b, 1);
impl_reduction_op(Min, a < b ? a : b, numeric_limits<T>::max());
impl_reduction_op(Max, a > b ? a : b, numeric_limits<T>::min());
#undef impl_reduction_op

static constant constexpr int THREADGROUP_SIZE = 2048;

template<typename T, typename ReductionOp, uint BLOCKSIZE>
METAL_FUNC void load_from_global(
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides,
    constant size_t &el_to_sum_per_block,
    device const T *src,
    threadgroup T shared[BLOCKSIZE],
    uint tid [[ thread_index_in_threadgroup ]],
    uint dst_id [[ threadgroup_position_in_grid ]],
    uint block_dim [[ threads_per_threadgroup ]]
) {
    ReductionOp op;

    size_t start_idx = dst_id * el_to_sum_per_block;
    size_t stop_idx = start_idx + el_to_sum_per_block;
    size_t idx = start_idx + tid;
    while (idx < stop_idx) {
        size_t strided_i = get_strided_index(idx, num_dims, dims, strides);
        shared[tid] = op(shared[tid], src[strided_i]);
        idx += block_dim;
    }
    threadgroup_barrier(mem_flags::mem_none);
}

template<typename T, typename ReductionOp, uint BLOCKSIZE>
METAL_FUNC void load_from_global(
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t &el_to_sum_per_block,
    device const T *src,
    threadgroup T shared[BLOCKSIZE],
    uint tid [[ thread_index_in_threadgroup ]],
    uint dst_id [[ threadgroup_position_in_grid ]],
    uint block_dim [[ threads_per_threadgroup ]]
) {
    ReductionOp op;

    size_t start_idx = dst_id * el_to_sum_per_block;
    size_t stop_idx = start_idx + el_to_sum_per_block;
    size_t idx = start_idx + tid;
    while (idx < stop_idx) {
        shared[tid] = op(shared[tid], src[idx]);
        idx += block_dim;
    }
    threadgroup_barrier(mem_flags::mem_none);
}

template<typename T, typename ReductionOp, uint BLOCKSIZE>
METAL_FUNC void threadgroup_reduce(
    threadgroup T shared[BLOCKSIZE],
    uint tid [[thread_index_in_threadgroup]],
    uint block_dim [[ threads_per_threadgroup ]]
) {
    ReductionOp op;
    if (BLOCKSIZE >= 64) {
        if (block_dim >= 64) {
            shared[tid] = op(shared[tid], shared[tid + 32]);
        }
    }
    if (BLOCKSIZE >= 32) {
        if (block_dim >= 32) {
            shared[tid] = op(shared[tid], shared[tid + 16]);
        }
    }
    if (BLOCKSIZE >= 16) {
        if (block_dim >= 16) {
            shared[tid] = op(shared[tid], shared[tid + 8]);
        }
    }
    if (BLOCKSIZE >= 8) {
        if (block_dim >= 8) {
            shared[tid] = op(shared[tid], shared[tid + 4]);
        }
    }
    if (BLOCKSIZE >= 4) {
        if (block_dim >= 4) {
            shared[tid] = op(shared[tid], shared[tid + 2]);
        }
    }
    if (BLOCKSIZE >= 2) {
        if (block_dim >= 2) {
            shared[tid] = op(shared[tid], shared[tid + 1]);
        }
    }
}

// Inspired by "Optimizing Parallel Reduction in CUDA" by Mark Harris
template<
    typename T,
    typename ReductionOp,
    uint BLOCKSIZE,
    bool STRIDED
>
METAL_FUNC void block_reduce(
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides,
    constant size_t &el_to_sum_per_block,
    device const T *src,
    device T *dst,
    constant uint &num_elements,
    threadgroup T shared[BLOCKSIZE],
    uint id [[ thread_position_in_grid ]],
    uint tid [[ thread_index_in_threadgroup ]],
    uint dst_id [[ threadgroup_position_in_grid ]],
    uint block_dim [[ threads_per_threadgroup ]]
) {
    ReductionOp op;

    shared[tid] = ReductionOp::init;

    if (STRIDED) {
        load_from_global<T, ReductionOp, BLOCKSIZE>(
            num_dims,
            dims,
            strides,
            el_to_sum_per_block,
            src,
            shared,
            tid,
            dst_id,
            block_dim
        );
    } else {
        load_from_global<T, ReductionOp, BLOCKSIZE>(
            num_dims,
            dims,
            el_to_sum_per_block,
            src,
            shared,
            tid,
            dst_id,
            block_dim
        );
    }

    if (BLOCKSIZE >= 1024) {
        if (tid < 512 && block_dim >= 1024) {
            shared[tid] = op(shared[tid], shared[tid + 512]);
            threadgroup_barrier(mem_flags::mem_none);
        }
    }
    if (BLOCKSIZE >= 512) {
        if (tid < 256 && block_dim >= 512) {
            shared[tid] = op(shared[tid], shared[tid + 256]);
            threadgroup_barrier(mem_flags::mem_none);
        }
    }
    if (BLOCKSIZE >= 256) {
        if (tid < 128 && block_dim >= 256) {
            shared[tid] = op(shared[tid], shared[tid + 128]);
            threadgroup_barrier(mem_flags::mem_none);
        }
    }
    if (BLOCKSIZE >= 128) {
        if (tid < 64 && block_dim >= 128) {
            shared[tid] = op(shared[tid], shared[tid + 64]);
            threadgroup_barrier(mem_flags::mem_none);
        }
    }
    if (tid < 32) {
        threadgroup_reduce<T, ReductionOp, BLOCKSIZE>(shared, tid, block_dim);
        threadgroup_barrier(mem_flags::mem_none);
    }

    if (tid == 0) {
        dst[dst_id] = shared[tid];
    }
}

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))

static constant constexpr int BLOCKSIZE = 2048;

#define REDUCE(OP, NAME, T)                         \
kernel void NAME(                                   \
    constant size_t &num_dims,                      \
    constant size_t *dims,                          \
    constant size_t *strides,                       \
    constant size_t &el_to_sum_per_block,           \
    device const T *src,                            \
    device T *dst,                                  \
    constant uint &num_elements,                    \
    uint id [[ thread_position_in_grid ]],          \
    uint tid [[ thread_index_in_threadgroup ]],     \
    uint dst_id [[ threadgroup_position_in_grid ]], \
    uint block_dim [[ threads_per_threadgroup ]]    \
) {                                                 \
    threadgroup T shared[BLOCKSIZE];                \
    block_reduce<T, OP<T>, BLOCKSIZE, false>(       \
        num_dims,                                   \
        dims,                                       \
        strides,                                    \
        el_to_sum_per_block,                        \
        src,                                        \
        dst,                                        \
        num_elements,                               \
        shared,                                     \
        id,                                         \
        tid,                                        \
        dst_id,                                     \
        block_dim);                                 \
}                                                   \
kernel void NAME##_strided(                         \
    constant size_t &num_dims,                      \
    constant size_t *dims,                          \
    constant size_t *strides,                       \
    constant size_t &el_to_sum_per_block,           \
    device const T *src,                            \
    device T *dst,                                  \
    constant uint &num_elements,                    \
    uint id [[ thread_position_in_grid ]],          \
    uint tid [[ thread_index_in_threadgroup ]],     \
    uint dst_id [[ threadgroup_position_in_grid ]], \
    uint block_dim [[ threads_per_threadgroup ]]    \
) {                                                 \
    threadgroup T shared[BLOCKSIZE];                \
    block_reduce<T, OP<T>, BLOCKSIZE, false>(       \
        num_dims,                                   \
        dims,                                       \
        strides,                                    \
        el_to_sum_per_block,                        \
        src,                                        \
        dst,                                        \
        num_elements,                               \
        shared,                                     \
        id,                                         \
        tid,                                        \
        dst_id,                                     \
        block_dim);                                 \
}                                                   \

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
}

REDUCE(Sum, fast_sum_f32, float)
REDUCE(Sum, fast_sum_u32, uint)
REDUCE(Sum, fast_sum_f16, half)
REDUCE(Sum, fast_sum_u8, uint8_t)
REDUCE(Mul, fast_mul_f32, float)
REDUCE(Mul, fast_mul_u32, uint)
REDUCE(Mul, fast_mul_f16, half)
REDUCE(Max, fast_max_f32, float)
REDUCE(Max, fast_max_u32, uint)
REDUCE(Max, fast_max_f16, half)
REDUCE(Max, fast_max_u8, uint8_t)
REDUCE(Min, fast_min_f32, float)
REDUCE(Min, fast_min_u32, uint)
REDUCE(Min, fast_min_f16, half)
REDUCE(Min, fast_min_u8, uint8_t)

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

#if __METAL_VERSION__ >= 220
REDUCE(Sum, fast_sum_i64, int64_t)
REDUCE(Mul, fast_mul_i64, int64_t)
REDUCE(Min, fast_min_i64, int64_t)
REDUCE(Max, fast_max_i64, int64_t)

ARGMIN(fast_argmin_i64_strided, int64_t, INT_MAX)
ARGMAX(fast_argmax_i64_strided, int64_t, INT_MIN)
#endif

#if __METAL_VERSION__ >= 310
REDUCE(Sum, fast_sum_bf16, bfloat)
REDUCE(Mul, fast_mul_bf16, bfloat)
REDUCE(Max, fast_max_bf16, bfloat)
REDUCE(Min, fast_min_bf16, bfloat)

ARGMIN(fast_argmin_bf16, bfloat, HUGE_VALBF)
ARGMAX(fast_argmax_bf16, bfloat, -HUGE_VALBF)
SOFTMAX(softmax_bf16, bfloat)
#endif
