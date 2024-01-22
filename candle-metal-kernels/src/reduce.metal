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
template<typename T, typename R = T>                                                \
struct name {                                                                       \
                                                                                    \
    static constexpr constant T init = init_val;                                    \
                                                                                    \
    METAL_FUNC R operator()(thread const T &a, thread const T &b) const {           \
        return op;                                                                  \
    }                                                                               \
                                                                                    \
    METAL_FUNC R operator()(threadgroup const T &a, threadgroup const T &b) const { \
        return op;                                                                  \
    }                                                                               \
                                                                                    \
    METAL_FUNC R operator()(device const T &a, device const T &b) const {           \
        return op;                                                                  \
    }                                                                               \
                                                                                    \
    METAL_FUNC R operator()(T a, T b) {                                             \
        return op;                                                                  \
    }                                                                               \
}                                                                                   \

impl_reduction_op(Sum, a + b, 0);
impl_reduction_op(Mul, a * b, 1);
impl_reduction_op(Min, a < b ? a : b, numeric_limits<T>::max());
impl_reduction_op(Max, a > b ? a : b, numeric_limits<T>::min());
impl_reduction_op(ArgMin, a < b, numeric_limits<T>::max());
impl_reduction_op(ArgMax, a > b, numeric_limits<T>::min());
#undef impl_reduction_op

static constant constexpr int THREADGROUP_SIZE = 2048;

// Load strided elements from global memory into shared memory.
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
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Load strided elements from global memory into shared memory with indices.
template<typename T, typename ArgReductionOp, uint BLOCKSIZE>
METAL_FUNC void load_from_global(
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides,
    constant size_t &el_to_sum_per_block,
    device const T *src,
    threadgroup T shared[BLOCKSIZE],
    threadgroup uint shared_indices[BLOCKSIZE],
    uint tid [[ thread_index_in_threadgroup ]],
    uint dst_id [[ threadgroup_position_in_grid ]],
    uint block_dim [[ threads_per_threadgroup ]]
) {
    ArgReductionOp op;
    bool notset = true;

    size_t start_idx = dst_id * el_to_sum_per_block;
    size_t stop_idx = start_idx + el_to_sum_per_block;
    size_t idx = start_idx + tid;
    while (idx < stop_idx) {
        size_t strided_i = get_strided_index(idx, num_dims, dims, strides);
        if (notset || op(src[strided_i], shared[tid])) {
            shared[tid] = src[strided_i];
            // Assume that the reduction takes place over the last dimension which is contiguous.
            shared_indices[tid] = idx % dims[num_dims - 1];
            notset = false;
        }
        idx += block_dim;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Load contiguous elements from global memory into shared memory.
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
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Load contiguous elements from global memory into shared memory with indices.
template<typename T, typename ArgReductionOp, uint BLOCKSIZE>
METAL_FUNC void load_from_global(
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t &el_to_sum_per_block,
    device const T *src,
    threadgroup T shared[BLOCKSIZE],
    threadgroup uint shared_indices[BLOCKSIZE],
    uint tid [[ thread_index_in_threadgroup ]],
    uint dst_id [[ threadgroup_position_in_grid ]],
    uint block_dim [[ threads_per_threadgroup ]]
) {
    ArgReductionOp op;
    bool notset = true;

    size_t start_idx = dst_id * el_to_sum_per_block;
    size_t stop_idx = start_idx + el_to_sum_per_block;
    size_t idx = start_idx + tid;
    while (idx < stop_idx) {
        if (notset || op(src[idx], shared[tid])) {
            shared[tid] = src[idx];
            // Assume that the reduction takes place over the last dimension which is contiguous.
            shared_indices[tid] = idx % dims[num_dims - 1];
            notset = false;
        }
        idx += block_dim;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

#define reduce_threadgroup(SIZE)                                \
if (BLOCKSIZE >= SIZE) {                                        \
    if (block_dim >= SIZE) {                                    \
        shared[tid] = op(shared[tid], shared[tid + SIZE / 2]);  \
    }                                                           \
    threadgroup_barrier(mem_flags::mem_threadgroup);            \
}

template<typename T, typename ReductionOp, uint BLOCKSIZE>
METAL_FUNC void threadgroup_reduce(
    threadgroup T shared[BLOCKSIZE],
    uint tid [[thread_index_in_threadgroup]],
    uint block_dim [[ threads_per_threadgroup ]]
) {
    ReductionOp op;
    reduce_threadgroup(64);
    reduce_threadgroup(32);
    reduce_threadgroup(16);
    reduce_threadgroup(8);
    reduce_threadgroup(4);
    reduce_threadgroup(2);
}
#undef reduce_threadgroup

#define arg_reduce_threadgroup(SIZE)                            \
if (BLOCKSIZE >= SIZE) {                                        \
    if (block_dim >= SIZE &&                                    \
        op(shared[tid], shared[tid + SIZE / 2])                 \
    ) {                                                         \
        shared_indices[tid] = shared_indices[tid + SIZE / 2];   \
        shared[tid] = shared[tid + SIZE / 2];                   \
    }                                                           \
    threadgroup_barrier(mem_flags::mem_threadgroup);            \
}

template<typename T, typename ArgReductionOp, uint BLOCKSIZE>
METAL_FUNC void threadgroup_reduce(
    threadgroup T shared[BLOCKSIZE],
    threadgroup uint shared_indices[BLOCKSIZE],
    uint tid [[thread_index_in_threadgroup]],
    uint block_dim [[ threads_per_threadgroup ]]
) {
    ArgReductionOp op;
    arg_reduce_threadgroup(64);
    arg_reduce_threadgroup(32);
    arg_reduce_threadgroup(16);
    arg_reduce_threadgroup(8);
    arg_reduce_threadgroup(4);
    arg_reduce_threadgroup(2);
}
#undef arg_reduce_threadgroup

#define reduce_block(SIZE)                                      \
if (BLOCKSIZE >= SIZE) {                                        \
    if (tid < SIZE / 2 && block_dim >= SIZE) {                  \
        shared[tid] = op(shared[tid], shared[tid + SIZE / 2]);  \
    }                                                           \
    threadgroup_barrier(mem_flags::mem_threadgroup);            \
}                                                               \

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

    reduce_block(1024);
    reduce_block(512);
    reduce_block(256);
    reduce_block(128);

    if (tid < 32) {
        threadgroup_reduce<T, ReductionOp, BLOCKSIZE>(shared, tid, block_dim);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        dst[dst_id] = shared[tid];
    }
}
#undef reduce_block

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

#define arg_reduce_block(SIZE)                                  \
if (BLOCKSIZE >= SIZE) {                                        \
    if (tid < SIZE / 2                                          \
        && block_dim >= SIZE                                    \
        && arg_op(shared[tid], shared[tid + SIZE / 2])          \
    ) {                                                         \
        shared_indices[tid] = shared_indices[tid + SIZE / 2];   \
        shared[tid] = shared[tid + SIZE / 2];                   \
    }                                                           \
    threadgroup_barrier(mem_flags::mem_threadgroup);            \
}                                                               \

template<
    typename T,
    typename ArgReductionOp,
    uint BLOCKSIZE,
    bool STRIDED
>
METAL_FUNC void arg_block_reduce(
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides,
    constant size_t &el_to_sum_per_block,
    device const T *src,
    device uint *dst,
    threadgroup T shared[BLOCKSIZE],
    threadgroup uint shared_indices[BLOCKSIZE],
    uint id [[ thread_position_in_grid ]],
    uint tid [[ thread_index_in_threadgroup ]],
    uint dst_id [[ threadgroup_position_in_grid ]],
    uint block_dim [[ threads_per_threadgroup ]]
) {
    ArgReductionOp arg_op;

    shared[tid] = ArgReductionOp::init;
    shared_indices[tid] = numeric_limits<uint>::max();

    if (STRIDED) {
        load_from_global<T, ArgReductionOp, BLOCKSIZE>(
            num_dims,
            dims,
            strides,
            el_to_sum_per_block,
            src,
            shared,
            shared_indices,
            tid,
            dst_id,
            block_dim
        );
    } else {
        load_from_global<T, ArgReductionOp, BLOCKSIZE>(
            num_dims,
            dims,
            el_to_sum_per_block,
            src,
            shared,
            shared_indices,
            tid,
            dst_id,
            block_dim
        );
    }
    arg_reduce_block(1024);
    arg_reduce_block(512);
    arg_reduce_block(256);
    arg_reduce_block(128);

    if (tid < 32) {
        threadgroup_reduce<T, ArgReductionOp, BLOCKSIZE>(shared, shared_indices, tid, block_dim);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        dst[dst_id] = shared_indices[0];
    }
}
#undef arg_reduce_block

#define ARG_REDUCE(OP, NAME, T)                             \
kernel void NAME(                                           \
    constant size_t &num_dims,                              \
    constant size_t *dims,                                  \
    constant size_t *strides,                               \
    constant size_t &el_to_sum_per_block,                   \
    device const T *src,                                    \
    device uint *dst,                                       \
    uint id [[ thread_position_in_grid ]],                  \
    uint tid [[ thread_index_in_threadgroup ]],             \
    uint dst_id [[ threadgroup_position_in_grid ]],         \
    uint block_dim [[ threads_per_threadgroup ]]            \
) {                                                         \
    threadgroup T shared[BLOCKSIZE];                        \
    threadgroup uint shared_indices[BLOCKSIZE];             \
    arg_block_reduce<T, OP<T, bool>, BLOCKSIZE, false>(     \
        num_dims,                                           \
        dims,                                               \
        strides,                                            \
        el_to_sum_per_block,                                \
        src,                                                \
        dst,                                                \
        shared,                                             \
        shared_indices,                                     \
        id,                                                 \
        tid,                                                \
        dst_id,                                             \
        block_dim);                                         \
}                                                           \
kernel void NAME##_strided(                                 \
    constant size_t &num_dims,                              \
    constant size_t *dims,                                  \
    constant size_t *strides,                               \
    constant size_t &el_to_sum_per_block,                   \
    device const T *src,                                    \
    device uint *dst,                                       \
    uint id [[ thread_position_in_grid ]],                  \
    uint tid [[ thread_index_in_threadgroup ]],             \
    uint dst_id [[ threadgroup_position_in_grid ]],         \
    uint block_dim [[ threads_per_threadgroup ]]            \
) {                                                         \
    threadgroup T shared[BLOCKSIZE];                        \
    threadgroup uint shared_indices[BLOCKSIZE];             \
    arg_block_reduce<T, OP<T, bool>, BLOCKSIZE, true>(      \
        num_dims,                                           \
        dims,                                               \
        strides,                                            \
        el_to_sum_per_block,                                \
        src,                                                \
        dst,                                                \
        shared,                                             \
        shared_indices,                                     \
        id,                                                 \
        tid,                                                \
        dst_id,                                             \
        block_dim);                                         \
}


#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))


#define softmax_max_block(SIZE)                                     \
if (BLOCKSIZE >= SIZE) {                                            \
    if (tid < SIZE / 2 && block_dim >= SIZE) {                      \
        shared[tid] = max_op(shared[tid], shared[tid + SIZE / 2]);  \
    }                                                               \
    threadgroup_barrier(mem_flags::mem_threadgroup);                \
}

#define softmax_acc_block(SIZE)                                     \
if (BLOCKSIZE >= SIZE) {                                            \
    if (tid < SIZE / 2 && block_dim >= SIZE) {                      \
        shared[tid] += shared[tid + SIZE / 2];                      \
    }                                                               \
    threadgroup_barrier(mem_flags::mem_threadgroup);                \
}

template<
    typename T,
    typename ACC,
    uint BLOCKSIZE
>
METAL_FUNC void softmax(
    constant size_t &src_numel,
    constant size_t &el_to_sum_per_block,
    device const T *src,
    device T *dst,
    threadgroup ACC shared[BLOCKSIZE],

    uint id [[ thread_position_in_grid ]],
    uint tid [[ thread_index_in_threadgroup ]],
    uint dst_id [[ threadgroup_position_in_grid ]],
    uint block_dim [[ threads_per_threadgroup ]]
) {
    Max<ACC> max_op;

    shared[tid] = numeric_limits<ACC>::min();
    ACC tmp = numeric_limits<ACC>::min();

    size_t start_idx = dst_id * el_to_sum_per_block;
    size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);
    size_t idx = start_idx + tid;

    while (idx < stop_idx) {
        tmp = max_op(tmp, static_cast<ACC>(src[idx]));
        idx += block_dim;
    }
    shared[tid] = tmp;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    softmax_max_block(1024);
    softmax_max_block(512);
    softmax_max_block(256);
    softmax_max_block(128);
    if (tid < 32) {
        threadgroup_reduce<ACC, Max<ACC>, BLOCKSIZE>(shared, tid, block_dim);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ACC _max = shared[0];

    // prevent tid 0 from overwriting _max before other threads have written
    threadgroup_barrier(mem_flags::mem_threadgroup);
    shared[tid] = 0;

    idx = start_idx + tid;
    while (idx < stop_idx) {
        const ACC val = exp(static_cast<ACC>(src[idx]) - _max);
        dst[idx] = static_cast<T>(val);
        shared[tid] += val;

        idx += block_dim;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    softmax_acc_block(1024);
    softmax_acc_block(512);
    softmax_acc_block(256);
    softmax_acc_block(128);
    if (tid < 32) {
        threadgroup_reduce<ACC, Sum<ACC>, BLOCKSIZE>(shared, tid, block_dim);
        threadgroup_barrier(mem_flags::mem_none);
    }

    const T inv_acc = T(1.0/shared[0]);
    idx = start_idx + tid;
    while (idx < stop_idx) {
        dst[idx] *= inv_acc;
        idx += block_dim;
    }
}


#define SOFTMAX(NAME, T, ACC)                       \
kernel void NAME(                                   \
    constant size_t &src_numel,                     \
    constant size_t &el_to_sum_per_block,           \
    device const T *src,                            \
    device T *dst,                                  \
                                                    \
    uint id [[ thread_position_in_grid ]],          \
    uint tid [[ thread_index_in_threadgroup ]],     \
    uint dst_id [[ threadgroup_position_in_grid ]], \
    uint block_dim [[ threads_per_threadgroup ]]    \
) {                                                 \
    threadgroup ACC shared_memory[BLOCKSIZE];       \
    softmax<T, ACC, BLOCKSIZE>(                     \
        src_numel,                                  \
        el_to_sum_per_block,                        \
        src,                                        \
        dst,                                        \
        shared_memory,                              \
        id,                                         \
        tid,                                        \
        dst_id,                                     \
        block_dim);                                 \
}

REDUCE(Sum, fast_sum_f32, float)
REDUCE(Sum, fast_sum_u32, uint)
REDUCE(Sum, fast_sum_f16, half)
REDUCE(Sum, fast_sum_u8, uint8_t)

REDUCE(Mul, fast_mul_f32, float)
REDUCE(Mul, fast_mul_u32, uint)
REDUCE(Mul, fast_mul_f16, half)
REDUCE(Mul, fast_mul_u8, uint8_t)

REDUCE(Max, fast_max_f32, float)
REDUCE(Max, fast_max_u32, uint)
REDUCE(Max, fast_max_f16, half)
REDUCE(Max, fast_max_u8, uint8_t)

REDUCE(Min, fast_min_f32, float)
REDUCE(Min, fast_min_u32, uint)
REDUCE(Min, fast_min_f16, half)
REDUCE(Min, fast_min_u8, uint8_t)

ARG_REDUCE(ArgMin, fast_argmin_f32, float)
ARG_REDUCE(ArgMin, fast_argmin_f16, half)
ARG_REDUCE(ArgMin, fast_argmin_u32, uint)
ARG_REDUCE(ArgMin, fast_argmin_u8, uint8_t)

ARG_REDUCE(ArgMax, fast_argmax_f32, float)
ARG_REDUCE(ArgMax, fast_argmax_f16, half)
ARG_REDUCE(ArgMax, fast_argmax_u32, uint)
ARG_REDUCE(ArgMax, fast_argmax_u8, uint8_t)

SOFTMAX(softmax_f32, float, float)
SOFTMAX(softmax_f16, half, float)

#if __METAL_VERSION__ >= 220
REDUCE(Sum, fast_sum_i64, int64_t)
REDUCE(Mul, fast_mul_i64, int64_t)
REDUCE(Min, fast_min_i64, int64_t)
REDUCE(Max, fast_max_i64, int64_t)

ARG_REDUCE(ArgMin, fast_argmin_i64, int64_t)
ARG_REDUCE(ArgMax, fast_argmax_i64, int64_t)

#endif

#if __METAL_VERSION__ >= 310
REDUCE(Sum, fast_sum_bf16, bfloat)
REDUCE(Mul, fast_mul_bf16, bfloat)
REDUCE(Max, fast_max_bf16, bfloat)
REDUCE(Min, fast_min_bf16, bfloat)

ARG_REDUCE(ArgMin, fast_argmin_bf16, bfloat)
ARG_REDUCE(ArgMax, fast_argmax_bf16, bfloat)

SOFTMAX(softmax_bf16, bfloat, float)
#endif
