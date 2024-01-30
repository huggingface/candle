#include <metal_stdlib>
#include <metal_limits>
using namespace metal;

// TODO: Load multiple values per thread to improve memory bandwidth utilization
// static constant constexpr uint VALUES_PER_THREAD = 1;

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

template <typename V>
struct Indexed {
    uint i;
    V val;
    typedef V type;

    constexpr Indexed<V>() thread = default;
    constexpr Indexed<V>() threadgroup = default;
    constexpr Indexed<V>() device = default;
    constexpr Indexed<V>() constant = default;

    constexpr Indexed<V>(uint _i, V _val) : i(_i), val(_val) {}

    template <typename U, typename = typename enable_if<is_convertible_v<U, V>>::type>
    constexpr Indexed<V>(uint _i, U _val) : i(_i), val(static_cast<U>(_val)) {}

    template <typename U>
    constexpr Indexed<V>(const thread Indexed<U> &iv): Indexed<V>(iv.i, iv.val) {}

    template <typename U>
    constexpr Indexed<V>(const threadgroup Indexed<V> &iv): Indexed<V>(iv.i, iv.val) {}

    Indexed<V> operator=(const thread Indexed<V> &iv) thread {
        this->i = iv.i;
        this->val = iv.val;
        return *this;
    }
    Indexed<V> operator=(const thread Indexed<V> &iv) threadgroup {
        this->i = iv.i;
        this->val = iv.val;
        return *this;
    }
};

template<typename V>
constexpr METAL_FUNC bool operator<(Indexed<V> lhs, Indexed<V> rhs) {
    return lhs.val < rhs.val || (lhs.val == rhs.val && lhs.i < rhs.i);
}

template<typename V>
constexpr METAL_FUNC bool operator>(Indexed<V> lhs, Indexed<V> rhs) {
    return lhs.val > rhs.val || (lhs.val == rhs.val && lhs.i > rhs.i);
}

template<typename T>
struct _numeric_limits_impl<Indexed<T>> {
    static constexpr Indexed<T> lowest() {
        return Indexed<T>(0, numeric_limits<T>::lowest());
    }

    static constexpr Indexed<T> max() {
        return Indexed<T>(0, numeric_limits<T>::max());
    }
};

#if defined(__HAVE_BFLOAT__)
// Metal does not have simd_shuffle_down for bfloat16
// TODO: Check if volatile threadgroup memory reduction is faster than simd_shuffle_down for bfloat
bfloat simd_shuffle_down(bfloat value, ushort delta) {
    return static_cast<bfloat>(__metal_simd_shuffle_down(static_cast<float>(value), delta));
}
#endif

template <typename V>
Indexed<V> simd_shuffle_down(Indexed<V> iv, ushort delta) {
    return Indexed<V>(
        simd_shuffle_down(iv.i, delta),
        simd_shuffle_down(iv.val, delta)
    );
}

#define impl_reduction_op_helper(name, op, init_val, __result_type__)  \
template<typename T, typename R = __result_type__>                                  \
struct name {                                                                       \
    static constexpr T init() {                                                     \
        return init_val;                                                            \
    }                                                                               \
    METAL_FUNC R operator()(T a, T b) {                                             \
        return op;                                                                  \
    }                                                                               \
    METAL_FUNC R operator()(thread const T& a, thread const T& b) const {           \
        return op;                                                                  \
    }                                                                               \
    METAL_FUNC R operator()(threadgroup const T& a, threadgroup const T& b) const { \
        return op;                                                                  \
    }                                                                               \
}                                                                                   \

#define impl_reduction_op(name, op, init_val) \
impl_reduction_op_helper(name, op, init_val, T);

#define impl_arg_reduction_op(name, op, init_val) \
impl_reduction_op_helper(name, op, init_val, tuple<bool, Indexed<T>>);

impl_reduction_op(Sum, a + b, 0);
impl_reduction_op(Mul, a * b, 1);
impl_reduction_op(Min, a < b ? a : b, numeric_limits<T>::max());
impl_reduction_op(Max, a > b ? a : b, numeric_limits<T>::lowest());
#undef impl_reduction_op

// These are used when loading elements from global memory into shared memory.
// They let us use the same code for both indexed and non-indexed types.
template<typename Op, typename T, typename U>
METAL_FUNC T apply_operator(Op op, size_t _idx, T a, U b) {
    return op(a, static_cast<T>(b));
}

template<typename Op, typename T, typename U>
METAL_FUNC Indexed<T> apply_operator(Op op, size_t idx, Indexed<T> a, U b) {
    return op(a, Indexed<T>(idx, b));
}

// Load elements from global memory into shared memory.
// Handles both indexed and non-indexed types by using apply_operator.
template<
    typename T,
    typename R,
    typename ReductionOp,
    ushort BLOCKSIZE,
    bool STRIDED = false
>
METAL_FUNC R load_from_global(
    R value,
    constant size_t &num_elements,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides,
    constant size_t &el_to_sum_per_block,
    const device T *src,
    const ushort offset,
    threadgroup R shared[BLOCKSIZE],
    const ushort tid
) {
    ReductionOp op;

    size_t stop_idx = offset + el_to_sum_per_block;
    size_t idx = offset + tid;

    while (idx < stop_idx) {
        if (STRIDED) {
            idx = get_strided_index(idx, num_dims, dims, strides);
        }
        value = apply_operator(op, idx, value, src[idx]);
        idx += BLOCKSIZE;
    }
    return value;
}


// Convenience function for when we don't need to sum over multiple dimensions.
template<
    typename T,
    typename R,
    typename ReductionOp,
    ushort BLOCKSIZE
>
METAL_FUNC R load_from_global(
    R value,
    constant size_t &num_elements,
    constant size_t &el_to_sum_per_block,
    const device T *src,
    const size_t offset,
    threadgroup R shared[BLOCKSIZE],
    const ushort tid
) {
    return load_from_global<T, R, ReductionOp, BLOCKSIZE, false>(
        value,
        num_elements,
        // Dummy values for num_dims, dims, and strides
        num_elements,
        nullptr,
        nullptr,
        // end dummy values
        el_to_sum_per_block,
        src,
        offset,
        shared,
        tid
    );
}

// Since we are using simd_shuffle_down with a BLOCKSIZE guard we don't need any barriers.
template<typename ReductionOp, ushort BLOCKSIZE, typename T>
METAL_FUNC T simdgroup_reduce(T value) {
    ReductionOp op;
    if (BLOCKSIZE >= 32) value = op(value, simd_shuffle_down(value, 16));
    if (BLOCKSIZE >= 16) value = op(value, simd_shuffle_down(value,  8));
    if (BLOCKSIZE >=  8) value = op(value, simd_shuffle_down(value,  4));
    if (BLOCKSIZE >=  4) value = op(value, simd_shuffle_down(value,  2));
    if (BLOCKSIZE >=  2) value = op(value, simd_shuffle_down(value,  1));
    return value;
}

template<
   typename ReductionOp,
   ushort BLOCKSIZE,
   typename T
>
METAL_FUNC T threadgroup_reduce(
    threadgroup T shared[BLOCKSIZE],
    ushort tid [[ thread_index_in_threadgroup ]]
) {
    ReductionOp op;

    // Fully unrolled reduction loop from BLOCKSIZE down to 64.
    #pragma clang loop unroll(full)
    for (uint s = BLOCKSIZE / 2; s >= 64; s >>= 1) {
        if (tid < s) {
            shared[tid] = op(shared[tid], shared[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_none);
    }

    if (tid < 32) {
        // Last shared memory reduce can be done without tid < s check.
        if (BLOCKSIZE >= 64) {
            shared[tid] = op(shared[tid], shared[tid + 32]);
            simdgroup_barrier(mem_flags::mem_none);
        }
        // Remaining 32 threads can be reduced with simdgroup_reduce.
        shared[tid] = simdgroup_reduce<ReductionOp, BLOCKSIZE>(shared[tid]);
    }

    return shared[tid];
}

// Inspired by "Optimizing Parallel Reduction in CUDA" by Mark Harris
template<
    typename T,
    typename R,
    typename ReductionOp,
    ushort BLOCKSIZE,
    bool STRIDED = false
>
METAL_FUNC void reduce(
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides,
    constant size_t &el_to_sum_per_block,
    device const T *src,
    device R *dst,
    constant size_t &num_elements,
    threadgroup T shared[BLOCKSIZE],
    ushort tid [[ thread_index_in_threadgroup ]],
    ushort dst_id [[ threadgroup_position_in_grid ]]
) {
    // Initialize shared memory for current thread to correct value for reduction operation
    shared[tid] = ReductionOp::init();

    // Calcluate offset for the threadgroup of current thread
    ushort offset = dst_id * el_to_sum_per_block;
    R initial = ReductionOp::init();
    // Load with reduction from global memory into shared memory
    shared[tid] = load_from_global<T, R, ReductionOp, BLOCKSIZE, STRIDED>(
        initial,
        num_elements,
        num_dims,
        dims,
        strides,
        el_to_sum_per_block,
        src,
        offset,
        shared,
        tid
    );
    // Threadgroup barrier is needed to ensure that all threads have written to shared memory
    // Memory space is not shared between threadgroups so we can use the mem_none flag for all threadgroup barriers.
    threadgroup_barrier(mem_flags::mem_none);

    // Complete reduction
    R value = threadgroup_reduce<ReductionOp, BLOCKSIZE>(shared, tid);

    if (tid == 0) dst[dst_id] = value;
}


#define reduce_case(OP, T, R, N)                        \
case N: {                                               \
    threadgroup R shared[N];                            \
    reduce<T, R, OP<R>, N, STRIDED>(                    \
        num_dims,                                       \
        dims,                                           \
        strides,                                        \
        el_to_sum_per_block,                            \
        src,                                            \
        dst,                                            \
        num_elements,                                   \
        shared,                                         \
        tid,                                            \
        dst_id);                                        \
    break;                                              \
}

#define impl_reduce(OP, NAME, T)                        \
kernel void NAME(                                       \
    constant size_t &num_dims,                          \
    constant size_t &el_to_sum_per_block,               \
    device const T *src,                                \
    device T *dst,                                      \
    constant size_t &num_elements,                      \
    ushort tid [[ thread_index_in_threadgroup ]],       \
    ushort dst_id [[ threadgroup_position_in_grid ]],   \
    ushort block_dim [[ threads_per_threadgroup ]]      \
) {                                                     \
    constant size_t *dims = {};                         \
    constant size_t *strides = {};                      \
    const bool STRIDED = false;                         \
    switch (block_dim) {                                \
        reduce_case(OP, T, T, 2048);                    \
        reduce_case(OP, T, T, 1024);                    \
        reduce_case(OP, T, T, 512);                     \
        reduce_case(OP, T, T, 256);                     \
        reduce_case(OP, T, T, 128);                     \
        reduce_case(OP, T, T, 64);                      \
        reduce_case(OP, T, T, 32);                      \
        reduce_case(OP, T, T, 16);                      \
        reduce_case(OP, T, T, 8);                       \
        reduce_case(OP, T, T, 4);                       \
        reduce_case(OP, T, T, 2);                       \
        reduce_case(OP, T, T, 1);                       \
    }                                                   \
}                                                       \
kernel void NAME##_strided(                             \
    constant size_t &num_dims,                          \
    constant size_t *dims,                              \
    constant size_t *strides,                           \
    constant size_t &el_to_sum_per_block,               \
    device const T *src,                                \
    device T *dst,                                      \
    constant size_t &num_elements,                      \
    ushort tid [[ thread_index_in_threadgroup ]],       \
    ushort dst_id [[ threadgroup_position_in_grid ]],   \
    ushort block_dim [[ threads_per_threadgroup ]]      \
) {                                                     \
    const bool STRIDED = true;                          \
    switch (block_dim) {                                \
        reduce_case(OP, T, T, 2048);                    \
        reduce_case(OP, T, T, 1024);                    \
        reduce_case(OP, T, T, 512);                     \
        reduce_case(OP, T, T, 256);                     \
        reduce_case(OP, T, T, 128);                     \
        reduce_case(OP, T, T, 64);                      \
        reduce_case(OP, T, T, 32);                      \
        reduce_case(OP, T, T, 16);                      \
        reduce_case(OP, T, T, 8);                       \
        reduce_case(OP, T, T, 4);                       \
        reduce_case(OP, T, T, 2);                       \
        reduce_case(OP, T, T, 1);                       \
    }                                                   \
}

template<
    typename T,
    typename ReductionOp,
    ushort BLOCKSIZE,
    bool STRIDED
>
METAL_FUNC void reduce(
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides,
    constant size_t &el_to_sum_per_block,
    device const T *src,
    device uint *dst,
    constant size_t &num_elements,
    threadgroup Indexed<T> shared[BLOCKSIZE],
    ushort tid [[ thread_index_in_threadgroup ]],
    ushort dst_id [[ threadgroup_position_in_grid ]]
) {
    // Initialize shared memory for current thread to correct value for reduction operation
    shared[tid] = ReductionOp::init();

    // Calcluate offset for the threadgroup of current thread
    ushort offset = dst_id * el_to_sum_per_block;
    Indexed<T> initial = ReductionOp::init();
    // Load with reduction from global memory into shared memory
    shared[tid] = load_from_global<T, Indexed<T>, ReductionOp, BLOCKSIZE, STRIDED>(
        initial,
        num_elements,
        num_dims,
        dims,
        strides,
        el_to_sum_per_block,
        src,
        offset,
        shared,
        tid
    );
    // Threadgroup barrier is needed to ensure that all threads have written to shared memory
    // Memory space is not shared between threadgroups so we can use the mem_none flag for all threadgroup barriers.
    threadgroup_barrier(mem_flags::mem_none);

    // Complete reduction
    Indexed<T> value = threadgroup_reduce<ReductionOp, BLOCKSIZE, Indexed<T>>(shared, tid);

    // Return index of reduce result
    if (tid == 0) dst[dst_id] = value.i;
}

#define arg_reduce_case(OP, T, N)                       \
case N: {                                               \
    threadgroup Indexed<T> shared[N];                   \
    reduce<T, OP<Indexed<T>>, N, STRIDED>(              \
        num_dims,                                       \
        dims,                                           \
        strides,                                        \
        el_to_sum_per_block,                            \
        src,                                            \
        dst,                                            \
        num_elements,                                   \
        shared,                                         \
        tid,                                            \
        dst_id);                                        \
    break;                                              \
}

#define impl_arg_reduce(OP, NAME, T)                    \
kernel void NAME(                                       \
    constant size_t &num_dims,                          \
    constant size_t &el_to_sum_per_block,               \
    device const T *src,                                \
    device uint *dst,                                   \
    constant size_t &num_elements,                      \
    ushort tid [[ thread_index_in_threadgroup ]],       \
    ushort dst_id [[ threadgroup_position_in_grid ]],   \
    ushort block_dim [[ threads_per_threadgroup ]]      \
) {                                                     \
    constant size_t *dims = {};                         \
    constant size_t *strides = {};                      \
    const bool STRIDED = false;                         \
    switch (block_dim) {                                \
        arg_reduce_case(OP, T, 2048);                   \
        arg_reduce_case(OP, T, 1024);                   \
        arg_reduce_case(OP, T, 512);                    \
        arg_reduce_case(OP, T, 256);                    \
        arg_reduce_case(OP, T, 128);                    \
        arg_reduce_case(OP, T, 64);                     \
        arg_reduce_case(OP, T, 32);                     \
        arg_reduce_case(OP, T, 16);                     \
        arg_reduce_case(OP, T, 8);                      \
        arg_reduce_case(OP, T, 4);                      \
        arg_reduce_case(OP, T, 2);                      \
        arg_reduce_case(OP, T, 1);                      \
    }                                                   \
}                                                       \
kernel void NAME##_strided(                             \
    constant size_t &num_dims,                          \
    constant size_t *dims,                              \
    constant size_t *strides,                           \
    constant size_t &el_to_sum_per_block,               \
    device const T *src,                                \
    device uint *dst,                                   \
    constant size_t &num_elements,                      \
    ushort tid [[ thread_index_in_threadgroup ]],       \
    ushort dst_id [[ threadgroup_position_in_grid ]],   \
    ushort block_dim [[ threads_per_threadgroup ]]      \
) {                                                     \
    const bool STRIDED = true;                          \
    switch (block_dim) {                                \
        arg_reduce_case(OP, T, 2048);                   \
        arg_reduce_case(OP, T, 1024);                   \
        arg_reduce_case(OP, T, 512);                    \
        arg_reduce_case(OP, T, 256);                    \
        arg_reduce_case(OP, T, 128);                    \
        arg_reduce_case(OP, T, 64);                     \
        arg_reduce_case(OP, T, 32);                     \
        arg_reduce_case(OP, T, 16);                     \
        arg_reduce_case(OP, T, 8);                      \
        arg_reduce_case(OP, T, 4);                      \
        arg_reduce_case(OP, T, 2);                      \
        arg_reduce_case(OP, T, 1);                      \
    }                                                   \
}

template<
    typename T,
    typename ACC = float,
    ushort BLOCKSIZE
>
METAL_FUNC void softmax(
    constant size_t &src_numel,
    constant size_t &el_to_sum_per_block,
    const device T *src,
    device T *dst,
    threadgroup ACC shared[BLOCKSIZE],

    ushort tid [[ thread_index_in_threadgroup ]],
    ushort dst_id [[ threadgroup_position_in_grid ]]
) {
    // Initialize shared memory for current thread to lowest value
    shared[tid] = numeric_limits<ACC>::lowest();

    // Calcluate offset for the threadgroup of current thread
    size_t offset = dst_id * el_to_sum_per_block;
    ACC initial = numeric_limits<ACC>::lowest();
    // Load with reduction from global memory into shared memory
    shared[tid] = load_from_global<T, ACC, Max<ACC>, BLOCKSIZE>(
        initial,
        src_numel,
        el_to_sum_per_block,
        src,
        offset,
        shared,
        tid
    );
    // Threadgroup barrier is needed to ensure that all threads have written to shared memory
    // Memory space is not shared between threadgroups so we can use the mem_none flag for all threadgroup barriers.
    threadgroup_barrier(mem_flags::mem_none);

    // Reduce shared memory to find max value
    threadgroup_reduce<Max<ACC>, BLOCKSIZE>(shared, tid);
    ACC max_result = shared[0];

    // Ensure all threads have max_result = shared[0] before we set shared[0] = 0.
    threadgroup_barrier(mem_flags::mem_none);
    shared[tid] = 0;

    // Calculate softmax values
    size_t stop_idx = min(offset + el_to_sum_per_block, src_numel);
    size_t idx = offset + tid;
    while (idx < stop_idx) {
        const ACC val = exp(ACC(src[idx]) - max_result);
        dst[idx] = T(val);
        shared[tid] += val;
        idx += BLOCKSIZE;
    }
    threadgroup_barrier(mem_flags::mem_none);

    threadgroup_reduce<Sum<ACC>, BLOCKSIZE>(shared, tid);
    threadgroup_barrier(mem_flags::mem_none);

    const T inv_acc = T(1.0/shared[0]);
    idx = offset + tid;
    while (idx < stop_idx) {
        dst[idx] *= inv_acc;
        idx += BLOCKSIZE;
    }
}

#define softmax_case(T, ACC, N)                         \
case N: {                                               \
    threadgroup ACC shared[N];                          \
    softmax<T, ACC, N>(                                 \
        src_numel,                                      \
        el_to_sum_per_block,                            \
        src,                                            \
        dst,                                            \
        shared,                                         \
        tid,                                            \
        dst_id);                                        \
    break;                                              \
}

#define impl_softmax(NAME, T, ACC)                      \
kernel void NAME(                                       \
    constant size_t &src_numel,                         \
    constant size_t &el_to_sum_per_block,               \
    device const T *src,                                \
    device T *dst,                                      \
                                                        \
    ushort tid [[ thread_index_in_threadgroup ]],       \
    ushort dst_id [[ threadgroup_position_in_grid ]],   \
    ushort block_dim [[ threads_per_threadgroup ]]      \
) {                                                     \
    switch (block_dim) {                                \
        softmax_case(T, ACC, 2048);                     \
        softmax_case(T, ACC, 1024);                     \
        softmax_case(T, ACC, 512);                      \
        softmax_case(T, ACC, 256);                      \
        softmax_case(T, ACC, 128);                      \
        softmax_case(T, ACC, 64);                       \
        softmax_case(T, ACC, 32);                       \
        softmax_case(T, ACC, 16);                       \
        softmax_case(T, ACC, 8);                        \
        softmax_case(T, ACC, 4);                        \
        softmax_case(T, ACC, 2);                        \
        softmax_case(T, ACC, 1);                        \
    }                                                   \
}

impl_reduce(Sum, fast_sum_f32, float)
impl_reduce(Sum, fast_sum_u32, uint)
impl_reduce(Sum, fast_sum_f16, half)
impl_reduce(Sum, fast_sum_u8, uint8_t)

impl_reduce(Mul, fast_mul_f32, float)
impl_reduce(Mul, fast_mul_u32, uint)
impl_reduce(Mul, fast_mul_f16, half)
impl_reduce(Mul, fast_mul_u8, uint8_t)

impl_reduce(Max, fast_max_f32, float)
impl_reduce(Max, fast_max_u32, uint)
impl_reduce(Max, fast_max_f16, half)
impl_reduce(Max, fast_max_u8, uint8_t)

impl_reduce(Min, fast_min_f32, float)
impl_reduce(Min, fast_min_u32, uint)
impl_reduce(Min, fast_min_f16, half)
impl_reduce(Min, fast_min_u8, uint8_t)

impl_arg_reduce(Min, fast_argmin_f32, float)
impl_arg_reduce(Min, fast_argmin_f16, half)
impl_arg_reduce(Min, fast_argmin_u32, uint)
impl_arg_reduce(Min, fast_argmin_u8, uint8_t)

impl_arg_reduce(Max, fast_argmax_f32, float)
impl_arg_reduce(Max, fast_argmax_f16, half)
impl_arg_reduce(Max, fast_argmax_u32, uint)
impl_arg_reduce(Max, fast_argmax_u8, uint8_t)

impl_softmax(softmax_f32, float, float)
impl_softmax(softmax_f16, half, float)

#if __METAL_VERSION__ >= 220
impl_reduce(Sum, fast_sum_i64, int64_t)
impl_reduce(Mul, fast_mul_i64, int64_t)
impl_reduce(Min, fast_min_i64, int64_t)
impl_reduce(Max, fast_max_i64, int64_t)

impl_arg_reduce(Min, fast_argmin_i64, int64_t)
impl_arg_reduce(Max, fast_argmax_i64, int64_t)
#endif

#if defined(__HAVE_BFLOAT__)
impl_reduce(Sum, fast_sum_bf16, bfloat)
impl_reduce(Mul, fast_mul_bf16, bfloat)
impl_reduce(Max, fast_max_bf16, bfloat)
impl_reduce(Min, fast_min_bf16, bfloat)

impl_arg_reduce(Min, fast_argmin_bf16, bfloat)
impl_arg_reduce(Max, fast_argmax_bf16, bfloat)

impl_softmax(softmax_bf16, bfloat, float)
#endif
