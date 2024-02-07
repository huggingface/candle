#include <metal_stdlib>
#include <metal_limits>
using namespace metal;

// Use this to get the scalar type of a vector type
template <typename T>
using Scalar = make_scalar_t<T>;

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

#pragma METAL internals : enable
// Vector helper traits
template <typename T, int N>
struct is_len : false_type {};

template <typename T, int N>
struct is_len<vec<T, N>, N>: true_type {};

template <typename T, int N>
constexpr constant int is_len_v = is_len<T, N>::value;

template <typename T, typename U>
struct _is_same_len : false_type {};

template <typename T, typename U, int N>
struct _is_same_len<vec<T, N>, vec<U, N>>: true_type {};

template <typename T, typename U>
struct is_same_len : _is_same_len<remove_cv_t<T>, remove_cv_t<U>> {};

template <typename T, typename U>
constexpr constant int is_same_len_v = _is_same_len<T, U>::value;
#pragma METAL internals : disable

// Keeps track of the index of the value in the reduction operation (argmin, argmax, etc.)
// and the value itself. The index is also used to break ties in the reduction operation.
// There are two specializations of the indexed class, one for scalar values and one for vector values.
template <typename T, typename = void>
class indexed;


template <typename T>
struct _make_scalar_impl<indexed<T>>
{
  typedef indexed<Scalar<T>> type;
};

// Specialization for scalar values
template <typename T>
class indexed<T, typename metal::enable_if_t<is_scalar_v<T>>> {
public:
    uint i;
    T val;

    constexpr indexed<T>() thread = default;
    constexpr indexed<T>() threadgroup = default;
    constexpr indexed<T>() device = default;
    constexpr indexed<T>() constant = default;

    constexpr indexed<T>(uint _i, T _val) : i(_i), val(_val) {}

    template <typename U>
    constexpr indexed<T>(const thread indexed<U> &iv): indexed<T>(iv.i, iv.val) {}

    constexpr indexed<T>(const threadgroup indexed<T> &iv): indexed<T>(iv.i, iv.val) {}

    constexpr indexed<T>(const volatile threadgroup indexed<T> &iv): indexed<T>(iv.i, iv.val) {}

    indexed<T> operator=(const thread indexed<T> &iv) thread {
        this->i = iv.i;
        this->val = iv.val;
        return *this;
    }

    indexed<T> operator=(const thread indexed<T> &iv) threadgroup {
        this->i = iv.i;
        this->val = iv.val;
        return *this;
    }

    // To align with below implementation.
    constexpr auto operator[](uint n) {
        assert(n == 0);
        return *this;
    }
};

// Specialization for vector values
template <typename T>
class indexed<T, typename metal::enable_if_t<is_vector_v<T>>> {
public:
    typedef vec<uint, vec_elements<T>::value> I;
    I i;
    T val;

    constexpr indexed<T>() thread = default;
    constexpr indexed<T>() threadgroup = default;
    constexpr indexed<T>() device = default;
    constexpr indexed<T>() constant = default;

    constexpr indexed<T>(I _i, T _val) : i(_i), val(_val) {}

    // Return 1-dimensional indexed value
    constexpr indexed<Scalar<T>> operator[](uint n) {
        assert(n < N);
        return indexed<Scalar<T>>(i[n], val[n]);
    }
    constexpr const indexed<Scalar<T>> operator[](uint n) const {
        assert(n < N);
        return indexed<Scalar<T>>(i[n], val[n]);
    }

    constexpr indexed<T>(const thread indexed<T> &iv): indexed<T>(iv.i, iv.val) {}

    constexpr indexed<T>(const threadgroup indexed<T> &iv): indexed<T>(iv.i, iv.val) {}

    constexpr indexed<T>(const volatile threadgroup indexed<T> &iv): indexed<T>(iv.i, iv.val) {}

    indexed<T> operator=(const thread indexed<T> &iv) thread {
        this->i = iv.i;
        this->val = iv.val;
        return *this;
    }

    indexed<T> operator=(const thread indexed<T> &iv) threadgroup {
        this->i = iv.i;
        this->val = iv.val;
        return *this;
    }
};

template<typename T, typename _E = typename metal::enable_if_t<is_scalar_v<T>>>
constexpr METAL_FUNC bool operator<(indexed<T> lhs, indexed<T> rhs) {
    return lhs.val < rhs.val || (lhs.val == rhs.val && lhs.i < rhs.i);
}

template<typename T, typename _E = typename metal::enable_if_t<is_scalar_v<T>>>
constexpr METAL_FUNC bool operator>(indexed<T> lhs, indexed<T> rhs) {
    return lhs.val > rhs.val || (lhs.val == rhs.val && lhs.i > rhs.i);
}

template<typename T>
struct _numeric_limits_impl<indexed<T>> {
    static constexpr METAL_FUNC indexed<T> lowest() {
        return indexed<T>(0, numeric_limits<T>::lowest());
    }

    static constexpr METAL_FUNC indexed<T> max() {
        return indexed<T>(0, numeric_limits<T>::max());
    }
};

#if defined(__HAVE_BFLOAT__)
// Metal does not have simd_shuffle_down for bfloat16
bfloat simd_shuffle_down(bfloat value, ushort delta) {
    return static_cast<bfloat>(simd_shuffle_down(static_cast<float>(value), delta));
}

template<uint N>
vec<bfloat, N> simd_shuffle_down(vec<bfloat, N> value, ushort delta) {
    return static_cast<vec<bfloat, N>>(simd_shuffle_down(static_cast<vec<float, N>>(value), delta));
}
#endif

template <typename T>
indexed<T> simd_shuffle_down(indexed<T> iv, ushort delta) {
    return indexed<T>(
        simd_shuffle_down(iv.i, delta),
        simd_shuffle_down(iv.val, delta)
    );
}

// Reduction base class.
// The reduction class is used to define the reduction operation for a given type.
// The reduction operation is defined by the operator() method, which is overloaded for both indexed and non-indexed types.
// Operation definitions are provided for default, thread, and threadgroup address spaces.
// Operations are defined as scalar, but is applied for both scalar and vector via apply_operation.
template <typename T, typename R = T, typename = void>
class reduction {
private:
    // Scalar representation of the input type.
    typedef Scalar<T> type;
    // Scalar representation of the result type.
    typedef Scalar<R> result_type;
public:
    // The initial value for the reduction operation. This returns R as opposed to result_type because it can
    // potentially be a vector type, so that it can be used for both scalar and vector types.
    static constexpr METAL_FUNC R init();
    // The reduction operation for default, thread, and threadgroup address spaces respectively.
    METAL_FUNC result_type operator()(type a, type b);
    METAL_FUNC result_type operator()(thread const type& a, thread const type& b) const;
    METAL_FUNC result_type operator()(threadgroup const type &a, threadgroup const type &b) const;
};

template<typename T, typename R = T>
struct Sum {
    typedef Scalar<T> type;
    typedef Scalar<R> result_type;

    static constexpr METAL_FUNC R init() {
        return 0;
    }
    METAL_FUNC result_type operator()(type a, type b) {
        return a + b;
    }
    METAL_FUNC result_type operator()(
        thread const type& a,
        thread const type& b
    ) const {
        return a + b;
    }
    METAL_FUNC result_type operator()(
        threadgroup const type &a,
        threadgroup const type &b
    ) const {
        return a + b;
    }
};


template<typename T, typename R = T>
struct Mul {
    typedef Scalar<T> type;
    typedef Scalar<R> result_type;

    static constexpr METAL_FUNC R init() {
        return 1;
    }
    METAL_FUNC result_type operator()(type a, type b) {
        return a * b;
    }
    METAL_FUNC result_type operator()(
        thread const type& a,
        thread const type& b
    ) const {
        return a * b;
    }
    METAL_FUNC result_type operator()(
        threadgroup const type &a,
        threadgroup const type &b
    ) const {
        return a * b;
    }
};

template<typename T, typename R = T>
struct Min {
    typedef Scalar<T> type;
    typedef Scalar<R> result_type;

    static constexpr METAL_FUNC R init() {
        return numeric_limits<R>::max();
    }
    METAL_FUNC result_type operator()(type a, type b) {
        return a < b ? a : b;
    }
    METAL_FUNC result_type operator()(
        thread const type& a,
        thread const type& b
    ) const {
        return a < b ? a : b;
    }
    METAL_FUNC result_type operator()(
        threadgroup const type &a,
        threadgroup const type &b
    ) const {
        return a < b ? a : b;
    }
};

template<typename T, typename R = T>
struct Max {
    typedef Scalar<T> type;
    typedef Scalar<R> result_type;

    static constexpr METAL_FUNC R init() {
        return numeric_limits<R>::lowest();
    }
    METAL_FUNC result_type operator()(type a, type b) {
        return a > b ? a : b;
    }
    METAL_FUNC result_type operator()(
        thread const type& a,
        thread const type& b
    ) const {
        return a > b ? a : b;
    }
    METAL_FUNC result_type operator()(
        threadgroup const type &a,
        threadgroup const type &b
    ) const {
        return a > b ? a : b;
    }
};


// Helper methods for applying operators to both indexed and non-indexed types.
//
// The apply_operator function is used to apply an operator to two values.
// It handles both indexed and non-indexed types by using the operator() method of the operator.
// The operator() method is overloaded for both indexed and non-indexed types.
// The operator() method for indexed types takes an index as an argument.
// The operator() method for non-indexed types does not take an index as an argument.

// Scalar values.
template<typename Op, typename T, typename _E = typename metal::enable_if<is_scalar_v<T>>::type>
METAL_FUNC T apply_operator(Op op, T a, T b) {
     return op(a, b);
}
template<typename Op, typename T, typename _E = typename metal::enable_if<is_scalar_v<T>>::type>
METAL_FUNC T apply_operator(Op op, T a, T b, size_t _idx) {
     return op(a, b);
}

// Convertible types.
template<typename Op, typename T, typename U, typename _E = typename metal::enable_if_t<is_convertible_v<U, T>>>
METAL_FUNC T apply_operator(Op op, T a, U b, size_t _idx) {
     return op(a, static_cast<T>(b));
}
template<typename Op, typename T, typename U, typename _E = typename metal::enable_if_t<is_convertible_v<U, T>>>
METAL_FUNC T apply_operator(Op op, T a, U b) {
     return op(a, static_cast<T>(b));
}

// indexed scalar and scalar.
template<typename Op, typename T, typename _E = typename metal::enable_if<is_scalar_v<T>>::type>
METAL_FUNC indexed<T> apply_operator(Op op, indexed<T> a, T b, size_t idx) {
    return op(a, indexed<T>(idx, b));
}

// Indexed vector and vector.
template<typename Op, typename T, uint N>
METAL_FUNC indexed<vec<T, N>> apply_operator(Op op, indexed<vec<T, N>> a, vec<T, N> b, size_t idx) {
    #pragma clang loop unroll(full)
    for (ushort n = 0; n < N; n++) {
        a[n] = op(a[n], indexed<Scalar<T>>(idx + n, b[n]));
    }
    return a;
}

// indexed vectors with same length and convertible types.
template<typename Op, typename T, typename U, typename _E = typename metal::enable_if_t<is_same_len_v<T, U>>>
METAL_FUNC indexed<T> apply_operator(Op op, indexed<T> a, indexed<U> b) {
    #pragma clang loop unroll(full)
    for (ushort n = 0; n < vec_elements<T>::value; n++) {
        a[n] = op(a[n], b[n]);
    }
    return a;
}

// Vectors with same length and convertible types.
template<typename Op, typename T, typename U, uint N>
METAL_FUNC vec<T, N> apply_operator(Op op, vec<T, N> a, vec<U, N> b) {
    #pragma clang loop unroll(full)
    for (ushort n = 0; n < N; n++) {
        a[n] = op(a[n], static_cast<T>(b[n]));
    }
    return a;
}

template<typename Op, typename T, typename U, uint N>
METAL_FUNC vec<T, N> apply_operator(Op op, vec<T, N> a, vec<U, N> b, size_t _idx) {
    #pragma clang loop unroll(full)
    for (ushort n = 0; n < N; n++) {
        a[n] = op(a[n], static_cast<T>(b[n]));
    }
    return a;
}

template<typename Op, typename T, typename _E = typename metal::enable_if_t<is_scalar_v<T>>>
METAL_FUNC uint finalize(indexed<T> value) {
    return value.i;
}

template<typename Op, typename T, uint N, typename _E = typename metal::enable_if_t<is_scalar_v<T>>>
METAL_FUNC uint finalize(indexed<vec<T, N>> value) {
    Op op;
    indexed<T> result = value[0];

    #pragma clang loop unroll(full)
    for (ushort n = 1; n < N; n++) {
        result = op(result, value[n]);
    }
    return result.i;
}

template<typename Op, typename T, typename _E = typename metal::enable_if_t<is_scalar_v<T>>>
METAL_FUNC T finalize(T value) {
    return value;
}

template<typename Op, typename T, typename _E = typename metal::enable_if_t<is_vector_v<T>>>
METAL_FUNC Scalar<T> finalize(T value) {
    Op op;
    Scalar<T> result = value[0];
    #pragma clang loop unroll(full)
    for (ushort n = 1; n < vec_elements<T>::value; n++) {
        result = op(result, value[n]);
    }
    return result;
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

    uint idx = offset + tid;
    #pragma clang loop unroll(full)
    for (uint i = offset + tid; i < offset + el_to_sum_per_block; i += BLOCKSIZE) {
        if (STRIDED) {
            idx = get_strided_index(i, num_dims, dims, strides);
            value = apply_operator(op, value, src[idx], idx);
        } else {
            value = apply_operator(op, value, src[i], i);
        }
    }
    return value;
}


// Convenience function for when we don't need to reduce over multiple dimensions.
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

// TODO: This allows int64_t simd_shuffle_down, but it is buggy so we should remove it.
//#if __METAL_VERSION__ >= 220
// Specialization for int64_t since it does not support simd_shuffle_down.
//template<typename ReductionOp, ushort BLOCKSIZE, typename T, typename _E = typename enable_if<!__is_valid_simdgroup_type<T>::value>::type>
//METAL_FUNC T simdgroup_reduce(
//    volatile threadgroup T shared[BLOCKSIZE],
//    const ushort tid
//) {
//    ReductionOp op;
//    T value = shared[tid];
//    if (BLOCKSIZE >= 32) value = apply_operator(op, shared[tid], shared[tid + 16]);
//    if (BLOCKSIZE >= 16) value = apply_operator(op, shared[tid], shared[tid +  8]);
//    if (BLOCKSIZE >=  8) value = apply_operator(op, shared[tid], shared[tid +  4]);
//    if (BLOCKSIZE >=  4) value = apply_operator(op, shared[tid], shared[tid +  2]);
//    if (BLOCKSIZE >=  2) value = apply_operator(op, shared[tid], shared[tid +  1]);
//    return value;
//}
//#endif

// Since we are using simd_shuffle_down with a BLOCKSIZE guard we don't need any barriers.
template<typename ReductionOp, ushort BLOCKSIZE, typename T>
METAL_FUNC T simdgroup_reduce(T value) {
    ReductionOp op;
    if (BLOCKSIZE >= 32) value = apply_operator(op, value, simd_shuffle_down(value, 16));
    if (BLOCKSIZE >= 16) value = apply_operator(op, value, simd_shuffle_down(value,  8));
    if (BLOCKSIZE >=  8) value = apply_operator(op, value, simd_shuffle_down(value,  4));
    if (BLOCKSIZE >=  4) value = apply_operator(op, value, simd_shuffle_down(value,  2));
    if (BLOCKSIZE >=  2) value = apply_operator(op, value, simd_shuffle_down(value,  1));
    return value;
}

// Since we are using simd_shuffle_down with a BLOCKSIZE guard we don't need any barriers.
template<typename ReductionOp, ushort BLOCKSIZE, typename T>
METAL_FUNC indexed<T> simdgroup_reduce(indexed<T> value) {
    ReductionOp op;
    if (BLOCKSIZE >= 32) value = apply_operator(op, value, simd_shuffle_down(value, 16));
    if (BLOCKSIZE >= 16) value = apply_operator(op, value, simd_shuffle_down(value,  8));
    if (BLOCKSIZE >=  8) value = apply_operator(op, value, simd_shuffle_down(value,  4));
    if (BLOCKSIZE >=  4) value = apply_operator(op, value, simd_shuffle_down(value,  2));
    if (BLOCKSIZE >=  2) value = apply_operator(op, value, simd_shuffle_down(value,  1));
    return value;
}

template<
   typename T,
   typename ReductionOp,
   ushort BLOCKSIZE
>
METAL_FUNC T threadgroup_reduce(
    threadgroup T shared[BLOCKSIZE],
    const uint offset,
    const ushort tid
) {
    ReductionOp op;

    // Fully unrolled reduction loop from BLOCKSIZE down to 64.
    #pragma clang loop unroll(full)
    for (ushort s = BLOCKSIZE / 2; s >= 64; s >>= 1) {
        if (tid < s) {
            shared[tid] = apply_operator(op, shared[tid], shared[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_none);
    }

    if (tid < 32) {
        // Last shared memory reduce can be done without tid < s check.
        if (BLOCKSIZE >= 64) {
            shared[tid] = apply_operator(op, shared[tid], shared[tid + 32]);
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
    device Scalar<R> *dst,
    constant size_t &num_elements,
    threadgroup R shared[BLOCKSIZE],
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
    R value = threadgroup_reduce<R, ReductionOp, BLOCKSIZE>(shared, offset, tid);

    if (tid == 0) dst[dst_id] = finalize<ReductionOp>(value);
}


#define reduce_case(OP, T, R, N)                        \
case N: {                                               \
    constexpr uint GRANULARITY = vec_elements<T>::value == 0 ? 1 : vec_elements<T>::value; \
    if (N / GRANULARITY == 0) {                         \
        break;                                          \
    }                                                   \
    constexpr uint B = N / GRANULARITY == 0 ? 1 : N / GRANULARITY;                 \
    threadgroup R shared[B];                            \
    reduce<T, R, OP<R>, B, STRIDED>(                    \
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


#define ARG(...) __VA_ARGS__

#define impl_reduce_inner(OP, NAME, T)                  \
kernel void NAME(                                       \
    constant size_t &num_dims,                          \
    constant size_t &el_to_sum_per_block,               \
    device const T *src,                                \
    device Scalar<T> *dst,                              \
    constant size_t &num_elements,                      \
    ushort tid [[ thread_index_in_threadgroup ]],       \
    ushort dst_id [[ threadgroup_position_in_grid ]],   \
    ushort block_dim [[ threads_per_threadgroup ]]      \
) {                                                     \
    constant size_t *dims = {};                         \
    constant size_t *strides = {};                      \
    const bool STRIDED = false;                         \
    switch (block_dim) {                                \
        reduce_case(OP, ARG(T), ARG(T), 1024);          \
        reduce_case(OP, ARG(T), ARG(T),  512);          \
        reduce_case(OP, ARG(T), ARG(T),  256);          \
        reduce_case(OP, ARG(T), ARG(T),  128);          \
        reduce_case(OP, ARG(T), ARG(T),   64);          \
        reduce_case(OP, ARG(T), ARG(T),   32);          \
        reduce_case(OP, ARG(T), ARG(T),   16);          \
        reduce_case(OP, ARG(T), ARG(T),    8);          \
        reduce_case(OP, ARG(T), ARG(T),    4);          \
        reduce_case(OP, ARG(T), ARG(T),    2);          \
        reduce_case(OP, ARG(T), ARG(T),    1);          \
    }                                                   \
}

#define impl_reduce_strided(OP, NAME, T, NAME_SUFFIX)   \
kernel void NAME##_strided##NAME_SUFFIX(                \
    constant size_t &num_dims,                          \
    constant size_t *dims,                              \
    constant size_t *strides,                           \
    constant size_t &el_to_sum_per_block,               \
    device const T *src,                                \
    device Scalar<T> *dst,                              \
    constant size_t &num_elements,                      \
    ushort tid [[ thread_index_in_threadgroup ]],       \
    ushort dst_id [[ threadgroup_position_in_grid ]],   \
    ushort block_dim [[ threads_per_threadgroup ]]      \
) {                                                     \
    const bool STRIDED = true;                          \
    switch (block_dim) {                                \
        reduce_case(OP, ARG(T), ARG(T), 1024);          \
        reduce_case(OP, ARG(T), ARG(T),  512);          \
        reduce_case(OP, ARG(T), ARG(T),  256);          \
        reduce_case(OP, ARG(T), ARG(T),  128);          \
        reduce_case(OP, ARG(T), ARG(T),   64);          \
        reduce_case(OP, ARG(T), ARG(T),   32);          \
        reduce_case(OP, ARG(T), ARG(T),   16);          \
        reduce_case(OP, ARG(T), ARG(T),    8);          \
        reduce_case(OP, ARG(T), ARG(T),    4);          \
        reduce_case(OP, ARG(T), ARG(T),    2);          \
        reduce_case(OP, ARG(T), ARG(T),    1);          \
    }                                                   \
}

#define impl_reduce(OP, NAME, T)                    \
impl_reduce_inner(OP, NAME, T)                      \
impl_reduce_inner(OP, NAME##x##2, ARG(vec<T, 2>))   \
impl_reduce_inner(OP, NAME##x##4, ARG(vec<T, 4>))   \
impl_reduce_strided(OP, NAME, T, )                  \
impl_reduce_strided(OP, NAME, ARG(vec<T, 2>), x##2) \
impl_reduce_strided(OP, NAME, ARG(vec<T, 4>), x##4)

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
    threadgroup indexed<T> shared[BLOCKSIZE],
    ushort tid [[ thread_index_in_threadgroup ]],
    ushort dst_id [[ threadgroup_position_in_grid ]]
) {
    // Initialize shared memory for current thread to correct value for reduction operation
    shared[tid] = ReductionOp::init();

    // Calcluate offset for the threadgroup of current thread
    ushort offset = dst_id * el_to_sum_per_block;
    indexed<T> initial = ReductionOp::init();
    // Load with reduction from global memory into shared memory
    shared[tid] = load_from_global<T, indexed<T>, ReductionOp, BLOCKSIZE, STRIDED>(
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
    indexed<T> value = threadgroup_reduce<indexed<T>, ReductionOp, BLOCKSIZE>(shared, offset, tid);

    // Return index of reduce result
    if (tid == 0) dst[dst_id] = finalize<ReductionOp>(value);
}

#define arg_reduce_case(OP, N, T)                       \
case N: {                                               \
    constexpr uint GRANULARITY = vec_elements<T>::value == 0 ? 1 : vec_elements<T>::value; \
    if (N / GRANULARITY == 0) {                         \
        break;                                          \
    }                                                   \
    constexpr uint B = N / GRANULARITY == 0 ? 1 : N / GRANULARITY;                 \
    threadgroup indexed<T> shared[B];              \
    reduce<T, OP<indexed<T>>, B, STRIDED>(              \
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

#define impl_arg_reduce_inner(OP, NAME, T, NAME_SUFFIX) \
kernel void NAME##NAME_SUFFIX(                          \
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
        arg_reduce_case(OP, 1024, ARG(T));              \
        arg_reduce_case(OP,  512, ARG(T));              \
        arg_reduce_case(OP,  256, ARG(T));              \
        arg_reduce_case(OP,  128, ARG(T));              \
        arg_reduce_case(OP,   64, ARG(T));              \
        arg_reduce_case(OP,   32, ARG(T));              \
        arg_reduce_case(OP,   16, ARG(T));              \
        arg_reduce_case(OP,    8, ARG(T));              \
        arg_reduce_case(OP,    4, ARG(T));              \
        arg_reduce_case(OP,    2, ARG(T));              \
        arg_reduce_case(OP,    1, ARG(T));              \
    }                                                   \
}                                                       \
kernel void NAME##NAME_SUFFIX##_strided(                \
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
        arg_reduce_case(OP, 1024, ARG(T));              \
        arg_reduce_case(OP,  512, ARG(T));              \
        arg_reduce_case(OP,  256, ARG(T));              \
        arg_reduce_case(OP,  128, ARG(T));              \
        arg_reduce_case(OP,   64, ARG(T));              \
        arg_reduce_case(OP,   32, ARG(T));              \
        arg_reduce_case(OP,   16, ARG(T));              \
        arg_reduce_case(OP,    8, ARG(T));              \
        arg_reduce_case(OP,    4, ARG(T));              \
        arg_reduce_case(OP,    2, ARG(T));              \
        arg_reduce_case(OP,    1, ARG(T));              \
    }                                                   \
}


#define impl_arg_reduce(OP, NAME, T)                    \
impl_arg_reduce_inner(OP, NAME, T, )                    \
impl_arg_reduce_inner(OP, NAME, ARG(vec<T, 2>), x##2)   \
impl_arg_reduce_inner(OP, NAME, ARG(vec<T, 4>), x##4)

/*
template<
    typename T,
    typename ACC = float4,
    ushort BLOCKSIZE,
    typename _E = typename metal::enable_if_t<is_vector_v<T> && is_vector_v<ACC>>
>
METAL_FUNC void _softmax_partial(
    constant size_t &src_numel,
    constant size_t &el_to_sum_per_block,
    const device T *src,
    const size_t offset,
    device Scalar<T> *dst,
    threadgroup ACC shared[BLOCKSIZE],
    const Scalar<ACC> max_result,

    const ushort tid,
    const ushort dst_id
) {
    const size_t N = vec_elements<T>::value;

    #pragma clang loop unroll(full)
    for (uint n = 0; n < N; n++) {
        dst[idx + n] = static_cast<Scalar<T>>(val[n]);
    }
}
*/


template<
    typename T,
    typename ACC = float4,
    ushort BLOCKSIZE,
    typename _E = typename metal::enable_if_t<is_vector_v<T> && is_vector_v<ACC>>
>
METAL_FUNC void _softmax_exp(
    constant size_t &src_numel,
    constant size_t &el_to_sum_per_block,
    const device T *src,
    const size_t offset,
    device Scalar<T> *dst,
    threadgroup ACC shared[BLOCKSIZE],
    const Scalar<ACC> max_result,

    const ushort tid,
    const ushort dst_id
) {
    // Calculate softmax values
    size_t stop_idx = min(offset + el_to_sum_per_block, src_numel);
    size_t idx = offset + tid;

    while (idx < stop_idx) {
        const ACC val = exp(ACC(src[idx]) - max_result);

        #pragma clang loop unroll(full)
        for (ushort n = 0; n < vec_elements<T>::value; n++) {
            dst[idx + n] = static_cast<Scalar<T>>(val[n]);
        }

        shared[tid] += val;
        idx += BLOCKSIZE + vec_elements<T>::value;
    }
    threadgroup_barrier(mem_flags::mem_none);
}

template<
    typename T,
    typename ACC = float,
    ushort BLOCKSIZE,
    typename _E = typename metal::enable_if<is_scalar_v<T> && is_scalar_v<ACC>>::type
>
METAL_FUNC void _softmax_exp(
    constant size_t &src_numel,
    constant size_t &el_to_sum_per_block,
    const device T *src,
    const size_t offset,
    device T *dst,
    threadgroup ACC shared[BLOCKSIZE],
    const ACC max_result,

    const ushort tid,
    const ushort dst_id
) {
    // Calculate softmax values
    size_t stop_idx = min(offset + el_to_sum_per_block, src_numel);
    size_t idx = offset + tid;

    while (idx < stop_idx) {
        const ACC val = exp(ACC(src[idx]) - max_result);
        dst[idx] = static_cast<T>(val);
        shared[tid] += val;
        idx += BLOCKSIZE;
    }
    threadgroup_barrier(mem_flags::mem_none);
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
    device Scalar<T> *dst,
    threadgroup ACC shared[BLOCKSIZE],

    ushort tid [[ thread_index_in_threadgroup ]],
    ushort dst_id [[ threadgroup_position_in_grid ]]
) {
    // Initialize shared memory for current thread to lowest value
    shared[tid] = Max<ACC>::init();

    // Calcluate offset for the threadgroup of current thread
    size_t offset = dst_id * el_to_sum_per_block;
    ACC initial = Max<ACC>::init();
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
    threadgroup_reduce<ACC, Max<ACC>, BLOCKSIZE>(shared, offset, tid);
    Scalar<ACC> max_result = finalize<Max<ACC>>(shared[0]);

    // Ensure all threads have max_result = shared[0] before we set shared[0] = 0.
    threadgroup_barrier(mem_flags::mem_none);
    shared[tid] = ACC(0);

    // Calculate softmax values
    const size_t stop_idx = min(offset + el_to_sum_per_block, src_numel);
    _softmax_exp<T, ACC, BLOCKSIZE>(src_numel, el_to_sum_per_block, src, offset, dst, shared, max_result, tid, dst_id);

    threadgroup_reduce<ACC, Sum<ACC>, BLOCKSIZE>(shared, offset, tid);
    threadgroup_barrier(mem_flags::mem_none);

    Scalar<ACC> acc = finalize<Sum<ACC>>(shared[0]);

    const Scalar<T> inv_acc = static_cast<Scalar<T>>(1.0/acc);
    size_t idx = offset + tid;
    while (idx < stop_idx) {
        dst[idx] *= inv_acc;
        idx += BLOCKSIZE;
    }
}

#define softmax_case(T, ACC, N)                         \
case N: {                                               \
    constexpr uint GRANULARITY = vec_elements<ACC>::value == 0 ? 1 : vec_elements<ACC>::value; \
    if (N / GRANULARITY == 0) {                         \
        break;                                          \
    }                                                   \
    constexpr uint B = N / GRANULARITY == 0 ? 1 : N / GRANULARITY;                 \
    threadgroup ACC shared[B];                          \
    softmax<T, ACC, B>(                                 \
        src_numel,                                      \
        el_to_sum_per_block,                            \
        src,                                            \
        dst,                                            \
        shared,                                         \
        tid,                                            \
        dst_id);                                        \
    break;                                              \
}

#define impl_softmax_inner(NAME, T, ACC)                \
kernel void NAME(                                       \
    constant size_t &src_numel,                         \
    constant size_t &el_to_sum_per_block,               \
    device const T *src,                                \
    device Scalar<T> *dst,                              \
                                                        \
    ushort tid [[ thread_index_in_threadgroup ]],       \
    ushort dst_id [[ threadgroup_position_in_grid ]],   \
    ushort block_dim [[ threads_per_threadgroup ]]      \
) {                                                     \
    switch (block_dim) {                                \
        softmax_case(T, ACC, 1024);                     \
        softmax_case(T, ACC,  512);                     \
        softmax_case(T, ACC,  256);                     \
        softmax_case(T, ACC,  128);                     \
        softmax_case(T, ACC,   64);                     \
        softmax_case(T, ACC,   32);                     \
        softmax_case(T, ACC,   16);                     \
        softmax_case(T, ACC,    8);                     \
        softmax_case(T, ACC,    4);                     \
        softmax_case(T, ACC,    2);                     \
        softmax_case(T, ACC,    1);                     \
    }                                                   \
}

#define impl_softmax(NAME, T, ACC)                      \
impl_softmax_inner(NAME, T, ACC)                        \
impl_softmax_inner(NAME##x2, T##2, ACC##2)              \
impl_softmax_inner(NAME##x4, T##4, ACC##4)

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

//#if __METAL_VERSION__ >= 220
//impl_reduce(Sum, fast_sum_i64, int64_t)
//impl_reduce(Mul, fast_mul_i64, int64_t)
//impl_reduce(Min, fast_min_i64, int64_t)
//impl_reduce(Max, fast_max_i64, int64_t)
//
//impl_arg_reduce(Min, fast_argmin_i64, int64_t)
//impl_arg_reduce(Max, fast_argmax_i64, int64_t)
//#endif

#if defined(__HAVE_BFLOAT__)
impl_reduce(Sum, fast_sum_bf16, bfloat)
impl_reduce(Mul, fast_mul_bf16, bfloat)
impl_reduce(Max, fast_max_bf16, bfloat)
impl_reduce(Min, fast_min_bf16, bfloat)

impl_arg_reduce(Min, fast_argmin_bf16, bfloat)
impl_arg_reduce(Max, fast_argmax_bf16, bfloat)

impl_softmax(softmax_bf16, bfloat, float)
#endif
