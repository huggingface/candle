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

// Keeps track of the index of the value in the reduction operation (argmin, argmax, etc.)
// and the value itself. The index is also used to break ties in the reduction operation.
// There are two specializations of the indexed class, one for scalar values and one for vector values.
template <typename T, typename = void>
class indexed;

template <typename T>
struct _make_scalar_impl<indexed<T>> {
    typedef indexed<make_scalar_t<T>> type;
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

    // constexpr indexed<T>(const volatile threadgroup indexed<T> &iv): indexed<T>(iv.i, iv.val) {}


    indexed<T> operator=(const threadgroup indexed<T> &iv) {
        this->i = iv.i;
        this->val = iv.val;
        return *this;
    }

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

    // constexpr indexed<T>(const volatile threadgroup indexed<T> &iv): indexed<T>(iv.i, iv.val) {}

    indexed<T> operator=(const threadgroup indexed<T> &iv) {
        this->i = iv.i;
        this->val = iv.val;
        return *this;
    }

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
// OPeration definitions are provided for default, thread, and threadgroup address spaces.
// OPerations are defined as scalar, but is applied for both scalar and vector via apply_operation.
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

template<typename T>
struct Sum {
    typedef Scalar<T> type;

    static constexpr METAL_FUNC T init() {
        return 0;
    }
    METAL_FUNC type operator()(type a, type b) {
        return a + b;
    }
    METAL_FUNC type operator()(
        thread const type& a,
        thread const type& b
    ) const {
        return a + b;
    }
    METAL_FUNC type operator()(
        threadgroup const type &a,
        threadgroup const type &b
    ) const {
        return a + b;
    }

    static METAL_FUNC T simd_op(T a) {
        return simd_sum(a);
    }
};

template<typename T>
struct Mul {
    typedef Scalar<T> type;

    static constexpr METAL_FUNC T init() {
        return 1;
    }
    METAL_FUNC type operator()(type a, type b) {
        return a * b;
    }
    METAL_FUNC type operator()(
        thread const type& a,
        thread const type& b
    ) const {
        return a * b;
    }
    METAL_FUNC type operator()(
        threadgroup const type &a,
        threadgroup const type &b
    ) const {
        return a * b;
    }

    static METAL_FUNC T simd_op(T a) {
        return simd_product(a);
    }
};

template<typename T>
struct Min {
    typedef Scalar<T> type;

    static constexpr METAL_FUNC T init() {
        return numeric_limits<T>::max();
    }
    METAL_FUNC type operator()(type a, type b) {
        return a < b ? a : b;
    }
    METAL_FUNC type operator()(
        thread const type& a,
        thread const type& b
    ) const {
        return a < b ? a : b;
    }
    METAL_FUNC type operator()(
        threadgroup const type &a,
        threadgroup const type &b
    ) const {
        return a < b ? a : b;
    }

    static METAL_FUNC T simd_op(T a) {
        return simd_min(a);
    }
};

template<typename T>
struct Max {
    typedef Scalar<T> type;

    static constexpr METAL_FUNC T init() {
        return numeric_limits<T>::lowest();
    }
    METAL_FUNC type operator()(type a, type b) {
        return a > b ? a : b;
    }
    METAL_FUNC type operator()(
        thread const type& a,
        thread const type& b
    ) const {
        return a > b ? a : b;
    }
    METAL_FUNC type operator()(
        threadgroup const type &a,
        threadgroup const type &b
    ) const {
        return a > b ? a : b;
    }

    static METAL_FUNC T simd_op(T a) {
        return simd_max(a);
    }
};

template <typename T>
constexpr constant bool is_simd_t = __is_valid_simdgroup_type<T>::value;

template <typename T, typename _E = void>
struct is_valid_simd_type {
    static constant constexpr bool value = false;
};

template <typename T>
constexpr constant bool is_valid_simd_t = is_valid_simd_type<T>::value;

template <typename T>
struct is_valid_simd_type<T, typename metal::enable_if_t<is_simd_t<T>>> {
    static constant constexpr bool value = true;
};

template <typename T>
struct is_valid_simd_type<indexed<T>, typename metal::enable_if_t<is_valid_simd_t<T>>> {
    static constant constexpr bool value = true;
};

#if defined(__HAVE_BFLOAT__)
template <>
struct is_valid_simd_type<bfloat> {
    static constant constexpr bool value = true;
};

template <uint N>
struct is_valid_simd_type<vec<bfloat, N>> {
    static constant constexpr bool value = true;
};
#endif

template <typename T, typename _E = void>
struct is_simd_op {
    static constant constexpr bool value = false;
};
template <typename T>
struct is_simd_op<Sum<T>, typename metal::enable_if_t<is_simd_t<T>>> {
    static constant constexpr bool value = true;
};
template <typename T>
struct is_simd_op<Mul<T>, typename metal::enable_if_t<is_simd_t<T>>> {
    static constant constexpr bool value = true;
};
template <typename T>
struct is_simd_op<Min<T>, typename metal::enable_if_t<is_simd_t<T>>> {
    static constant constexpr bool value = true;
};
template <typename T>
struct is_simd_op<Max<T>, typename metal::enable_if_t<is_simd_t<T>>> {
    static constant constexpr bool value = true;
};

// Helper struct for applying operators.
// The overloaded operator() function is used to apply an operator to two values.
template<
    typename OP,
    typename T,
    typename _E = void
>
struct operation;

// Specialization for scalar values.
template<typename OP, typename T>
struct operation<OP, T, typename metal::enable_if_t<is_scalar_v<T>>> {
    METAL_FUNC T operator()(OP op, T a, T b) const {
        return op(a, b);
    }
    METAL_FUNC T operator()(OP op, T a, T b, uint idx) {
        return this->operator()(op, a, b);
    }
};

// Specialization for vector values.
template<typename OP, typename T, uint N>
struct operation<OP, vec<T, N>> {
    METAL_FUNC vec<T, N> operator()(OP op, vec<T, N> a, vec<T, N> b) {
        #pragma clang loop unroll(full)
        for (ushort n = 0; n < vec_elements<T>::value; n++) {
            a[n] = op(a[n], b[n]);
        }
        return a;
    }
    METAL_FUNC vec<T, N> operator()(OP op, vec<T, N> a, vec<T, N> b, uint _idx) {
        return this->operator()(op, a, b);
    }
};

// Specialization for indexed scalar values.
template<typename OP, typename T>
struct operation<OP, indexed<T>, typename metal::enable_if_t<is_scalar_v<T>>> {
    METAL_FUNC indexed<T> operator()(OP op, indexed<T> a, indexed<T> b) {
        return op(a, b);
    }
    METAL_FUNC indexed<T> operator()(OP op, indexed<T> a, T b, uint idx) {
        return this->operator()(op, a, indexed<T>(idx, b));
    }
};

// Specialization for indexed vector values.
template<typename OP, typename T, uint N>
struct operation<OP, indexed<vec<T, N>>> {
    METAL_FUNC indexed<vec<T, N>> operator()(OP op, indexed<vec<T, N>> a, indexed<vec<T, N>> b) {
        #pragma clang loop unroll(full)
        for (ushort n = 0; n < N; n++) {
            a[n] = op(a[n], b[n]);
        }
        return a;
    }
    METAL_FUNC indexed<vec<T, N>> operator()(OP op, indexed<vec<T, N>> a, vec<T, N> b, uint idx) {
        #pragma clang loop unroll(full)
        for (ushort n = 0; n < N; n++) {
            a[n] = op(a[n], indexed<T>(idx + n, b[n]));
        }
        return a;
    }
};

template<typename OP, typename T, typename _E = typename metal::enable_if_t<is_scalar_v<T>>>
METAL_FUNC uint finalize(indexed<T> value) {
    return value.i;
}

template<typename OP, typename T, uint N, typename _E = typename metal::enable_if_t<is_scalar_v<T>>>
METAL_FUNC uint finalize(indexed<vec<T, N>> value) {
    OP op;
    indexed<T> result = value[0];
    #pragma clang loop unroll(full)
    for (ushort n = 1; n < N; n++) {
        result = op(result, value[n]);
    }
    return result.i;
}

template<typename OP, typename T, typename _E = typename metal::enable_if_t<is_scalar_v<T>>>
METAL_FUNC T finalize(T value) {
    return value;
}

template<typename OP, typename T, typename _E = typename metal::enable_if_t<is_vector_v<T>>>
METAL_FUNC Scalar<T> finalize(T value) {
    OP op;
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
    typename OP,
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
    OP op;
    operation<OP, R> apply_operator;
    uint idx = offset + tid;
    #pragma clang loop unroll(full)
    for (uint i = offset + tid; i < offset + el_to_sum_per_block; i += BLOCKSIZE) {
        if (STRIDED) {
            // TODO: if vec.
            // #pragma clang loop unroll(full)
            //for (ushort n = 0; n < granularity<T>(); n++) {
            //
            //}
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
    typename OP,
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
    return load_from_global<T, R, OP, BLOCKSIZE, false>(
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

template<
    typename OP,
    ushort BLOCKSIZE,
    typename T,
    typename _E = void
>
struct simdgroup_reducer;


// Specialization for built-in simd operations.
template<typename OP, ushort BLOCKSIZE, typename T>
struct simdgroup_reducer<OP, BLOCKSIZE, T, typename metal::enable_if_t<is_simd_op<OP>::value && is_valid_simd_t<T>>> {
    METAL_FUNC T operator()(T value) {
        return OP::simd_op(value);
    }
    METAL_FUNC T operator()(threadgroup T shared[BLOCKSIZE], const ushort tid) {
        return this->operator()(shared[tid]);
    }
};

// Specialization for custom (non-built-in) simd operations.
template<typename OP, ushort BLOCKSIZE, typename T>
struct simdgroup_reducer<OP, BLOCKSIZE, T, typename metal::enable_if_t<!is_simd_op<OP>::value && is_valid_simd_t<T>>> {
    METAL_FUNC T operator()(T value) {
        OP op;
        operation<OP, T> apply_operator;
        if (BLOCKSIZE >= 32) value = apply_operator(op, value, simd_shuffle_down(value, 16));
        if (BLOCKSIZE >= 16) value = apply_operator(op, value, simd_shuffle_down(value,  8));
        if (BLOCKSIZE >=  8) value = apply_operator(op, value, simd_shuffle_down(value,  4));
        if (BLOCKSIZE >=  4) value = apply_operator(op, value, simd_shuffle_down(value,  2));
        if (BLOCKSIZE >=  2) value = apply_operator(op, value, simd_shuffle_down(value,  1));
        return value;
    }
    METAL_FUNC T operator()(threadgroup T shared[BLOCKSIZE], const ushort tid) {
        return this->operator()(shared[tid]);
    }
};

// Specialization for non-simd types.
//template<typename OP, ushort BLOCKSIZE, typename T>
//struct simdgroup_reducer<OP, BLOCKSIZE, T, typename metal::enable_if_t<!is_valid_simd_t<T>>> {
//    METAL_FUNC T operator()(
//        volatile threadgroup T shared[BLOCKSIZE],
//        const ushort tid
//    ) {
//        OP op;
//        operation<OP, T> apply_operator;
//        T value = shared[tid];
//        if (BLOCKSIZE >= 32) value = apply_operator(op, value, shared[tid + 16]);
//        if (BLOCKSIZE >= 16) value = apply_operator(op, value, shared[tid +  8]);
//        if (BLOCKSIZE >=  8) value = apply_operator(op, value, shared[tid +  4]);
//        if (BLOCKSIZE >=  4) value = apply_operator(op, value, shared[tid +  2]);
//        if (BLOCKSIZE >=  2) value = apply_operator(op, value, shared[tid +  1]);
//        return value;
//    }
//    METAL_FUNC T operator()(T value) {
//        return value;
//    }
//};

template<
   typename T,
   typename OP,
   ushort BLOCKSIZE
>
METAL_FUNC T threadgroup_reduce(
    T value,
    threadgroup T shared[BLOCKSIZE],
    const ushort tid
) {
    OP op;
    operation<OP, T> apply_operator;

    if (BLOCKSIZE >= 64) {
        // Only store in threadgroup shared memory if needed.
        shared[tid] = value;
        // Threadgroup barrier is needed to ensure that all threads have written to shared memory
        threadgroup_barrier(mem_flags::mem_none);
    }

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
            value = apply_operator(op, shared[tid], shared[tid + 32]);
            simdgroup_barrier(mem_flags::mem_none);
        }
        // Remaining 32 threads can be reduced with simdgroup_reduce.
        simdgroup_reducer<OP, BLOCKSIZE, T> reduce;
        value = reduce(value);
    }

    return value;
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
    R value = ReductionOp::init();
    // Load with reduction from global memory into shared memory
    value = load_from_global<T, R, ReductionOp, BLOCKSIZE, STRIDED>(
        value,
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

    // Complete reduction
    value = threadgroup_reduce<R, ReductionOp, BLOCKSIZE>(value, shared, tid);

    if (tid == 0) dst[dst_id] = finalize<ReductionOp>(value);
}

template<uint N>
constexpr uint nonzero() {
    return N == 0 ? 1 : N;
}

template<typename T>
constexpr uint granularity() {
    return nonzero<vec_elements<T>::value>();
}

#define reduce_case(OP, T, R, N)                        \
case N: {                                               \
    if (N / GRANULARITY == 0) {                         \
        break;                                          \
    }                                                   \
    constexpr uint B = nonzero<N / GRANULARITY>();      \
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
    constexpr uint GRANULARITY = granularity<T>();      \
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
    constexpr uint GRANULARITY = granularity<T>();      \
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
    indexed<T> value = ReductionOp::init();
    // Load with reduction from global memory into shared memory
    value = load_from_global<T, indexed<T>, ReductionOp, BLOCKSIZE, STRIDED>(
        value,
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

    // Complete reduction
    value = threadgroup_reduce<indexed<T>, ReductionOp, BLOCKSIZE>(value, shared, tid);

    // Return index of reduce result
    if (tid == 0) dst[dst_id] = finalize<ReductionOp>(value);
}

#define arg_reduce_case(OP, N, T)                       \
case N: {                                               \
    if (N / GRANULARITY == 0) {                         \
        break;                                          \
    }                                                   \
    constexpr uint B = nonzero<N / GRANULARITY>();      \
    threadgroup indexed<T> shared[B];                   \
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
    constexpr uint GRANULARITY = granularity<T>();      \
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
    constexpr uint GRANULARITY = granularity<T>();      \
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

// Contains the intermediate results for the online softmax calculation.
// m: max
// d: sum of the exponentials
template <typename T>
struct MD {
    T m;
    T d;
};

template <typename T>
METAL_FUNC MD<T> simd_shuffle_down(MD<T> md, ushort delta) {
    return MD<T> {
        simd_shuffle_down(md.m, delta),
        simd_shuffle_down(md.d, delta)
    };
}

// Enable simd_shuffle_down for softmax MD
template <typename T>
struct is_valid_simd_type<MD<T>, typename metal::enable_if_t<is_valid_simd_t<T>>> {
    static constant constexpr bool value = true;
};

// Enable operations for softmax MD
template<typename OP, typename T>
struct operation<OP, MD<T>, typename metal::enable_if_t<is_scalar_v<T>>> {
    METAL_FUNC MD<T> operator()(OP op, MD<T> a, MD<T> b) {
        return op(a, b);
    }
    METAL_FUNC MD<T> operator()(OP op, MD<T> a, MD<T> b, uint idx) {
        return this->operator()(op, a, b);
    }
};

template<typename T>
struct MDReduceOp {

    static constexpr METAL_FUNC MD<T> init() {
        return MD<T>{ numeric_limits<T>::lowest(), 0 };
    }

    METAL_FUNC MD<T> operator()(MD<T> a, MD<T> b) {
        bool a_bigger = a.m > b.m;
        MD<T> bigger_m = a_bigger ? a : b;
        MD<T> smaller_m = a_bigger ? b : a;
        MD<T> res;
        res.d = bigger_m.d + smaller_m.d *  static_cast<T>(fast::exp(smaller_m.m - bigger_m.m));
        res.m = bigger_m.m;
        return res;
    }
    METAL_FUNC MD<T> operator()(
        thread const MD<T> &a,
        thread const MD<T> &b
    ) const {
        bool a_bigger = a.m > b.m;
        MD<T> bigger_m = a_bigger ? a : b;
        MD<T> smaller_m = a_bigger ? b : a;
        MD<T> res;
        res.d = bigger_m.d + smaller_m.d *  static_cast<T>(fast::exp(smaller_m.m - bigger_m.m));
        res.m = bigger_m.m;
        return res;
    }
    METAL_FUNC MD<T> operator()(
        threadgroup const MD<T> &a,
        threadgroup const MD<T> &b
    ) const {
        bool a_bigger = a.m > b.m;
        MD<T> bigger_m = a_bigger ? a : b;
        MD<T> smaller_m = a_bigger ? b : a;
        MD<T> res;
        res.d = bigger_m.d + smaller_m.d *  static_cast<T>(fast::exp(smaller_m.m - bigger_m.m));
        res.m = bigger_m.m;
        return res;
    }

    METAL_FUNC MD<T> simd_op(MD<T> a) {
        return a;
    }
};

// Welford's algorithm approach for an online softmax implementation.
// Same as the Online normalizer calculation for softmax: https://arxiv.org/pdf/1805.02867.pdf
template<typename T, ushort BLOCKSIZE>
METAL_FUNC void softmax(
    constant size_t &src_numel,
    constant size_t &el_to_sum_per_block,
    const device T *src,
    device Scalar<T> *dst,
    threadgroup MD<T> shared[BLOCKSIZE],
    threadgroup MD<T> &md_total,

    ushort tid [[ thread_index_in_threadgroup ]],
    ushort dst_id [[ threadgroup_position_in_grid ]]
) {
    MDReduceOp<T> op;

    // Calcluate offset for the threadgroup of current thread
    const size_t offset = dst_id * el_to_sum_per_block;
    const size_t thread_id = tid + offset;
    const size_t stop_idx = min(el_to_sum_per_block + offset, src_numel);

    // Calculate partial result for current thread
    MD<T> md_partial = MD<T> { numeric_limits<T>::lowest(), 0 };
    for (size_t idx = thread_id; idx < stop_idx; idx += BLOCKSIZE) {
        md_partial = op(md_partial, MD<T> { src[idx], static_cast<T>(1.0) });
    }

    // Reduce in shared memory
    MD<T> md = threadgroup_reduce<MD<T>, MDReduceOp<T>, BLOCKSIZE>(md_partial, shared, tid);

    if (tid == 0) md_total = md;
    threadgroup_barrier(mem_flags::mem_none);

    const T d_total_inverse = static_cast<T>(fast::divide(1.0, md_total.d));
    for (size_t idx = thread_id; idx < stop_idx; idx += BLOCKSIZE) {
        dst[idx] = static_cast<T>(fast::exp(src[idx] - md_total.m)) * d_total_inverse;
    }
}

#define softmax_case(T, N)                              \
case N: {                                               \
    if (N / GRANULARITY == 0) {                         \
        break;                                          \
    }                                                   \
    constexpr uint B = nonzero<N / GRANULARITY>();      \
    threadgroup MD<T> shared[B];                        \
    threadgroup MD<T> md_total;                         \
    softmax<T, B>(                                      \
        src_numel,                                      \
        el_to_sum_per_block,                            \
        src,                                            \
        dst,                                            \
        shared,                                         \
        md_total,                                       \
        tid,                                            \
        dst_id);                                        \
    break;                                              \
}

#define impl_softmax_inner(NAME, T)                     \
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
    constexpr uint GRANULARITY = granularity<T>();      \
    switch (block_dim) {                                \
        softmax_case(T, 1024);                          \
        softmax_case(T,  512);                          \
        softmax_case(T,  256);                          \
        softmax_case(T,  128);                          \
        softmax_case(T,   64);                          \
        softmax_case(T,   32);                          \
        softmax_case(T,   16);                          \
        softmax_case(T,    8);                          \
        softmax_case(T,    4);                          \
        softmax_case(T,    2);                          \
        softmax_case(T,    1);                          \
    }                                                   \
}

#define impl_softmax(NAME, T)                      \
impl_softmax_inner(NAME, T)                        \
//impl_softmax_inner(NAME##x2, T##2)              \
//impl_softmax_inner(NAME##x4, T##4)

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

impl_softmax(softmax_f32, float)
impl_softmax(softmax_f16, half)

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

impl_softmax(softmax_bf16, bfloat)
#endif
