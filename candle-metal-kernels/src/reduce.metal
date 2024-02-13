#include <metal_stdlib>
#include <metal_limits>
using namespace metal;

template<uint N>
constexpr uint nonzero() {
    return N == 0 ? 1 : N;
}

template<typename T>
constexpr uint granularity() {
    return nonzero<vec_elements<T>::value>();
}

struct Divide {
    template<typename T>
    METAL_FUNC T operator()(T a, T b) { return a / b; }

    template<> METAL_FUNC float  operator()(float  a, float  b) { return fast::divide(a, b); }
    template<> METAL_FUNC float2 operator()(float2 a, float2 b) { return fast::divide(a, b); }
    template<> METAL_FUNC float4 operator()(float4 a, float4 b) { return fast::divide(a, b); }
    template<> METAL_FUNC half   operator()(half   a, half   b) { return divide(a, b); }
    template<> METAL_FUNC half2  operator()(half2  a, half2  b) { return divide(a, b); }
    template<> METAL_FUNC half4  operator()(half4  a, half4  b) { return divide(a, b); }
    #if defined(__HAVE_BFLOAT__)
    template<> METAL_FUNC bfloat  operator()(bfloat   a,  bfloat b) { return static_cast<bfloat>(fast::divide(a, b)); }
    template<> METAL_FUNC bfloat2 operator()(bfloat2  a, bfloat2 b) { return static_cast<bfloat2>( a / b ); }
    template<> METAL_FUNC bfloat4 operator()(bfloat4  a, bfloat4 b) { return static_cast<bfloat4>( a / b ); }
    #endif
};

struct Exp {
    template<typename T>
    METAL_FUNC T operator()(T a) { return fast::exp(a); }

    template<> METAL_FUNC float  operator()(float  a) { return fast::exp(a); }
    template<> METAL_FUNC float2 operator()(float2 a) { return fast::exp(a); }
    template<> METAL_FUNC float4 operator()(float4 a) { return fast::exp(a); }
    template<> METAL_FUNC half   operator()(half   a) { return exp(a); }
    template<> METAL_FUNC half2  operator()(half2  a) { return exp(a); }
    template<> METAL_FUNC half4  operator()(half4  a) { return exp(a); }
    #if defined(__HAVE_BFLOAT__)
    template<>
    METAL_FUNC bfloat  operator()(bfloat  a) { return static_cast<bfloat>(fast::exp(a)); }
    template<>
    METAL_FUNC bfloat2 operator()(bfloat2 a) { return static_cast<bfloat2>(fast::exp(static_cast<float2>(a))); }
    template<>
    METAL_FUNC bfloat4 operator()(bfloat4 a) { return static_cast<bfloat4>(fast::exp(static_cast<float4>(a))); }
    #endif
};

METAL_FUNC uint get_strided_index(
    uint idx,
    constant const uint &num_dims,
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
struct indexed;

// Specialization for scalar values
template <typename T>
struct indexed<T, typename metal::enable_if_t<is_scalar_v<T>>> {
    uint i;
    T val;

    constexpr indexed<T>() threadgroup = default;

    // To align with below implementation.
    constexpr auto operator[](uint n) {
        assert(n == 0);
        return *this;
    }
};

// Support turning indexed<T> into indexed<make_scalar_t<T>>.
template <typename T>
struct _make_scalar_impl<indexed<T>> {
    typedef indexed<make_scalar_t<T>> type;
};

// Specialization for vector values
template <typename T>
struct indexed<T, typename metal::enable_if_t<is_vector_v<T>>> {
    typedef indexed<make_scalar_t<T>> scalar;
    typedef vec<uint, vec_elements<T>::value> I;
    I i;
    T val;

    constexpr indexed<T>() threadgroup = default;

    // Return 1-dimensional indexed value
    constexpr scalar operator[](uint n) {
        assert(n < N);
        return scalar{ i[n], val[n] };
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
        return indexed<T>{ 0, numeric_limits<T>::lowest() };
    }

    static constexpr METAL_FUNC indexed<T> max() {
        return indexed<T>{ 0, numeric_limits<T>::max() };
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
    return indexed<T> {
        simd_shuffle_down(iv.i, delta),
        simd_shuffle_down(iv.val, delta)
    };
}

template<typename T>
struct Sum {
    typedef make_scalar_t<T> scalar;

    static constexpr METAL_FUNC T init() {
        return 0;
    }
    METAL_FUNC scalar operator()(scalar a, scalar b) {
        return a + b;
    }
    static METAL_FUNC T simd_op(T a) {
        return simd_sum(a);
    }
};

template<typename T>
struct Mul {
    typedef make_scalar_t<T> scalar;

    static constexpr METAL_FUNC T init() {
        return 1;
    }
    METAL_FUNC scalar operator()(scalar a, scalar b) {
        return a * b;
    }
    static METAL_FUNC T simd_op(T a) {
        return simd_product(a);
    }
};

template<typename T>
struct Min {
    typedef make_scalar_t<T> scalar;

    static constexpr METAL_FUNC T init() {
        return numeric_limits<T>::max();
    }
    METAL_FUNC scalar operator()(scalar a, scalar b) {
        return a < b ? a : b;
    }
    static METAL_FUNC T simd_op(T a) {
        return simd_min(a);
    }
};

template<typename T>
struct Max {
    typedef make_scalar_t<T> scalar;

    static constexpr METAL_FUNC T init() {
        return numeric_limits<T>::lowest();
    }
    METAL_FUNC scalar operator()(scalar a, scalar b) {
        return a > b ? a : b;
    }
    static METAL_FUNC T simd_op(T a) {
        return simd_max(a);
    }
};

// For testing purposes
template<typename T>
struct Identity {
    typedef make_scalar_t<T> scalar;

    static constexpr METAL_FUNC T init() {
        return numeric_limits<T>::lowest();
    }
    METAL_FUNC scalar operator()(scalar a, scalar b) {
        return b;
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
    OP op;

    METAL_FUNC T operator()(T a, T b) {
        return op(a, b);
    }
    METAL_FUNC T operator()(T a, T b, uint idx) {
        return this->operator()(a, b);
    }
};

// Specialization for vector values.
template<typename OP, typename T, uint N>
struct operation<OP, vec<T, N>> {
    OP op;

    METAL_FUNC vec<T, N> operator()(vec<T, N> a, vec<T, N> b) {
        #pragma clang loop unroll(full)
        for (ushort n = 0; n < N; n++) {
            a[n] = op(a[n], b[n]);
        }
        return a;
    }
    METAL_FUNC vec<T, N> operator()(vec<T, N> a, vec<T, N> b, uint _idx) {
        return this->operator()(a, b);
    }
};

// Specialization for indexed scalar values.
template<typename OP, typename T>
struct operation<OP, indexed<T>, typename metal::enable_if_t<is_scalar_v<T>>> {
    OP op;

    METAL_FUNC indexed<T> operator()(indexed<T> a, indexed<T> b) {
        return op(a, b);
    }
    METAL_FUNC indexed<T> operator()(indexed<T> a, T b, uint idx) {
        return this->operator()(a, indexed<T>{ idx, b });
    }
};

// Specialization for indexed vector values.
template<typename OP, typename T>
struct operation<OP, indexed<T>, typename metal::enable_if_t<is_vector_v<T>>> {
    typedef indexed<make_scalar_t<T>> scalar;
    OP op;

    METAL_FUNC indexed<T> operator()(indexed<T> a, indexed<T> b) {
        #pragma clang loop unroll(full)
        for (ushort n = 0; n < vec_elements<T>::value; n++) {
            a[n] = op(a[n], b[n]);
        }
        return a;
    }
    METAL_FUNC indexed<T> operator()(indexed<T> a, T b, uint idx) {
        #pragma clang loop unroll(full)
        for (ushort n = 0; n < vec_elements<T>::value; n++) {
            a[n] = op(a[n], scalar{ idx + n, b[n] });
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
METAL_FUNC make_scalar_t<T> finalize(T value) {
    OP op;
    make_scalar_t<T> result = value[0];

    #pragma clang loop unroll(full)
    for (ushort n = 1; n < vec_elements<T>::value; n++) {
        result = op(result, value[n]);
    }
    return result;
}

// Load elements from global memory into shared memory.
// Handles both indexed and non-indexed types by using operate.
template<
    typename T,
    typename R,
    typename OP,
    ushort BLOCKSIZE,
    bool STRIDED = false,
    typename _E = void
>
struct loader;

template<
    typename T,
    typename R,
    typename OP,
    bool STRIDED,
    ushort BLOCKSIZE
>
struct loader<T, R, OP, BLOCKSIZE, STRIDED, typename metal::enable_if_t<is_scalar_v<T>>> {
    operation<OP, R> operate;

    METAL_FUNC R operator()(
        R value,
        constant uint &src_numel,
        const uint el_to_sum_per_block,
        const device T *src,
        const ushort offset,
        const ushort tid
    ) {
        const uint thread_id = tid + offset;
        const uint stop_idx = el_to_sum_per_block + offset;

        #pragma clang loop unroll(full)
        for (uint i = thread_id; i < stop_idx; i += BLOCKSIZE) {
            value = operate(value, src[i], i);
        }
        return value;
    }

    METAL_FUNC R operator()(
        R value,
        constant uint &src_numel,
        constant size_t *dims,
        constant size_t *strides,
        const uint el_to_sum_per_block,
        const device T *src,
        const ushort offset,
        const ushort tid
    ) {
        if (!STRIDED) {
            return this->operator()(
                value,
                src_numel,
                el_to_sum_per_block,
                src,
                offset,
                tid
            );
        }
        const uint thread_id = tid + offset;
        const uint stop_idx = min(el_to_sum_per_block + offset, src_numel);

        uint idx = thread_id;
        #pragma clang loop unroll(full)
        for (uint i = thread_id; i < stop_idx; i += BLOCKSIZE) {
            idx = get_strided_index(i, src_numel, dims, strides);
            value = operate(value, src[idx], idx);
        }
        return value;
    }
};

template<
    typename T,
    typename R,
    typename OP,
    ushort BLOCKSIZE,
    bool STRIDED
>
struct loader<T, R, OP, BLOCKSIZE, STRIDED, typename metal::enable_if_t<is_vector_v<T>>> {
    operation<OP, R> operate;

    METAL_FUNC R operator()(
        R value,
        constant uint &src_numel,
        const uint el_per_block,
        const device T *src,
        const ushort offset,
        const ushort tid
    ) {
        // blocksize = 3
        // G = 2
        // offset = dst_id * 3
        constexpr uint G = granularity<T>();
        const uint thread_id = tid + (offset / G);
        const uint stop_idx = min(el_per_block + offset, src_numel) / G;

        #pragma clang loop unroll(full)
        for (uint i = thread_id; i < stop_idx; i += BLOCKSIZE) {
            value = operate(value, src[i], i);
        }
        return value;
    }

    METAL_FUNC R operator()(
        R value,
        constant uint &src_numel,
        constant size_t *dims,
        constant size_t *strides,
        const uint el_to_sum_per_block,
        const device T *src,
        const ushort offset,
        const ushort tid
    ) {
        if (!STRIDED) {
            return this->operator()(
                value,
                src_numel,
                el_to_sum_per_block,
                src,
                offset,
                tid
            );
        }
        //const uint thread_id = tid + offset;
        //const uint stop_idx = el_to_sum_per_block + offset;
        //
        //uint idx = thread_id;
        //#pragma clang loop unroll(full)
        //for (uint i = thread_id; i < stop_idx; i += BLOCKSIZE) {
        //    idx = get_strided_index(i, src_numel, dims, strides);
        //    value = operate(value, src[idx], idx);
        //}
        return value;
    }
};

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
    operation<OP, T> operate;

    METAL_FUNC T operator()(T value) {
        if (BLOCKSIZE >= 32) value = operate(value, simd_shuffle_down(value, 16));
        if (BLOCKSIZE >= 16) value = operate(value, simd_shuffle_down(value,  8));
        if (BLOCKSIZE >=  8) value = operate(value, simd_shuffle_down(value,  4));
        if (BLOCKSIZE >=  4) value = operate(value, simd_shuffle_down(value,  2));
        if (BLOCKSIZE >=  2) value = operate(value, simd_shuffle_down(value,  1));
        return value;
    }
    METAL_FUNC T operator()(threadgroup T shared[BLOCKSIZE], const ushort tid) {
        return this->operator()(shared[tid]);
    }
};

// Specialization for non-simd types.
//template<typename OP, ushort BLOCKSIZE, typename T>
//struct simdgroup_reducer<OP, BLOCKSIZE, T, typename metal::enable_if_t<!is_valid_simd_t<T>>> {
//    operation<OP, T> operate;
//
//    METAL_FUNC T operator()(
//        volatile threadgroup T shared[BLOCKSIZE],
//        const ushort tid
//    ) {
//        T value = shared[tid];
//        if (BLOCKSIZE >= 32) value = operate(value, shared[tid + 16]);
//        if (BLOCKSIZE >= 16) value = operate(value, shared[tid +  8]);
//        if (BLOCKSIZE >=  8) value = operate(value, shared[tid +  4]);
//        if (BLOCKSIZE >=  4) value = operate(value, shared[tid +  2]);
//        if (BLOCKSIZE >=  2) value = operate(value, shared[tid +  1]);
//        return value;
//    }
//    METAL_FUNC T operator()(T value) {
//        return value;
//    }
//};

template<typename T, typename OP, ushort BLOCKSIZE>
struct block_reducer {
    simdgroup_reducer<OP, BLOCKSIZE, T> simd_reduce;
    operation<OP, T> operate;
    threadgroup T *shared;

    block_reducer(threadgroup T shared[BLOCKSIZE]) {
        this->shared = shared;
    }

    METAL_FUNC T operator()(T value, const ushort tid) {
        if (BLOCKSIZE >= 64) {
            // Only store in threadgroup shared memory if needed.
            shared[tid] = value;
            // Threadgroup barrier is needed to ensure that all threads have written to shared memory
            threadgroup_barrier(mem_flags::mem_none);
        }

        #pragma clang loop unroll(full)
        for (ushort s = BLOCKSIZE / 2; s >= 64; s >>= 1) {
            if (tid < s) {
                shared[tid] = operate(shared[tid], shared[tid + s]);
            }
            threadgroup_barrier(mem_flags::mem_none);
        }
        if (tid < 32) {
            // Last shared memory reduce can be done without tid < s check.
            if (BLOCKSIZE >= 64) {
                value = operate(shared[tid], shared[tid + 32]);
                simdgroup_barrier(mem_flags::mem_none);
            }
            // Remaining 32 threads can be reduced with simdgroup_reduce.
            value = simd_reduce(value);
        }

        return value;
    }
};

// Inspired by "Optimizing Parallel Reduction in CUDA" by Mark Harris
template<
    typename T,
    typename R,
    typename ReductionOp,
    ushort BLOCKSIZE,
    bool STRIDED = false
>
METAL_FUNC void reduce(
    constant uint &num_dims,
    constant size_t *dims,
    constant size_t *strides,
    constant uint &el_to_sum_per_block,
    device const T *src,
    device make_scalar_t<R> *dst,
    threadgroup R shared[BLOCKSIZE],
    ushort tid [[ thread_index_in_threadgroup ]],
    ushort dst_id [[ threadgroup_position_in_grid ]]
) {
    // Initialize shared memory for current thread to correct value for reduction operation
    shared[tid] = ReductionOp::init();

    // Calcluate offset for the threadgroup of current thread;
    const uint el_to_sum = el_to_sum_per_block;
    const uint offset = dst_id * el_to_sum;
    R value = ReductionOp::init();

    loader<T, R, ReductionOp, BLOCKSIZE, STRIDED> load;
    // Load with reduction from global memory into shared memory
    value = load(
        value,
        num_dims,
        dims,
        strides,
        el_to_sum,
        src,
        offset,
        tid
    );

    // Complete reduction
    block_reducer<T, ReductionOp, BLOCKSIZE> block_reduce(shared);
    value = block_reduce(value, tid);

    if (tid == 0) dst[dst_id] = finalize<ReductionOp>(value);
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
        shared,                                         \
        tid,                                            \
        dst_id);                                        \
    break;                                              \
}

#define ARG(...) __VA_ARGS__

#define impl_reduce_inner(OP, NAME, T)                  \
kernel void NAME(                                       \
    constant uint &num_dims,                            \
    constant uint &el_to_sum_per_block,                 \
    device const T *src,                                \
    device make_scalar_t<T> *dst,                       \
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
    constant uint &num_dims,                            \
    constant size_t *dims,                              \
    constant size_t *strides,                           \
    constant uint &el_to_sum_per_block,                 \
    device const T *src,                                \
    device make_scalar_t<T> *dst,                       \
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
impl_reduce_inner(OP, NAME##x2, ARG(vec<T, 2>))     \
impl_reduce_inner(OP, NAME##x4, ARG(vec<T, 4>))     \
impl_reduce_strided(OP, NAME, T, )                  \
impl_reduce_strided(OP, NAME, ARG(vec<T, 2>), x2)   \
impl_reduce_strided(OP, NAME, ARG(vec<T, 4>), x4)

template<
    typename T,
    typename ReductionOp,
    ushort BLOCKSIZE,
    bool STRIDED
>
METAL_FUNC void reduce(
    constant uint &num_dims,
    constant size_t *dims,
    constant size_t *strides,
    constant uint &el_to_sum_per_block,
    device const T *src,
    device uint *dst,
    threadgroup indexed<T> shared[BLOCKSIZE],
    ushort tid [[ thread_index_in_threadgroup ]],
    ushort dst_id [[ threadgroup_position_in_grid ]]
) {
    loader<T, indexed<T>, ReductionOp, BLOCKSIZE, STRIDED> load;
    block_reducer<indexed<T>, ReductionOp, BLOCKSIZE> block_reduce(shared);

    // Initialize shared memory for current thread to correct value for reduction operation
    shared[tid] = ReductionOp::init();

    // Calcluate offset for the threadgroup of current thread
    const uint el_to_sum = el_to_sum_per_block;
    const uint offset = dst_id * el_to_sum;

    // Load with reduction from global memory into shared memory
    indexed<T> value = ReductionOp::init();
    value = load(
        value,
        num_dims,
        dims,
        strides,
        el_to_sum,
        src,
        offset,
        tid
    );

    // Complete reduction
    value = block_reduce(value, tid);

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
        shared,                                         \
        tid,                                            \
        dst_id);                                        \
    break;                                              \
}

#define impl_arg_reduce_inner(OP, NAME, T, NAME_SUFFIX) \
kernel void NAME##NAME_SUFFIX(                          \
    constant uint &num_dims,                            \
    constant uint &el_to_sum_per_block,                 \
    device const T *src,                                \
    device uint *dst,                                   \
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
    constant uint &num_dims,                            \
    constant size_t *dims,                              \
    constant size_t *strides,                           \
    constant uint &el_to_sum_per_block,                 \
    device const T *src,                                \
    device uint *dst,                                   \
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
impl_arg_reduce_inner(OP, NAME, ARG(vec<T, 2>), x2)     \
impl_arg_reduce_inner(OP, NAME, ARG(vec<T, 4>), x4)

// Contains the intermediate results for the online softmax calculation.
// m: max
// d: sum of the exponentials
template <typename T>
struct MD {
    T m;
    T d;

    constexpr MD<T>() = default;
    constexpr MD<T>() threadgroup = default;

    static constant constexpr uint N = vec_elements<T>::value;

    // Return 1-dimensional indexed value
    constexpr MD<make_scalar_t<T>> operator[](uint n) {
        assert(n < N);
        return MD<make_scalar_t<T>>{ m[n], d[n] };
    }
    constexpr const MD<make_scalar_t<T>> operator[](uint n) const {
        assert(n < N);
        return MD<make_scalar_t<T>>{ m[n], d[n] };
    }
};

// Enable operations for softmax MD
template<typename OP, typename T>
struct operation<OP, MD<T>, typename metal::enable_if_t<is_scalar_v<T>>> {
    OP op;

    METAL_FUNC MD<T> operator()(MD<T> a, MD<T> b) {
        return op(a, b);
    }
    METAL_FUNC MD<T> operator()(MD<T> a, T b, uint idx) {
        return this->operator()(a, MD<T>{ b, static_cast<T>(1.0) });
    }
};

// Specialization for indexed vector values.
template<typename OP, typename T>
struct operation<OP, MD<T>, typename metal::enable_if_t<is_vector_v<T>>> {
    OP op;

    METAL_FUNC MD<T> operator()(MD<T> a, MD<T> b) {
        #pragma clang loop unroll(full)
        for (ushort n = 0; n < vec_elements<T>::value; n++) {
            a[n] = op(a[n], b[n]);
        }
        return a;
    }
    METAL_FUNC MD<T> operator()(MD<T> a, T b, uint idx) {
        #pragma clang loop unroll(full)
        for (ushort n = 0; n < vec_elements<T>::value; n++) {
            a[n] = op(a[n], MD<make_scalar_t<T>>{ b[n], static_cast<make_scalar_t<T>>(1.0) });
        }
        return a;
    }
};

template<typename OP, typename T, typename _E = typename metal::enable_if_t<is_scalar_v<T>>>
METAL_FUNC MD<T> finalize(MD<T> value) {
    return value;
}

template<typename OP, typename T, typename _E = typename metal::enable_if_t<is_vector_v<T>>>
METAL_FUNC MD<make_scalar_t<T>> finalize(MD<T> value) {
    OP op;
    MD<make_scalar_t<T>> result = value[0];
    #pragma clang loop unroll(full)
    for (ushort n = 1; n < vec_elements<T>::value; n++) {
        result = op(result, value[n]);
    }
    return result;
}

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

template<typename T>
struct MDReduceOp {
    Exp exp;

    static constexpr METAL_FUNC MD<T> init() {
        return MD<T>{ numeric_limits<T>::lowest(), 0 };
    }

    METAL_FUNC MD<T> operator()(MD<T> a, MD<T> b) {
        bool a_bigger = a.m > b.m;
        MD<T> bigger_m = a_bigger ? a : b;
        MD<T> smaller_m = a_bigger ? b : a;
        MD<T> res;
        res.d = bigger_m.d + smaller_m.d * exp(smaller_m.m - bigger_m.m);
        res.m = bigger_m.m;
        return res;
    }

    METAL_FUNC MD<T> simd_op(MD<T> a) {
        return a;
    }
};
template<
    typename T,
    ushort BLOCKSIZE,
    typename _E = void
>
struct finalize_softmax;

template<typename T, ushort BLOCKSIZE>
struct finalize_softmax<T, BLOCKSIZE, typename metal::enable_if_t<is_scalar_v<T>>> {
    Divide divide;
    Exp exp;

    METAL_FUNC void operator()(
        const device T *src,
        device T *dst,
        threadgroup MD<T> &md_total,
        const uint thread_id,
        const uint stop_idx
    ) {
        const T d_total_inverse = divide(static_cast<T>(1.0), md_total.d);
        for (uint idx = thread_id; idx < stop_idx; idx += BLOCKSIZE) {
            dst[idx] = exp(src[idx] - md_total.m) * d_total_inverse;
        }
    }
};


template<typename T, ushort BLOCKSIZE>
struct finalize_softmax<T, BLOCKSIZE, typename metal::enable_if_t<is_vector_v<T>>> {
    Divide divide;
    Exp exp;

    METAL_FUNC void operator()(
        const device T *src,
        device make_scalar_t<T> *dst,
        threadgroup MD<make_scalar_t<T>> &md_total,
        const uint thread_id,
        const uint stop_idx
    ) {
        const make_scalar_t<T> d_total_inverse = divide(static_cast<make_scalar_t<T>>(1.0), md_total.d);
        for (uint idx = thread_id; idx < stop_idx; idx += BLOCKSIZE) {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < granularity<T>(); i++) {
                dst[idx + i] = exp(src[idx][i] - md_total.m) * d_total_inverse;
            }
        }
    }
};

// Welford's algorithm approach for an online softmax implementation.
// Same as the Online normalizer calculation for softmax: https://arxiv.org/pdf/1805.02867.pdf
template<typename T, ushort BLOCKSIZE>
METAL_FUNC void softmax(
    constant uint &src_numel,
    constant uint &el_to_sum_per_block,
    const device T *src,
    device make_scalar_t<T> *dst,
    threadgroup MD<T> shared[BLOCKSIZE],
    threadgroup MD<make_scalar_t<T>> &md_total,

    ushort tid [[ thread_index_in_threadgroup ]],
    ushort dst_id [[ threadgroup_position_in_grid ]]
) {
    loader<T, MD<T>, MDReduceOp<make_scalar_t<T>>, BLOCKSIZE> load;
    block_reducer<MD<T>, MDReduceOp<make_scalar_t<T>>, BLOCKSIZE> block_reduce(shared);
    finalize_softmax<T, BLOCKSIZE> softmax_finalize;

    // Calcluate offset for the threadgroup of current thread;
    const uint el_to_sum = el_to_sum_per_block;
    const uint offset = dst_id * el_to_sum;

    // Calculate partial result for current thread
    MD<T> md_partial = MD<T> { numeric_limits<T>::lowest(), 0 };
    md_partial = load(
        md_partial,
        src_numel,
        el_to_sum_per_block,
        src,
        offset,
        tid
    );

    // Reduce in shared memory
    MD<T> md = block_reduce(md_partial, tid);

    if (tid == 0) md_total = finalize<MDReduceOp<make_scalar_t<T>>>(md);
    threadgroup_barrier(mem_flags::mem_none);

    // Finalize softmax
    const uint thread_id = tid + offset;
    const uint stop_idx = min(el_to_sum_per_block + offset, src_numel);
    softmax_finalize(src, dst, md_total, thread_id, stop_idx);
}

#define softmax_case(T, N)                              \
case N: {                                               \
    if (N / GRANULARITY == 0) {                         \
        break;                                          \
    }                                                   \
    constexpr uint B = nonzero<N / GRANULARITY>();      \
    threadgroup MD<T> shared[B];                        \
    threadgroup MD<make_scalar_t<T>> md_total;          \
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
    constant uint &src_numel,                           \
    constant uint &el_to_sum_per_block,                 \
    device const T *src,                                \
    device make_scalar_t<T> *dst,                       \
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

#define impl_softmax(NAME, T)                           \
impl_softmax_inner(NAME, T)                             \
impl_softmax_inner(NAME##x2, T##2)                      \
impl_softmax_inner(NAME##x4, T##4)

impl_reduce(Sum, fast_sum_f32, float)
//impl_reduce(Sum, fast_sum_u32, uint)
//impl_reduce(Sum, fast_sum_f16, half)
//impl_reduce(Sum, fast_sum_u8, uint8_t)
//
//impl_reduce(Mul, fast_mul_f32, float)
//impl_reduce(Mul, fast_mul_u32, uint)
//impl_reduce(Mul, fast_mul_f16, half)
//impl_reduce(Mul, fast_mul_u8, uint8_t)

impl_reduce(Max, fast_max_f32, float)
//impl_reduce(Max, fast_max_u32, uint)
//impl_reduce(Max, fast_max_f16, half)
//impl_reduce(Max, fast_max_u8, uint8_t)
//
//impl_reduce(Min, fast_min_f32, float)
//impl_reduce(Min, fast_min_u32, uint)
//impl_reduce(Min, fast_min_f16, half)
//impl_reduce(Min, fast_min_u8, uint8_t)
//
//impl_arg_reduce(Min, fast_argmin_f32, float)
//impl_arg_reduce(Min, fast_argmin_f16, half)
//impl_arg_reduce(Min, fast_argmin_u32, uint)
//impl_arg_reduce(Min, fast_argmin_u8, uint8_t)

impl_arg_reduce(Max, fast_argmax_f32, float)
//impl_arg_reduce(Max, fast_argmax_f16, half)
//impl_arg_reduce(Max, fast_argmax_u32, uint)
//impl_arg_reduce(Max, fast_argmax_u8, uint8_t)

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
//impl_reduce(Sum, fast_sum_bf16, bfloat)
//impl_reduce(Mul, fast_mul_bf16, bfloat)
//impl_reduce(Max, fast_max_bf16, bfloat)
//impl_reduce(Min, fast_min_bf16, bfloat)
//
//impl_arg_reduce(Min, fast_argmin_bf16, bfloat)
//impl_arg_reduce(Max, fast_argmax_bf16, bfloat)

impl_softmax(softmax_bf16, bfloat)
#endif
