#include <metal_stdlib>
using namespace metal;

// Utils
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

struct cont_indexer {
    METAL_FUNC uint operator()(
        uint idx,
        constant size_t &num_dims,
        constant size_t *dims,
        constant size_t *strides
    ) {
        return idx;
    }
};

struct strided_indexer {
    METAL_FUNC uint operator()(
        uint idx,
        constant size_t &num_dims,
        constant size_t *dims,
        constant size_t *strides
    ) {
        return get_strided_index(idx, num_dims, dims, strides);
    }
};

template<uint Y>
constexpr uint div_ceil(uint x) {
    return x / Y + (x % Y > 0);
}

template<uint X, uint Y>
constexpr uint div_ceil() {
    return X / Y + (X % Y > 0);
}

template<typename T>
constexpr uint work_per_thread() {
    return div_ceil<8, sizeof(T)>();
}

// Kernels
template <typename T, typename U, typename binary, uint W = work_per_thread<T>()>
[[kernel]] void binary_kernel(
    constant size_t &dim,
    device const T *left,
    device const T *right,
    device U *output,
    uint tid [[thread_position_in_grid]]
) {
    binary op;
    const uint step = div_ceil<W>(dim);
    #pragma clang loop unroll(full)
    for (uint i = tid; i < dim; i += step) {
        output[i] = static_cast<U>(op(left[i], right[i]));
    }
}

template <
    typename T,
    typename U,
    typename binary,
    typename l_indexer = strided_indexer,
    typename r_indexer = strided_indexer,
    uint W = work_per_thread<T>()>
[[kernel]] void binary_kernel_strided(
    constant size_t &dim,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *left_strides,
    constant size_t *right_strides,
    device const T *left,
    device const T *right,
    device U *output,
    uint tid [[ thread_position_in_grid ]]
) {
    binary op;
    l_indexer l_index;
    r_indexer r_index;
    const uint step = div_ceil<W>(dim);
    #pragma clang loop unroll(full)
    for (uint i = tid; i < dim; i += step) {
        uint l_idx = l_index(i, num_dims, dims, left_strides);
        uint r_idx = r_index(i, num_dims, dims, right_strides);
        output[i] = static_cast<U>(op(left[l_idx], right[r_idx]));
    }
}

// Macros to help initialize kernels
#define init_kernel(name, func, ...) \
  template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

#define init_binary_k(op_name, binary_op, tname, t, u)                                                                      \
    init_kernel(#op_name "_" #tname, binary_kernel, t, u, binary_op)                                                        \
    init_kernel(#op_name "_" #tname "_strided", binary_kernel_strided, t, u, binary_op)                                     \
    init_kernel(#op_name "_" #tname "_lstrided", binary_kernel_strided, t, u, binary_op, strided_indexer, cont_indexer)     \
    init_kernel(#op_name "_" #tname "_rstrided", binary_kernel_strided, t, u, binary_op, cont_indexer, strided_indexer)

#if defined(__HAVE_BFLOAT__)
#define init_binary(bop)                            \
    init_binary_k(bop, bop, f32, float, float)      \
    init_binary_k(bop, bop, f16, half, half)        \
    init_binary_k(bop, bop, bf16, bfloat, bfloat)   \
    init_binary_k(bop, bop, u8, uint8_t, uint8_t)   \
    init_binary_k(bop, bop, u32, uint32_t, uint32_t)\
    init_binary_k(bop, bop, i64, int64_t, int64_t)
#else
#define init_binary(bop)                                                       \
    init_binary_k(bop, bop, f32, float, float)      \
    init_binary_k(bop, bop, f16, half, half)        \
    init_binary_k(bop, bop, u8, uint8_t, uint8_t)   \
    init_binary_k(bop, bop, u32, uint32_t, uint32_t)\
    init_binary_k(bop, bop, i64, int64_t, int64_t)
#endif

#if defined(__HAVE_BFLOAT__)
#define init_boolean_binary(op_name, binary_op)             \
    init_binary_k(op_name, binary_op, f32, float, bool)     \
    init_binary_k(op_name, binary_op, f16, half, bool)      \
    init_binary_k(op_name, binary_op, bf16, bfloat, bool)   \
    init_binary_k(op_name, binary_op, u8, uint8_t, bool)    \
    init_binary_k(op_name, binary_op, u32, uint32_t, bool)  \
    init_binary_k(op_name, binary_op, i64, int64_t, bool)
#else
#define init_boolean_binary(op_name, binary_op)             \
    init_binary_k(op_name, binary_op, f32, float, bool)     \
    init_binary_k(op_name, binary_op, f16, half, bool)      \
    init_binary_k(op_name, binary_op, u8, uint8_t, bool)    \
    init_binary_k(op_name, binary_op, u32, uint32_t, bool)  \
    init_binary_k(op_name, binary_op, i64, int64_t, bool)
#endif

// Define binary ops
#define define_binary_op(name, op)      \
struct name {                           \
    template <typename T>               \
    METAL_FUNC T operator()(T x, T y) { \
        return static_cast<T>(op);      \
    }                                   \
};
#define define_binary_bool_op(name, op)     \
struct name {                               \
    template <typename T>                   \
    METAL_FUNC bool operator()(T x, T y) {  \
        return op;                          \
    }                                       \
};

// Define binary ops
define_binary_op(badd, x + y);
define_binary_op(bsub, x - y);
define_binary_op(bmul, x * y);
define_binary_op(bdiv, x / y);
define_binary_op(bminimum, MIN(x, y));
define_binary_op(bmaximum, MAX(x, y));

// Define binary ops that return a bool
define_binary_bool_op(beq, x == y);
define_binary_bool_op(bne, x != y);
define_binary_bool_op(ble, x <= y);
define_binary_bool_op(blt, x < y);
define_binary_bool_op(bge, x >= y);
define_binary_bool_op(bgt, x > y)

// Initialize kernels
init_binary(badd);
init_binary(bsub);
init_binary(bmul);
init_binary(bdiv);
init_binary(bminimum);
init_binary(bmaximum);

init_boolean_binary(eq, beq);
init_boolean_binary(ne, bne);
init_boolean_binary(le, ble);
init_boolean_binary(lt, blt);
init_boolean_binary(ge, bge);
init_boolean_binary(gt, bgt);
