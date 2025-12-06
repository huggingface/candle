#include <metal_stdlib>
#include <metal_math>
using namespace metal;


// Utils
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

#define MAX(x, y) ((x) > (y) ? (x) : (y))

template<typename T>
constexpr int work_per_thread() {
    constexpr int wpt = 8 / sizeof(T);
    return MAX(1, wpt);
}

// Kernels
template <typename T, typename U, typename unary, int W = work_per_thread<T>()>
[[kernel]] void unary_kernel(
    constant size_t &dim,
    device const T* input,
    device U* output,
    uint tid [[thread_position_in_grid]]
) {
    tid *= W;
    if (W > 1 && tid + W > dim) {
        for (int i = 0; tid + i < dim; ++i) {
            output[tid + i] = static_cast<U>(unary()(input[tid + i]));
        }
    } else {
        for (int i = 0; i < W; ++i) {
            output[tid + i] = static_cast<U>(unary()(input[tid + i]));
        }
    }
}

template <typename T, typename U, typename unary>
[[kernel]] void unary_kernel_strided(
    constant size_t &dim,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides,
    constant const T *input,
    device U *output,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= dim) return;
    uint idx = get_strided_index(tid, num_dims, dims, strides);
    output[tid] = static_cast<U>(unary()(input[idx]));
}

template <typename T, int W = work_per_thread<T>()>
[[kernel]] void const_set(
    constant size_t &dim,
    device const T &input,
    device T *output,
    uint tid [[thread_position_in_grid]]
) {
    tid *= W;
    if (W > 1 && tid + W > dim) {
        for (int i = 0; tid + i < dim; ++i) {
            output[tid + i] = input;
        }
    } else {
        for (int i = 0; i < W; ++i) {
            output[tid + i] = input;
        }
    }
}

template <typename T>
[[kernel]] void const_set_strided(
    constant size_t &dim,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides,
    device const T &input,
    device T *output,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= dim) {
        return;
    }
    uint idx = get_strided_index(tid, num_dims, dims, strides);
    output[idx] = input;
}

template <typename T>
[[kernel]] void copy2d(
    constant int64_t &d1,
    constant int64_t &d2,
    constant int64_t &src_s,
    constant int64_t &dst_s,
    device const T *input,
    device T *output,
    uint2 idx [[thread_position_in_grid]]
) {
    if (idx.x >= d1 || idx.y >= d2) return;
    int64_t src_idx = idx.x * src_s + idx.y;
    int64_t dst_idx = idx.x * dst_s + idx.y;
    output[dst_idx] = input[src_idx];
}

// Unary functions
template <typename T> METAL_FUNC T erf(T in){
    // constants
    constexpr const float a1 =  0.254829592;
    constexpr const float a2 = -0.284496736;
    constexpr const float a3 =  1.421413741;
    constexpr const float a4 = -1.453152027;
    constexpr const float a5 =  1.061405429;
    constexpr const float p  =  0.3275911;

    float x = static_cast<float>(in);

    // Save the sign of x
    int sign = 1;
    if (x < 0)
        sign = -1;
    x = fabs(x);

    // A&S formula 7.1.26
    float t = 1.0/(1.0 + p*x);
    float y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);

    return T(sign*y);
}
template <typename T> METAL_FUNC T id(T in) { return in; }
template <typename T> METAL_FUNC T gelu_erf(T x) {
    return static_cast<T>(x * (1 + erf(x * M_SQRT1_2_F)) / 2);
}
template <typename T> METAL_FUNC T gelu(T x) {
    if (x > 5) {
        return x;
    }
    T x_sq = x * x;
    T x_cube = x_sq * x;
    T alpha = x + static_cast<T>(0.044715) * x_cube;
    T beta =  (static_cast<T>(M_2_SQRTPI_F * M_SQRT1_2_F) * alpha);
    return static_cast<T>(0.5) * x * (static_cast<T>(1.0) + T(precise::tanh(beta)));
}
template <typename T> METAL_FUNC T relu(T x) {
    if (x > 5) {
        return x;
    }
    T x_sq = x * x;
    T x_cube = x_sq * x;
    T alpha = x + static_cast<T>(0.044715) * x_cube;
    T beta =  (static_cast<T>(M_2_SQRTPI_F * M_SQRT1_2_F) * alpha);
    return static_cast<T>(0.5) * x * (static_cast<T>(1.0) + T(precise::tanh(beta)));
}
template <typename T> METAL_FUNC T recip(T x) {
    return static_cast<T>(1.0 / x);
}
template <typename T> METAL_FUNC T sigmoid(T x) {
    return static_cast<T>(recip(1 + exp(-x)));
}

// Define unary ops
#define define_unary_op(name, op)   \
struct name {                       \
    template <typename T>           \
    METAL_FUNC T operator()(T x) {  \
        return static_cast<T>(op);  \
    }                               \
};

define_unary_op(usqr, x * x);
define_unary_op(urecip, recip(x));
define_unary_op(uneg, -x);
define_unary_op(uid, x);
define_unary_op(ugelu, gelu(x));
define_unary_op(urelu, x < 0 ? 0 : x);
define_unary_op(usilu, x / (1 + exp(-x)));
define_unary_op(ugelu_erf, gelu_erf(x));
define_unary_op(usqrt, sqrt(x));
define_unary_op(ucos, cos(x));
define_unary_op(usin, sin(x));
define_unary_op(uexp, exp(x));
define_unary_op(ulog, log(x));
define_unary_op(uabs, abs(static_cast<float>(x)));
define_unary_op(uceil, ceil(x));
define_unary_op(ufloor, floor(x));
define_unary_op(uround, round(x));
define_unary_op(uerf, erf(x));
define_unary_op(usign, sign(x));
define_unary_op(usigmoid, sigmoid(x));
// tanh may create NaN on large values, e.g. 45 rather than outputting 1.
// This has been an issue for the encodec example.
define_unary_op(utanh, precise::tanh(x));

// Macros to help initialize kernels
#define init_kernel(name, func, ...) \
  template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

#define init_unary(op_name, unary_op, tname, t)                                         \
    init_kernel(#op_name "_" #tname, unary_kernel, t, t, unary_op)                      \
    init_kernel(#op_name "_" #tname "_strided", unary_kernel_strided, t, t, unary_op)

#if defined(__HAVE_BFLOAT__)
#define init_unary_float(op_name, unary_op)   \
    init_unary(op_name, unary_op, f32, float) \
    init_unary(op_name, unary_op, f16, half)  \
    init_unary(op_name, unary_op, bf16, bfloat)
#else
#define init_unary_float(op_name, unary_op)   \
    init_unary(op_name, unary_op, f32, float) \
    init_unary(op_name, unary_op, f16, half)
#endif

#define init_copy2d(tname, t)  \
    init_kernel("copy2d_" #tname, copy2d, t)

#define init_const_set(tname, t)                    \
    init_kernel("const_set_" #tname, const_set, t)  \
    init_kernel("const_set_" #tname "_strided", const_set_strided, t)

// Initialize all unary kernels for floating point types
init_unary_float(gelu_erf, ugelu_erf);
init_unary_float(sqrt, usqrt);
init_unary_float(sqr, usqr);
init_unary_float(neg, uneg);
init_unary_float(recip, urecip);
init_unary_float(copy, uid);
init_unary_float(silu, usilu);
init_unary_float(gelu, ugelu);
init_unary_float(relu, urelu);
init_unary_float(cos, ucos);
init_unary_float(sin, usin);
init_unary_float(exp, uexp);
init_unary_float(log, ulog);
init_unary_float(abs, uabs);
init_unary_float(ceil, uceil);
init_unary_float(floor, ufloor);
init_unary_float(round, uround);
init_unary_float(erf, uerf);
init_unary_float(sign, usign);
init_unary_float(sigmoid, usigmoid);
init_unary_float(tanh, utanh);

// Initialize copy2d kernels
init_copy2d(f32, float);
init_copy2d(f16, half);

// Initialize const_set kernels
init_const_set(f32, float);
init_const_set(f16, half);

#if defined(__HAVE_BFLOAT__)
init_copy2d(bf16, bfloat);
init_const_set(bf16, bfloat);
#endif

// Initialize unary kernels for integer dtypes
init_unary(copy, uid, u8, uint8_t);
init_unary(copy, uid, u32, uint32_t);

init_copy2d(u8, uint8_t);
init_copy2d(u32, uint32_t);

init_const_set(u8, uint8_t);
init_const_set(u32, uint32_t);

#if __METAL_VERSION__ >= 220
init_unary(copy, uid, i64, int64_t);
init_copy2d(i64, int64_t);
init_const_set(i64, int64_t);
#endif
