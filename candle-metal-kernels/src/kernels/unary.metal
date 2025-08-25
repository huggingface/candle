#include <metal_stdlib>
#include <metal_math>
#
using namespace metal;

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

template <typename T> METAL_FUNC T sqr(T in){ return in * in; }
template <typename T> METAL_FUNC T recip(T in){ return T(1.0 / in); }
template <typename T> METAL_FUNC T neg(T in){ return -in; }

template <typename T> METAL_FUNC T erf(T in){
    float x = (float) in;
    // constants
    float a1 =  0.254829592;
    float a2 = -0.284496736;
    float a3 =  1.421413741;
    float a4 = -1.453152027;
    float a5 =  1.061405429;
    float p  =  0.3275911;

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
    return T(x * (1 + erf(x * M_SQRT1_2_F)) / 2);
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
template <typename T> METAL_FUNC T relu(T in){
    if (in < 0) {
        return 0;
    }
    return in;
}
template <typename T> METAL_FUNC T silu(T in){
    return in / (static_cast<T>(1) + exp(-in));
}
template <typename T> METAL_FUNC T sigmoid(T in) {
    return recip(static_cast<T>(1) + exp(-in));
}

#define TILE_SIZE 2

#define CONST_SET(TYPENAME, FN_NAME) \
kernel void FN_NAME( \
    constant size_t &dim, \
    constant TYPENAME &input,  \
    device TYPENAME *output, \
    uint tid [[ thread_position_in_grid ]] \
) { \
    if (tid >= dim) { \
        return; \
    } \
    output[tid] = input; \
} \
kernel void FN_NAME##_##strided( \
    constant size_t &dim, \
    constant size_t &num_dims, \
    constant size_t *dims, \
    constant size_t *strides, \
    constant TYPENAME &input,  \
    device TYPENAME *output, \
    uint tid [[ thread_position_in_grid ]] \
) { \
    if (tid >= dim) { \
        return; \
    } \
    output[get_strided_index(tid, num_dims, dims, strides)] = input; \
} \
kernel void FN_NAME##_##tiled( \
    constant size_t &dim, \
    constant TYPENAME &input,  \
    device TYPENAME *output, \
    uint tid [[ thread_position_in_grid ]] \
) { \
    for (uint i = 0; i < TILE_SIZE; i++) { \
        const uint idx = tid * TILE_SIZE + i; \
        output[idx] = input; \
    } \
}

#define UNARY(FN, TYPENAME, FN_NAME, FN_NAME_STRIDED) \
kernel void FN_NAME( \
    constant size_t &dim, \
    device const TYPENAME *input,  \
    device TYPENAME *output, \
    uint tid [[ thread_position_in_grid ]] \
) { \
    if (tid >= dim) { \
        return; \
    } \
    output[tid] = TYPENAME(FN(float(input[tid]))); \
} \
kernel void FN_NAME##_##strided( \
    constant size_t &dim, \
    constant size_t &num_dims, \
    constant size_t *dims, \
    constant size_t *strides, \
    device const TYPENAME *input,  \
    device TYPENAME *output, \
    uint tid [[ thread_position_in_grid ]] \
) { \
    if (tid >= dim) { \
        return; \
    } \
    output[tid] = TYPENAME(FN(float(input[get_strided_index(tid, num_dims, dims, strides)]))); \
} \
kernel void FN_NAME##_##tiled( \
    constant size_t &dim, \
    device const TYPENAME *input,  \
    device TYPENAME *output, \
    uint tid [[ thread_position_in_grid ]] \
) { \
    for (uint i = 0; i < TILE_SIZE; i++) { \
        const uint idx = tid * TILE_SIZE + i; \
        output[idx] = TYPENAME(FN(float(input[idx]))); \
    } \
}

#define UNARY_OP(NAME) \
UNARY(NAME, float, NAME##_f32, NAME##_f32_strided); \
UNARY(NAME, half, NAME##_f16, NAME##_f16_strided);

#define BFLOAT_UNARY_OP(NAME) \
UNARY(NAME, bfloat, NAME##_bf16, NAME##_bf16_strided);

#define COPY2D(FN_NAME, TYPENAME) \
kernel void FN_NAME( \
    constant int64_t &d1, \
    constant int64_t &d2, \
    constant int64_t &src_s, \
    constant int64_t &dst_s, \
    device const TYPENAME *input,  \
    device TYPENAME *output, \
    uint2 idx [[thread_position_in_grid]] \
) { \
    if (idx.x >= d1 || idx.y >= d2) return; \
    int64_t src_idx = idx.x * src_s + idx.y; \
    int64_t dst_idx = idx.x * dst_s + idx.y; \
    output[dst_idx] = input[src_idx]; \
}

COPY2D(copy2d_f32, float)
COPY2D(copy2d_f16, half)
COPY2D(copy2d_u8, uint8_t)
COPY2D(copy2d_u32, uint32_t)

CONST_SET(float, const_set_f32)
CONST_SET(half, const_set_f16)
CONST_SET(uint8_t, const_set_u8)
CONST_SET(uint32_t, const_set_u32)

UNARY_OP(cos)
UNARY_OP(sin)
UNARY_OP(sqr)
UNARY_OP(sqrt)
UNARY_OP(neg)
UNARY_OP(exp)
UNARY_OP(log)
UNARY_OP(gelu)
UNARY_OP(silu)
UNARY_OP(abs)
UNARY_OP(ceil)
UNARY_OP(floor)
UNARY_OP(round)
UNARY_OP(gelu_erf)
UNARY_OP(erf)
UNARY_OP(recip)
UNARY_OP(relu)
UNARY_OP(sign)
UNARY_OP(sigmoid)
UNARY(id, float, copy_f32, copy_f32_strided)
UNARY(id, half, copy_f16, copy_f16_strided)
UNARY(id, uint8_t, copy_u8, copy_u8_strided)
UNARY(id, uint32_t, copy_u32, copy_u32_strided)

// tanh may create NaN on large values, e.g. 45 rather than outputing 1.
// This has been an issue for the encodec example.
UNARY(precise::tanh, float, tanh_f32, tanh_f32_strided);
UNARY(precise::tanh, half, tanh_f16, tanh_f16_strided);

#if __METAL_VERSION__ >= 220
UNARY(id, int64_t, copy_i64, copy_i64_strided)
COPY2D(copy2d_i64, int64_t)
CONST_SET(int64_t, const_set_i64)
#endif

#if defined(__HAVE_BFLOAT__)
BFLOAT_UNARY_OP(cos)
BFLOAT_UNARY_OP(sin)
BFLOAT_UNARY_OP(sqr)
BFLOAT_UNARY_OP(sqrt)
BFLOAT_UNARY_OP(neg)
BFLOAT_UNARY_OP(exp)
BFLOAT_UNARY_OP(log)
BFLOAT_UNARY_OP(gelu)
BFLOAT_UNARY_OP(silu)
BFLOAT_UNARY_OP(abs)
BFLOAT_UNARY_OP(ceil)
BFLOAT_UNARY_OP(floor)
BFLOAT_UNARY_OP(round)
BFLOAT_UNARY_OP(gelu_erf)
BFLOAT_UNARY_OP(erf)
BFLOAT_UNARY_OP(recip)
BFLOAT_UNARY_OP(relu)
BFLOAT_UNARY_OP(sign)
BFLOAT_UNARY_OP(sigmoid)

UNARY(id, bfloat, copy_bf16, copy_bf16_strided)

UNARY(precise::tanh, bfloat, tanh_bf16, tanh_bf16_strided);

COPY2D(copy2d_bf16, bfloat)
CONST_SET(bfloat, const_set_bf16)
#endif
