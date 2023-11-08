#include <metal_stdlib>

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
template <typename T> METAL_FUNC T neg(T in){ return -in; }
template <typename T> METAL_FUNC T id(T in){ return in; }


using namespace metal;

#define UNARY(FN, TYPENAME, FN_NAME, FN_NAME_STRIDED) \
kernel void FN_NAME( \
    constant size_t &dim, \
    device const TYPENAME *input,  \
    device TYPENAME *output, \
    uint threadgroup_size [[threads_per_threadgroup]], \
    uint thread_index [[thread_index_in_threadgroup]] \
) { \
    const size_t length = (dim  + threadgroup_size - 1) / threadgroup_size; \
    const size_t start = thread_index * length; \
    const size_t stop = min(start + length, dim); \
    for (size_t i = start; i < stop; i++){ \
        output[i] = TYPENAME(FN(input[i])); \
    } \
}\
kernel void FN_NAME_STRIDED( \
    constant size_t &dim, \
    constant size_t &num_dims, \
    constant size_t *dims, \
    constant size_t *strides, \
    device const TYPENAME *input,  \
    device TYPENAME *output, \
    uint threadgroup_size [[threads_per_threadgroup]], \
    uint thread_index [[thread_index_in_threadgroup]] \
) { \
    const size_t length = (dim  + threadgroup_size - 1) / threadgroup_size; \
    const size_t start = thread_index * length; \
    const size_t stop = min(start + length, dim); \
    for (size_t i = start; i < stop; i++){ \
        output[i] = TYPENAME(FN(input[get_strided_index(i, num_dims, dims, strides)])); \
    } \
}

#define UNARY_OP(NAME) \
UNARY(NAME, float, NAME##_float, NAME##_float_strided); \
UNARY(NAME, half, NAME##_half, NAME##_half_strided);

#define BFLOAT_UNARY_OP(NAME) \
UNARY(NAME, bfloat, NAME##_bfloat, NAME##_bfloat_strided);


UNARY_OP(cos)
UNARY_OP(sin)
UNARY_OP(sqr)
UNARY_OP(sqrt)
UNARY_OP(neg)
UNARY_OP(exp)
UNARY(id, float, copy_float, copy_float_strided)
UNARY(id, half, copy_half, copy_half_strided)

#if __METAL_VERSION__ >= 310
BFLOAT_UNARY_OP(cos)
BFLOAT_UNARY_OP(sin)
BFLOAT_UNARY_OP(sqr)
BFLOAT_UNARY_OP(sqrt)
BFLOAT_UNARY_OP(neg)
BFLOAT_UNARY_OP(exp)
#endif
