#include <metal_stdlib>
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

template<typename T, typename ID>
METAL_FUNC void where_cond(
    constant size_t &numel,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides,
    constant size_t *strides_t,
    constant size_t *strides_f,
    device const ID *ids,
    device const T *t,
    device const T *f,
    device T *out,
    uint i [[ thread_position_in_grid ]]
) {
    if (i >= numel){
       return;
    }
    uint strided_i = get_strided_index(i, num_dims, dims, strides);
    uint strided_i_t = get_strided_index(i, num_dims, dims, strides_t);
    uint strided_i_f = get_strided_index(i, num_dims, dims, strides_f);
    out[i] = ids[strided_i] ? t[strided_i_t] : f[strided_i_f];
}

#define WHERE_OP(T, ID, FN_NAME)                                                                \
kernel void FN_NAME(                                                                            \
    constant size_t &numel,                                                                     \
    constant size_t &num_dims,                                                                  \
    constant size_t *dims,                                                                      \
    constant size_t *strides,                                                                   \
    constant size_t *strides_t,                                                                 \
    constant size_t *strides_f,                                                                 \
    device const ID *ids,                                                                       \
    device const T *t,                                                                          \
    device const T *f,                                                                          \
    device T *out,                                                                              \
    uint i [[ thread_position_in_grid ]]                                                        \
) {                                                                                             \
   where_cond<T, ID>(numel, num_dims, dims, strides, strides_t, strides_f, ids, t, f, out, i);  \
}                                                                                               \

WHERE_OP(half, uint32_t, where_u32_f16)
WHERE_OP(float, uint32_t, where_u32_f32)
WHERE_OP(uint8_t, uint32_t, where_u32_u8)
WHERE_OP(uint32_t, uint32_t, where_u32_u32)

WHERE_OP(half, uint8_t, where_u8_f16)
WHERE_OP(float, uint8_t, where_u8_f32)
WHERE_OP(uint8_t, uint8_t, where_u8_u8)
WHERE_OP(uint32_t, uint8_t, where_u8_u32)

#if __METAL_VERSION__ >= 220
WHERE_OP(int64_t, uint8_t, where_u8_i64)
WHERE_OP(int64_t, uint32_t, where_u32_i64)

WHERE_OP(half, int64_t, where_i64_f16)
WHERE_OP(float, int64_t, where_i64_f32)
WHERE_OP(uint8_t, int64_t, where_i64_u8)
WHERE_OP(uint32_t, int64_t, where_i64_u32)
WHERE_OP(int64_t, int64_t, where_i64_i64)
#if defined(__HAVE_BFLOAT__)
WHERE_OP(bfloat, int64_t, where_i64_bf16)
#endif
#endif

#if defined(__HAVE_BFLOAT__)
WHERE_OP(bfloat, uint8_t, where_u8_bf16)
WHERE_OP(bfloat, uint32_t, where_u32_bf16)
#endif
