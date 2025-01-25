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

template<typename TYPENAME, typename INDEX_TYPENAME>
METAL_FUNC void index(
    constant size_t &dst_size,
    constant size_t &left_size,
    constant size_t &src_dim_size,
    constant size_t &right_size,
    constant size_t &ids_size,
    constant bool &contiguous,
    constant size_t *src_dims,
    constant size_t *src_strides,
    const device TYPENAME *input,
    const device INDEX_TYPENAME *input_ids,
    device TYPENAME *output,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= dst_size) {
        return;
    }
    const size_t id_i = (tid / right_size) % ids_size;
    const INDEX_TYPENAME input_i = min(input_ids[id_i], (INDEX_TYPENAME)(src_dim_size - 1));
    const size_t right_rank_i = tid % right_size;
    const size_t left_rank_i = tid / right_size / ids_size;
    /*
    // Force prevent out of bounds indexing
    // since there doesn't seem to be a good way to force crash
    // No need to check for zero we're only allowing unsized.
    */
    const size_t src_i = left_rank_i * src_dim_size * right_size + input_i * right_size + right_rank_i;
    const size_t strided_src_i = contiguous ? src_i : get_strided_index(src_i, src_dim_size, src_dims, src_strides);
    output[tid] = input[strided_src_i];
}

# define INDEX_OP(NAME, INDEX_TYPENAME, TYPENAME) \
kernel void NAME( \
    constant size_t &dst_size, \
    constant size_t &left_size, \
    constant size_t &src_dim_size, \
    constant size_t &right_size, \
    constant size_t &ids_size, \
    constant bool &contiguous, \
    constant size_t *src_dims, \
    constant size_t *src_strides, \
    const device TYPENAME *input, \
    const device INDEX_TYPENAME *input_ids, \
    device TYPENAME *output, \
    uint tid [[ thread_position_in_grid ]] \
) { \
    index<TYPENAME, INDEX_TYPENAME>(dst_size, left_size, src_dim_size, right_size, ids_size, contiguous, src_dims, src_strides, input, input_ids, output, tid); \
}


template<typename TYPENAME, typename INDEX_TYPENAME>
METAL_FUNC void gather(
    constant size_t &dst_size,
    constant size_t &left_size,
    constant size_t &src_dim_size,
    constant size_t &right_size,
    constant size_t &ids_size,
    const device TYPENAME *input,
    const device INDEX_TYPENAME *input_ids,
    device TYPENAME *output,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= dst_size) {
        return;
    }
    const INDEX_TYPENAME input_i = input_ids[tid];
    const size_t right_rank_i = tid % right_size;
    const size_t left_rank_i = tid / right_size / ids_size;
    const size_t src_i = (left_rank_i * src_dim_size + input_i) * right_size + right_rank_i;
    output[tid] = input[src_i];
}

# define GATHER_OP(NAME, INDEX_TYPENAME, TYPENAME) \
kernel void NAME( \
    constant size_t &dst_size, \
    constant size_t &left_size, \
    constant size_t &src_dim_size, \
    constant size_t &right_size, \
    constant size_t &ids_size, \
    const device TYPENAME *input, \
    const device INDEX_TYPENAME *input_ids, \
    device TYPENAME *output, \
    uint tid [[ thread_position_in_grid ]] \
) { \
    gather<TYPENAME, INDEX_TYPENAME>(dst_size, left_size, src_dim_size, right_size, ids_size, input, input_ids, output, tid); \
}

template<typename TYPENAME, typename INDEX_TYPENAME>
METAL_FUNC void scatter_add(
    constant size_t &dst_size,
    constant size_t &left_size,
    constant size_t &src_dim_size,
    constant size_t &right_size,
    constant size_t &dst_dim_size,
    const device TYPENAME *input,
    const device INDEX_TYPENAME *input_ids,
    device TYPENAME *output,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= dst_size) {
        return;
    }
    const size_t right_rank_i = tid % right_size;
    const size_t left_rank_i = tid / right_size;
    for (unsigned int j = 0; j < src_dim_size; ++j) {
        const size_t src_i = (left_rank_i * src_dim_size + j) * right_size + right_rank_i;
        const INDEX_TYPENAME idx = input_ids[src_i];
        const size_t dst_i = (left_rank_i * dst_dim_size + idx) * right_size + right_rank_i;
        output[dst_i] += input[src_i];
    }
}

# define SCATTER_ADD_OP(NAME, INDEX_TYPENAME, TYPENAME) \
kernel void NAME( \
    constant size_t &dst_size, \
    constant size_t &left_size, \
    constant size_t &src_dim_size, \
    constant size_t &right_size, \
    constant size_t &dst_dim_size, \
    const device TYPENAME *input, \
    const device INDEX_TYPENAME *input_ids, \
    device TYPENAME *output, \
    uint tid [[ thread_position_in_grid ]] \
) { \
    scatter_add<TYPENAME, INDEX_TYPENAME>(dst_size, left_size, src_dim_size, right_size, dst_dim_size, input, input_ids, output, tid); \
}

template<typename TYPENAME, typename INDEX_TYPENAME>
METAL_FUNC void index_add(
    constant size_t &dst_size,
    constant size_t &left_size,
    constant size_t &src_dim_size,
    constant size_t &right_size,
    constant size_t &dst_dim_size,
    constant size_t &ids_dim_size,
    const device TYPENAME *input,
    const device INDEX_TYPENAME *input_ids,
    device TYPENAME *output,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= dst_size) {
        return;
    }
    const size_t right_rank_i = tid % right_size;
    const size_t left_rank_i = tid / right_size;
    for (unsigned int j = 0; j < ids_dim_size; ++j) {
        const INDEX_TYPENAME idx = input_ids[j];
        const size_t src_i = (left_rank_i * src_dim_size + j) * right_size + right_rank_i;
        const size_t dst_i = (left_rank_i * dst_dim_size + idx) * right_size + right_rank_i;
        output[dst_i] += input[src_i];
    }
}

# define INDEX_ADD_OP(NAME, INDEX_TYPENAME, TYPENAME) \
kernel void NAME( \
    constant size_t &dst_size, \
    constant size_t &left_size, \
    constant size_t &src_dim_size, \
    constant size_t &right_size, \
    constant size_t &dst_dim_size, \
    constant size_t &ids_dim_size, \
    const device TYPENAME *input, \
    const device INDEX_TYPENAME *input_ids, \
    device TYPENAME *output, \
    uint tid [[ thread_position_in_grid ]] \
) { \
    index_add<TYPENAME, INDEX_TYPENAME>(dst_size, left_size, src_dim_size, right_size, dst_dim_size, ids_dim_size, input, input_ids, output, tid); \
}


INDEX_OP(is_i64_f32, int64_t, float)
INDEX_OP(is_i64_f16, int64_t, half)
#if defined(__HAVE_BFLOAT__)
INDEX_OP(is_i64_bf16, int64_t, bfloat)
#endif

INDEX_OP(is_u32_u8, uint32_t, uint8_t)
INDEX_OP(is_u32_u32, uint32_t, uint32_t)
INDEX_OP(is_u32_f32, uint32_t, float)
INDEX_OP(is_u32_f16, uint32_t, half)
#if defined(__HAVE_BFLOAT__)
INDEX_OP(is_u32_bf16, uint32_t, bfloat)
#endif

INDEX_OP(is_u8_u8, uint8_t, uint8_t)
INDEX_OP(is_u8_u32, uint8_t, uint32_t)
INDEX_OP(is_u8_f32, uint8_t, float)
INDEX_OP(is_u8_f16, uint8_t, half)
#if defined(__HAVE_BFLOAT__)
INDEX_OP(is_u8_bf16, uint8_t, bfloat)
#endif

GATHER_OP(gather_i64_f32, int64_t, float)
GATHER_OP(gather_i64_f16, int64_t, half)
GATHER_OP(gather_u32_f32, uint, float)
GATHER_OP(gather_u32_f16, uint, half)
#if defined(__HAVE_BFLOAT__)
GATHER_OP(gather_i64_bf16, int64_t, bfloat)
GATHER_OP(gather_u32_bf16, uint, bfloat)
#endif
GATHER_OP(gather_i64_u32, int64_t, uint)
GATHER_OP(gather_u32_u32, uint, uint)
GATHER_OP(gather_i64_i64, int64_t, int64_t)
GATHER_OP(gather_u32_i64, uint, int64_t)

SCATTER_ADD_OP(sa_u32_f32, uint32_t, float)
SCATTER_ADD_OP(sa_u8_f32, uint8_t, float)
SCATTER_ADD_OP(sa_i64_f32, int64_t, float)
SCATTER_ADD_OP(sa_u32_u32, uint32_t, uint32_t)
SCATTER_ADD_OP(sa_u32_f16, uint32_t, half)
SCATTER_ADD_OP(sa_u8_f16, uint8_t, half)
SCATTER_ADD_OP(sa_i64_f16, int64_t, half)
#if defined(__HAVE_BFLOAT__)
SCATTER_ADD_OP(sa_u32_bf16, uint32_t, bfloat)
SCATTER_ADD_OP(sa_u8_bf16, uint8_t, bfloat)
SCATTER_ADD_OP(sa_i64_bf16, int64_t, bfloat)
#endif

// i64
INDEX_ADD_OP(ia_i64_f16, int64_t, half)
INDEX_ADD_OP(ia_i64_f32, int64_t, float)
INDEX_ADD_OP(ia_i64_i64, int64_t, int64_t)
INDEX_ADD_OP(ia_i64_u32, int64_t, uint32_t)
INDEX_ADD_OP(ia_i64_u8, int64_t, uint8_t)
#if defined(__HAVE_BFLOAT__)
INDEX_ADD_OP(ia_i64_bf16, int64_t, bfloat)
#endif

// u32
INDEX_ADD_OP(ia_u32_f16, uint32_t, half)
INDEX_ADD_OP(ia_u32_f32, uint32_t, float)
INDEX_ADD_OP(ia_u32_i64, uint32_t, int64_t)
INDEX_ADD_OP(ia_u32_u32, uint32_t, uint32_t)
INDEX_ADD_OP(ia_u32_u8, uint32_t, uint8_t)
#if defined(__HAVE_BFLOAT__)
INDEX_ADD_OP(ia_u32_bf16, uint32_t, bfloat)
#endif

// u8
INDEX_ADD_OP(ia_u8_f16, uint8_t, half)
INDEX_ADD_OP(ia_u8_f32, uint8_t, float)
INDEX_ADD_OP(ia_u8_i64, uint8_t, int64_t)
INDEX_ADD_OP(ia_u8_u32, uint8_t, uint32_t)
INDEX_ADD_OP(ia_u8_u8, uint8_t, uint8_t)
#if defined(__HAVE_BFLOAT__)
INDEX_ADD_OP(ia_u8_bf16, uint8_t, bfloat)
#endif
