#include <metal_stdlib>
using namespace metal;

template<typename TYPENAME, typename INDEX_TYPENAME>
METAL_FUNC void index( 
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
    output[tid] = input[src_i]; 
}

# define INDEX_OP(NAME, INDEX_TYPENAME, TYPENAME) \
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
    index<TYPENAME, INDEX_TYPENAME>(dst_size, left_size, src_dim_size, right_size, ids_size, input, input_ids, output, tid); \
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


template <typename T, typename I>
void index_add(
    device I *ids [[buffer(0)]],
    device T *inp [[buffer(1)]],
    device T *out [[buffer(2)]],

    constant uint &ids_dim_size,
    constant uint &left_size,
    constant uint &dst_dim_size,
    constant uint &right_size,

    uint gid [[ thread_position_in_grid ]] \
) {

    if (gid >= left_size * right_size) {
        return;
    }

    const uint i = gid;
    const uint pre = i / right_size;
    const uint post = i % right_size;

    for (uint j = 0; j < ids_dim_size; j++) {
        const uint idx = ids[j];
        const uint src_i = (pre * ids_dim_size + j) * right_size + post;
        const uint dst_i = (pre * dst_dim_size + idx) * right_size + post;
        out[dst_i] += inp[src_i];
    }
}

#define IA_OP(TYPENAME, INDEX_TYPENAME, FN_NAME) \
kernel void FN_NAME( \
    device INDEX_TYPENAME *ids [[buffer(0)]], \
    device TYPENAME *inp [[buffer(1)]], \
    device TYPENAME *out [[buffer(2)]], \
    constant uint &ids_dim_size, \
    constant uint &left_size, \
    constant uint &dst_dim_size, \
    constant uint &right_size, \
    uint gid [[ thread_position_in_grid ]] \
) { index_add<TYPENAME, INDEX_TYPENAME>(ids, inp, out, ids_dim_size, left_size, dst_dim_size, right_size, gid); } \


INDEX_OP(is_u32_f32, uint, float)
INDEX_OP(is_u32_f16, uint, half)
GATHER_OP(gather_u32_f32, uint, float)
GATHER_OP(gather_u32_f16, uint, half)
SCATTER_ADD_OP(sa_u32_f32, uint, float)
SCATTER_ADD_OP(sa_u32_f16, uint, half)


#if __METAL_VERSION__ >= 310
IA_OP(bfloat, int64_t, ia_i64_bf16)
IA_OP(bfloat, uint32_t, ia_u32_bf16)
IA_OP(bfloat, uint8_t, ia_u8_bf16)
#endif

IA_OP(half, uint32_t, ia_u32_f16)
IA_OP(half, uint8_t, ia_u8_f16)

IA_OP(float, int64_t, ia_i64_f32)
IA_OP(uint8_t, int64_t, ia_i64_u8)
IA_OP(int64_t, int64_t, ia_i64_i64)
IA_OP(uint32_t, int64_t, ia_i64_u32)

IA_OP(float, uint32_t, ia_u32_f32)
IA_OP(uint8_t, uint32_t, ia_u32_u8)
IA_OP(int64_t, uint32_t, ia_u32_i64)
IA_OP(uint32_t, uint32_t, ia_u32_u32)

IA_OP(float, uint8_t, ia_u8_f32)
IA_OP(uint8_t, uint8_t, ia_u8_u8)
IA_OP(uint32_t, uint8_t, ia_u8_u32)
IA_OP(int64_t, uint8_t, ia_u8_i64)
