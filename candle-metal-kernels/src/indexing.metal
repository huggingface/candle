#include <metal_stdlib>

using namespace metal;

#if defined(__HAVE_BFLOAT__)

typedef bfloat bfloat16_t;

#else

/////////////////////////////////////////////////////////////////////////////
// Helpers
/////////////////////////////////////////////////////////////////////////////

constexpr METAL_FUNC uint16_t float_to_bfloat_bits(float x) {
  // Check for nan
  if ((as_type<uint32_t>(x) & ~_fp_encoding_traits<float>::sign_mask) >
      _fp_encoding_traits<float>::inf_mask) {
    return uint16_t(as_type<uint32_t>(0x7FC0));
  }
  // Take bits
  uint32_t float_bits = as_type<uint32_t>(x);

  // Round to nearest even
  float_bits += ((float_bits >> 16) & 1) + as_type<uint32_t>(0x7FFF);

  // Take upper 16 bits
  return float_bits >> 16;
}

constexpr METAL_FUNC float bfloat_bits_to_float(uint16_t x) {
  // Upper 16 bits are the data and lower 16 bits are 0s
  return as_type<float>((uint32_t)x << 16);
}

struct _MLX_BFloat16;

template <typename T>
static constexpr constant bool can_convert_to_bfloat =
    !is_same_v<T, _MLX_BFloat16> && is_convertible_v<T, float>;

template <typename T>
static constexpr constant bool can_convert_from_bfloat =
    !is_same_v<T, _MLX_BFloat16> && is_convertible_v<float, T>;

/////////////////////////////////////////////////////////////////////////////
// Bfloat struct
/////////////////////////////////////////////////////////////////////////////

struct _MLX_BFloat16 {
  /////////////////////////////////////////////////////////////////////////////
  // Constructors
  uint16_t bits_;
  _MLX_BFloat16() thread = default;
  _MLX_BFloat16() threadgroup = default;
  _MLX_BFloat16() device = default;
  _MLX_BFloat16() constant = default;

  struct bits_to_bfloat_struct {};
  static constexpr METAL_FUNC bits_to_bfloat_struct bits_to_bfloat() {
    return bits_to_bfloat_struct();
  }
  constexpr METAL_FUNC _MLX_BFloat16(uint16_t bits, bits_to_bfloat_struct)
      : bits_(bits) {}

  /////////////////////////////////////////////////////////////////////////////
  // Conversions to bfloat

  template <
      typename T,
      typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) thread
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) threadgroup
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) device
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) constant
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  /////////////////////////////////////////////////////////////////////////////
  // Conversions from bfloat

  template <
      typename T,
      typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const thread {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const threadgroup {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const device {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const constant {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }
};

/////////////////////////////////////////////////////////////////////////////
// Bfloat operators
/////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////
// Unary ops
constexpr METAL_FUNC _MLX_BFloat16 operator-(_MLX_BFloat16 x) {
  return -static_cast<float>(x);
}

/////////////////////////////////////////////////////////////////////////////
// Binary operators
#define bfloat_binop_base(__op__, __operator__, otype, atype, btype, ctype) \
  constexpr METAL_FUNC otype __operator__(atype lhs, btype rhs) {           \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);          \
  }

#define bfloat_binop_helper(__op__, __operator__, otype, itype, ctype)    \
  constexpr METAL_FUNC otype __operator__(_MLX_BFloat16 lhs, itype rhs) { \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);        \
  }                                                                       \
  constexpr METAL_FUNC otype __operator__(itype lhs, _MLX_BFloat16 rhs) { \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);        \
  }

/////////////////////////////////////////////////////////////////////////////
// Arithmetic Operators
#define bfloat_binop(_op_, _operator_)                                       \
  bfloat_binop_base(                                                         \
      _op_, _operator_, _MLX_BFloat16, _MLX_BFloat16, _MLX_BFloat16, float); \
  bfloat_binop_helper(_op_, _operator_, float, float, float);                \
  bfloat_binop_helper(_op_, _operator_, float, half, float);                 \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, int32_t, float);      \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, uint32_t, float);     \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, int64_t, float);      \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, uint64_t, float);

bfloat_binop(+, operator+);
bfloat_binop(-, operator-);
bfloat_binop(*, operator*);
bfloat_binop(/, operator/);

/////////////////////////////////////////////////////////////////////////////
// Comparison ops
#define bfloat_compop(__op__, __operator__)                             \
  bfloat_binop_base(                                                    \
      __op__, __operator__, bool, _MLX_BFloat16, _MLX_BFloat16, float); \
  bfloat_binop_helper(__op__, __operator__, bool, float, float);        \
  bfloat_binop_helper(__op__, __operator__, bool, half, float);         \
  bfloat_binop_helper(__op__, __operator__, bool, int32_t, float);      \
  bfloat_binop_helper(__op__, __operator__, bool, uint32_t, float);     \
  bfloat_binop_helper(__op__, __operator__, bool, int64_t, float);      \
  bfloat_binop_helper(__op__, __operator__, bool, uint64_t, float);

bfloat_compop(>, operator>);
bfloat_compop(<, operator<);
bfloat_compop(>=, operator>=);
bfloat_compop(<=, operator<=);
bfloat_compop(==, operator==);
bfloat_compop(!=, operator!=);

#undef bfloat_compop
#undef bfloat_binop_base
#undef bfloat_binop_helper
#undef bfloat_binop

/////////////////////////////////////////////////////////////////////////////
// Inplace Operators
#define bfloat_inplace_op_helper(__op__, __operator__, itype, addr_space) \
  constexpr METAL_FUNC addr_space _MLX_BFloat16& __operator__(            \
      addr_space _MLX_BFloat16& lhs, itype rhs) {                         \
    lhs = static_cast<float>(lhs) __op__ static_cast<float>(rhs);         \
    return lhs;                                                           \
  }                                                                       \
  constexpr METAL_FUNC addr_space itype& __operator__(                    \
      addr_space itype& lhs, _MLX_BFloat16 rhs) {                         \
    lhs = static_cast<float>(lhs) __op__ static_cast<float>(rhs);         \
    return lhs;                                                           \
  }

#define bfloat_inplace_op_addr_space_helper(__op__, __operator__, itype) \
  bfloat_inplace_op_helper(__op__, __operator__, itype, device);         \
  bfloat_inplace_op_helper(__op__, __operator__, itype, thread);         \
  bfloat_inplace_op_helper(__op__, __operator__, itype, threadgroup);

#define bfloat_inplace_op(itype)                             \
  bfloat_inplace_op_addr_space_helper(+, operator+=, itype); \
  bfloat_inplace_op_addr_space_helper(-, operator-=, itype); \
  bfloat_inplace_op_addr_space_helper(*, operator*=, itype); \
  bfloat_inplace_op_addr_space_helper(/, operator/=, itype);

bfloat_inplace_op(float);
bfloat_inplace_op(half);
bfloat_inplace_op(int16_t);
bfloat_inplace_op(int32_t);
bfloat_inplace_op(int64_t);
bfloat_inplace_op(uint16_t);
bfloat_inplace_op(uint32_t);
bfloat_inplace_op(uint64_t);

#undef bfloat_inplace_op_helper
#undef bfloat_inplace_op_addr_space_helper
#undef bfloat_inplace_op

#define bfloat_inplace_op_helper(__op__, __operator__, addr_space) \
  constexpr METAL_FUNC addr_space _MLX_BFloat16& __operator__(     \
      addr_space _MLX_BFloat16& lhs, _MLX_BFloat16 rhs) {          \
    lhs = static_cast<float>(lhs) __op__ static_cast<float>(rhs);  \
    return lhs;                                                    \
  }

#define bfloat_inplace_op_addr_space_helper(__op__, __operator__) \
  bfloat_inplace_op_helper(__op__, __operator__, device);         \
  bfloat_inplace_op_helper(__op__, __operator__, thread);         \
  bfloat_inplace_op_helper(__op__, __operator__, threadgroup);

bfloat_inplace_op_addr_space_helper(+, operator+=);
bfloat_inplace_op_addr_space_helper(-, operator-=);
bfloat_inplace_op_addr_space_helper(*, operator*=);
bfloat_inplace_op_addr_space_helper(/, operator/=);

#undef bfloat_inplace_op_helper
#undef bfloat_inplace_op_addr_space_helper

/////////////////////////////////////////////////////////////////////////////
// Bfloat typedef
/////////////////////////////////////////////////////////////////////////////

typedef struct _MLX_BFloat16 bfloat16_t;

#endif

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
INDEX_OP(is_i64_bf16, int64_t, bfloat16_t)

INDEX_OP(is_i32_f32, int32_t, float)
INDEX_OP(is_i32_f16, int32_t, half)
INDEX_OP(is_i32_bf16, int32_t, bfloat16_t)

INDEX_OP(is_i16_f32, int16_t, float)
INDEX_OP(is_i16_f16, int16_t, half)
INDEX_OP(is_i16_bf16, int16_t, bfloat16_t)

INDEX_OP(is_u32_u8, uint32_t, uint8_t)
INDEX_OP(is_u32_u32, uint32_t, uint32_t)
INDEX_OP(is_u32_f32, uint32_t, float)
INDEX_OP(is_u32_f16, uint32_t, half)
INDEX_OP(is_u32_bf16, uint32_t, bfloat16_t)

INDEX_OP(is_u8_u8, uint8_t, uint8_t)
INDEX_OP(is_u8_u32, uint8_t, uint32_t)
INDEX_OP(is_u8_f32, uint8_t, float)
INDEX_OP(is_u8_f16, uint8_t, half)
INDEX_OP(is_u8_bf16, uint8_t, bfloat16_t)

GATHER_OP(gather_u32_f32, uint, float)
GATHER_OP(gather_u32_f16, uint, half)
GATHER_OP(gather_u32_bf16, uint, bfloat16_t)

SCATTER_ADD_OP(sa_u32_f32, uint32_t, float)
SCATTER_ADD_OP(sa_u8_f32, uint8_t, float)
SCATTER_ADD_OP(sa_i16_f32, int16_t, float)
SCATTER_ADD_OP(sa_i32_f32, int32_t, float)
SCATTER_ADD_OP(sa_i64_f32, int64_t, float)
SCATTER_ADD_OP(sa_u32_f16, uint32_t, half)
SCATTER_ADD_OP(sa_u8_f16, uint8_t, half)
SCATTER_ADD_OP(sa_i16_f16, int16_t, half)
SCATTER_ADD_OP(sa_i32_f16, int32_t, half)
SCATTER_ADD_OP(sa_i64_f16, int64_t, half)
SCATTER_ADD_OP(sa_u32_bf16, uint32_t, bfloat16_t)
SCATTER_ADD_OP(sa_u8_bf16, uint8_t, bfloat16_t)
SCATTER_ADD_OP(sa_i64_bf16, int64_t, bfloat16_t)

// i64
INDEX_ADD_OP(ia_i64_f16, int64_t, half)
INDEX_ADD_OP(ia_i64_f32, int64_t, float)
INDEX_ADD_OP(ia_i64_i16, int64_t, int16_t)
INDEX_ADD_OP(ia_i64_i32, int64_t, int32_t)
INDEX_ADD_OP(ia_i64_i64, int64_t, int64_t)
INDEX_ADD_OP(ia_i64_u32, int64_t, uint32_t)
INDEX_ADD_OP(ia_i64_u8, int64_t, uint8_t)
INDEX_ADD_OP(ia_i64_bf16, int64_t, bfloat16_t)

// i32
INDEX_ADD_OP(ia_i32_f16, int32_t, half)
INDEX_ADD_OP(ia_i32_f32, int32_t, float)
INDEX_ADD_OP(ia_i32_i64, int32_t, int64_t)
INDEX_ADD_OP(ia_i32_i32, int32_t, int32_t)
INDEX_ADD_OP(ia_i32_u32, int32_t, uint32_t)
INDEX_ADD_OP(ia_i32_u8, int32_t, uint8_t)
INDEX_ADD_OP(ia_i32_bf16, int32_t, bfloat16_t)

// i16
INDEX_ADD_OP(ia_i16_f16, int16_t, half)
INDEX_ADD_OP(ia_i16_f32, int16_t, float)
INDEX_ADD_OP(ia_i16_i16, int16_t, int16_t)
INDEX_ADD_OP(ia_i16_i32, int16_t, int32_t)
INDEX_ADD_OP(ia_i16_i64, int16_t, int64_t)
INDEX_ADD_OP(ia_i16_u32, int16_t, uint32_t)
INDEX_ADD_OP(ia_i16_u8, int16_t, uint8_t)
INDEX_ADD_OP(ia_i16_bf16, int16_t, bfloat16_t)


// u32
INDEX_ADD_OP(ia_u32_f16, uint32_t, half)
INDEX_ADD_OP(ia_u32_f32, uint32_t, float)
INDEX_ADD_OP(ia_u32_i16, uint32_t, int16_t)
INDEX_ADD_OP(ia_u32_i32, uint32_t, int32_t)
INDEX_ADD_OP(ia_u32_i64, uint32_t, int64_t)
INDEX_ADD_OP(ia_u32_u32, uint32_t, uint32_t)
INDEX_ADD_OP(ia_u32_u8, uint32_t, uint8_t)
INDEX_ADD_OP(ia_u32_bf16, uint32_t, bfloat16_t)

// u8
INDEX_ADD_OP(ia_u8_f16, uint8_t, half)
INDEX_ADD_OP(ia_u8_f32, uint8_t, float)
INDEX_ADD_OP(ia_u8_i16, uint8_t, int16_t)
INDEX_ADD_OP(ia_u8_i32, uint8_t, int32_t)
INDEX_ADD_OP(ia_u8_i64, uint8_t, int64_t)
INDEX_ADD_OP(ia_u8_u32, uint8_t, uint32_t)
INDEX_ADD_OP(ia_u8_u8, uint8_t, uint8_t)
INDEX_ADD_OP(ia_u8_bf16, uint8_t, bfloat16_t)
