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

#define CAST(FN_NAME, FN_NAME_STRIDED, LEFT_TYPENAME, RIGHT_TYPENAME) \
kernel void FN_NAME( \
    constant size_t &dim, \
    device const LEFT_TYPENAME *input,  \
    device RIGHT_TYPENAME *output, \
    uint tid [[ thread_position_in_grid ]] \
) { \
    if (tid >= dim) { \
        return; \
    } \
    output[tid] = static_cast<RIGHT_TYPENAME>(input[tid]); \
} \
kernel void FN_NAME_STRIDED( \
    constant size_t &dim, \
    constant size_t &num_dims, \
    constant size_t *dims, \
    constant size_t *strides, \
    device const LEFT_TYPENAME *input,  \
    device RIGHT_TYPENAME *output, \
    uint tid [[ thread_position_in_grid ]] \
) { \
    if (tid >= dim) { \
        return; \
    } \
    output[tid] = static_cast<RIGHT_TYPENAME>(input[get_strided_index(tid, num_dims, dims, strides)]); \
} \

#define CAST_THROUGH(FN_NAME, FN_NAME_STRIDED, LEFT_TYPENAME, RIGHT_TYPENAME, IR_TYPENAME) \
kernel void FN_NAME( \
    constant size_t &dim, \
    device const LEFT_TYPENAME *input,  \
    device RIGHT_TYPENAME *output, \
    uint tid [[ thread_position_in_grid ]] \
) { \
    if (tid >= dim) { \
        return; \
    } \
    output[tid] = static_cast<RIGHT_TYPENAME>(static_cast<IR_TYPENAME>(input[tid])); \
} \
kernel void FN_NAME_STRIDED( \
    constant size_t &dim, \
    constant size_t &num_dims, \
    constant size_t *dims, \
    constant size_t *strides, \
    device const LEFT_TYPENAME *input,  \
    device RIGHT_TYPENAME *output, \
    uint tid [[ thread_position_in_grid ]] \
) { \
    if (tid >= dim) { \
        return; \
    } \
    output[tid] = static_cast<RIGHT_TYPENAME>(static_cast<IR_TYPENAME>(input[get_strided_index(tid, num_dims, dims, strides)])); \
} \

uint8_t i64_to_fp8e4m3(int64_t x) {
    uint64_t xbits = as_type<uint64_t>(x);
    
    // Constants for E4M3 interpretation
    uint8_t FP8_MANTISSA_MASK = 0x7;
    uint16_t FP8_EXP_BIAS = 7;
    uint64_t FP8_SIGNIFICAND_BITS = 4;
    uint64_t FP8_MINDENORM_O2 = 0x3F50000000000000;
    uint64_t FP8_OVERFLOW_THRESHOLD = 0x407D000000000000;
    uint64_t FP8_MINNORM = 0x3F90000000000000;
    uint64_t DP_INF_BITS = 0x7FF0000000000000;
    
    uint64_t fp8_dp_half_ulp = 1 << (53 - FP8_SIGNIFICAND_BITS - 1);
    
    uint8_t sign = ((xbits >> 63) << 7);
    uint16_t exp = ((((xbits >> 52) & 0x7FF) - 1023 + FP8_EXP_BIAS) & 0xFF);
    uint8_t mantissa = ((xbits >> (53 - FP8_SIGNIFICAND_BITS)) & FP8_MANTISSA_MASK);
    uint64_t absx = xbits & 0x7FFFFFFFFFFFFFFF;
    
    uint8_t res;
    
    if (absx <= FP8_MINDENORM_O2) {
        // Zero or underflow
        res = 0;
    }
    else if (absx > DP_INF_BITS) {
        // Preserve NaNs
        res = 0x7F;
    }
    else if (absx > FP8_OVERFLOW_THRESHOLD) {
        // No saturation, so use NaN representation
        res = 0x7F;
    }
    else if (absx >= FP8_MINNORM) {
        // Round, normal range
        uint8_t intermediate = (exp << (FP8_SIGNIFICAND_BITS - 1)) | mantissa;
        
        // Round off bits and round-to-nearest-even adjustment
        uint64_t round = xbits & ((fp8_dp_half_ulp << 1) - 1);
        if ((round > fp8_dp_half_ulp) || ((round == fp8_dp_half_ulp) && (mantissa & 1 != 0))) {
            intermediate = intermediate + 1;
        }
        
        res = intermediate;
    }
    else {
        // Denormal numbers
        uint8_t shift = 1 - exp;
        uint8_t denormal_mantissa = mantissa | (1 << (FP8_SIGNIFICAND_BITS - 1));
        uint8_t intermediate = denormal_mantissa >> shift;
        
        // Round off bits and round-to-nearest-even adjustment
        uint64_t round = (xbits | (1 << (53 - 1))) & ((fp8_dp_half_ulp << (shift + 1)) - 1);
        if ((round > (fp8_dp_half_ulp << shift)) || 
            ((round == (fp8_dp_half_ulp << shift)) && (intermediate & 1 != 0))) {
            intermediate = intermediate + 1;
        }
        
        res = intermediate;
    }

    return res | sign;
}

ushort fp8e4m3_to_fp16(uchar x) {
    ushort ur = (ushort(x) << 8);

    ushort sign = ur & 0x8000U;
    ushort exponent = static_cast<ushort>(((ur & 0x7800U) >> 1) + 0x2000U);
    ushort mantissa = static_cast<ushort>((ur & 0x0700U) >> 1);
    uchar absx = 0x7FU & static_cast<uchar>(x);

    if (absx == 0x7FU) {
        // NaN
        ur = 0x7FFFU; // fp16 canonical NaN, discard sign
    } else if (exponent == 0x2000U) {
        // zero or denormal
        if (mantissa != 0U) {
            // normalize
            mantissa = static_cast<ushort>(mantissa << 1);
            while ((mantissa & 0x0400U) == 0U) {
                mantissa = static_cast<ushort>(mantissa << 1);
                exponent = static_cast<ushort>(exponent - 0x0400U);
            }
            // discard implicit leading bit
            mantissa &= 0x03FFU;
        } else {
            // Zero
            exponent = 0U;
        }

        ur = (sign | exponent) | mantissa;
    } else {
        ur = (sign | exponent) | mantissa;
    }

    return ur;
}

kernel void cast_f8e4m3_bf16(
    constant size_t &dim,
    device const uint8_t *input,
    device bfloat16_t *output,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= dim) {
        return;
    }
    half x = as_type<half>(fp8e4m3_to_fp16(input[tid]));
    output[tid] = static_cast<bfloat16_t>(x);
}

kernel void cast_i64_f8e4m3(
    constant size_t &dim,
    device const int64_t *input,
    device uint8_t *output,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= dim) {
        return;
    }
    int64_t x = input[tid];
    output[tid] = i64_to_fp8e4m3(x);
}

kernel void cast_f8e4m3_f32(
    constant size_t &dim,
    device const uint8_t *input,
    device float *output,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= dim) {
        return;
    }
    half x = as_type<half>(fp8e4m3_to_fp16(input[tid]));
    // output[tid] = static_cast<float>(x);
    // half x = as_type<half>(fp8e4m3_to_fp16(56));
    output[tid] = static_cast<float>(x);
}

// u32
CAST(cast_u32_f32, cast_u32_f32_strided, uint32_t, float)
CAST(cast_u32_u8, cast_u32_u8_strided, uint32_t, uint8_t)
CAST(cast_u32_f16, cast_u32_f16_strided, uint32_t, half)
CAST(cast_u32_i32, cast_u32_i32_strided, uint32_t, int32_t)
CAST(cast_u32_i16, cast_u32_i16_strided, uint32_t, int16_t)
#if __METAL_VERSION__ >= 220
CAST(cast_u32_i64, cast_u32_i64_strided, uint32_t, int64_t)
#endif
CAST(cast_u32_bf16, cast_u32_bf16_strided, uint32_t, bfloat16_t)

// u8
CAST(cast_u8_u32, cast_u8_u32_strided, uint8_t, uint32_t)
CAST(cast_u8_f32, cast_u8_f32_strided, uint8_t, float)
CAST(cast_u8_f16, cast_u8_f16_strided, uint8_t, half)
CAST(cast_u8_i32, cast_u8_i32_strided, uint8_t, int64_t)
CAST(cast_u8_i16, cast_u8_i16_strided, uint8_t, int16_t)
#if __METAL_VERSION__ >= 220
CAST(cast_u8_i64, cast_u8_i64_strided, uint8_t, int64_t)
#endif
CAST(cast_u8_bf16, cast_u8_bf16_strided, uint8_t, bfloat16_t)

// f16
CAST(cast_f16_f32, cast_f16_f32_strided, half, float)
CAST(cast_f16_u8, cast_f16_u8_strided, half, uint8_t)
CAST(cast_f16_u32, cast_f16_u32_strided, half, uint32_t)
CAST(cast_f16_i16, cast_f16_i16_strided, half, int16_t)
CAST(cast_f16_i32, cast_f16_i32_strided, half, int64_t)
CAST(cast_f16_i64, cast_f16_i64_strided, half, int64_t)
CAST_THROUGH(cast_f16_bf16, cast_f16_bf16_strided, half, bfloat16_t, float)

// i64
CAST(cast_i64_f32, cast_i64_f32_strided, int64_t, float)
CAST(cast_i64_u8, cast_i64_u8_strided, int64_t, uint8_t)
CAST(cast_i64_u32, cast_i64_u32_strided, int64_t, uint32_t)
CAST(cast_i64_i32, cast_i64_i32_strided, int64_t, int32_t)
CAST(cast_i64_i16, cast_i64_i16_strided, int64_t, int16_t)
CAST(cast_i64_f16, cast_i64_f16_strided, int64_t, half)
CAST_THROUGH(cast_i64_bf16, cast_i64_bf16_strided, int64_t, bfloat16_t, float)

// i32
CAST(cast_i32_f32, cast_i32_f32_strided, int32_t, float)
CAST(cast_i32_u8, cast_i32_u8_strided, int32_t, uint8_t)
CAST(cast_i32_u32, cast_i32_u32_strided, int32_t, uint32_t)
CAST(cast_i32_i64, cast_i32_i64_strided, int32_t, int64_t)
CAST(cast_i32_i16, cast_i32_i16_strided, int32_t, int16_t)
CAST(cast_i32_f16, cast_i32_f16_strided, int32_t, half)
CAST_THROUGH(cast_i32_bf16, cast_i32_bf16_strided, int64_t, bfloat16_t, float)

// i16
CAST(cast_i16_f32, cast_i16_f32_strided, int16_t, float)
CAST(cast_i16_u8, cast_i16_u8_strided, int16_t, uint8_t)
CAST(cast_i16_u32, cast_i16_u32_strided, int16_t, uint32_t)
CAST(cast_i16_i32, cast_i16_i32_strided, int16_t, int32_t)
CAST(cast_i16_i64, cast_i16_i64_strided, int16_t, int64_t)
CAST(cast_i16_f16, cast_i16_f16_strided, int16_t, half)
CAST_THROUGH(cast_i16_bf16, cast_i16_bf16_strided, int16_t, bfloat16_t, float)

// f32
CAST(cast_f32_f16, cast_f32_f16_strided, float, half)
CAST(cast_f32_u32, cast_f32_u32_strided, float, uint32_t)
CAST(cast_f32_u8, cast_f32_u8_strided, float, uint8_t)
CAST(cast_f32_i16, cast_f32_i16_strided, float, int16_t)
CAST(cast_f32_i32, cast_f32_i32_strided, float, int32_t)
CAST(cast_f32_i64, cast_f32_i64_strided, float, int64_t)
CAST(cast_f32_bf16, cast_f32_bf16_strided, float, bfloat16_t)

// bf16
CAST(cast_bf16_u32, cast_bf16_u32_strided, bfloat16_t, uint32_t)
CAST(cast_bf16_i16, cast_bf16_i16_strided, bfloat16_t, int16_t)
CAST(cast_bf16_i32, cast_bf16_i32_strided, bfloat16_t, int32_t)
CAST(cast_bf16_i64, cast_bf16_i64_strided, bfloat16_t, int64_t)
CAST(cast_bf16_f32, cast_bf16_f32_strided, bfloat16_t, float)
CAST_THROUGH(cast_bf16_u8, cast_bf16_u8_strided, bfloat16_t, uint8_t, float)
CAST_THROUGH(cast_bf16_f16, cast_bf16_f16_strided, bfloat16_t, half, float)