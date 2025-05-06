#include <metal_stdlib>
#include <metal_limits>

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


using namespace metal;

typedef unsigned char fp8_storage_t;
typedef unsigned short int fp8x2_storage_t;
typedef unsigned int fp8x4_storage_t;

template <typename T>
struct _fp8_cast_traits;

template <>
struct _fp8_cast_traits<float> {
    typedef _fp_encoding_traits<float> traits;
    typedef typename traits::encoding_type encoding_type;
    constexpr static constant encoding_type head_mask = 0xFF800000;
    constexpr static constant encoding_type mantissa_mask = 0x7FFFFF;
    constexpr static constant encoding_type mask = 0x7FFFFFFF;
};

template <>
struct _fp8_cast_traits<half> {
    typedef _fp_encoding_traits<float> traits;
    typedef typename traits::encoding_type encoding_type;
    constexpr static constant encoding_type head_mask = 0xFC00;
    constexpr static constant encoding_type mantissa_mask = 0x3FF;
    constexpr static constant encoding_type mask = 0x7FFF;
};

enum _fp8_variant_t {
  E4M3 = 0,      // OCP E4M3
  E5M2 = 1,      // OCP E5M2
  E4M3_FNUZ = 2, // Standard FP8
  E5M2_FNUZ = 3, // BF8
};


typedef enum fp8_variant_t {
  _E4M3 = 0,      // OCP E4M3
  _E5M2 = 1,      // OCP E5M2
  _E4M3_FNUZ = 2, // Standard FP8
  _E5M2_FNUZ = 3, // BF8
} fp8_variant_t;


template<int variant>
struct _fp8_variant_traits;

template <>
struct _fp8_variant_traits<_fp8_variant_t::E4M3> {
    constexpr static constant bool is_fnuz = false;
    constexpr static constant int we = 4;
    constexpr static constant int wm = 3;
};
template <>
struct _fp8_variant_traits<_fp8_variant_t::E5M2> {
    constexpr static constant bool is_fnuz = false;
    constexpr static constant int we = 5;
    constexpr static constant int wm = 2;
};
template <>
struct _fp8_variant_traits<_fp8_variant_t::E4M3_FNUZ> {
    constexpr static constant bool is_fnuz = true;
    constexpr static constant int we = 4;
    constexpr static constant int wm = 3;
};
template <>
struct _fp8_variant_traits<_fp8_variant_t::E5M2_FNUZ> {
    constexpr static constant bool is_fnuz = true;
    constexpr static constant int we = 5;
    constexpr static constant int wm = 2;
};


template<typename T, int variant>
struct _fp8_variant_cast_traits;

template <>
struct _fp8_variant_cast_traits<float, _fp8_variant_t::E4M3> {
    typedef _fp_encoding_traits<float> traits;
    constexpr static constant traits::encoding_type ifmax = 0x43E00000;
    constexpr static constant traits::encoding_type ifmin = 0x0; // unused
};
template <>
struct _fp8_variant_cast_traits<float, _fp8_variant_t::E5M2> {
    typedef _fp_encoding_traits<float> traits;
    constexpr static constant traits::encoding_type ifmax = 0x47600000;
    constexpr static constant traits::encoding_type ifmin = 0xC7600000;
};
template <>
struct _fp8_variant_cast_traits<float, _fp8_variant_t::E4M3_FNUZ> {
    typedef _fp_encoding_traits<float> traits;
    constexpr static constant traits::encoding_type ifmax = 0x43700000;
    constexpr static constant traits::encoding_type ifmin = 0x0; // unused
};
template <>
struct _fp8_variant_cast_traits<float, _fp8_variant_t::E5M2_FNUZ> {
    typedef _fp_encoding_traits<float> traits;
    constexpr static constant traits::encoding_type ifmax = 0x47600000;
    constexpr static constant traits::encoding_type ifmin = 0xC7600000;
};

template <>
struct _fp8_variant_cast_traits<half, _fp8_variant_t::E4M3> {
    typedef _fp_encoding_traits<half> traits;
    constexpr static constant traits::encoding_type ifmax = 0x5F00;
    constexpr static constant traits::encoding_type ifmin = 0x0; // unused
};
template <>
struct _fp8_variant_cast_traits<half, _fp8_variant_t::E5M2> {
    typedef _fp_encoding_traits<half> traits;
    constexpr static constant traits::encoding_type ifmax = 0x7B00;
    constexpr static constant traits::encoding_type ifmin = 0xFB00;
};
template <>
struct _fp8_variant_cast_traits<half, _fp8_variant_t::E4M3_FNUZ> {
    typedef _fp_encoding_traits<half> traits;
    constexpr static constant traits::encoding_type ifmax = 0x5B80;
    constexpr static constant traits::encoding_type ifmin = 0x0; // unused
};
template <>
struct _fp8_variant_cast_traits<half, _fp8_variant_t::E5M2_FNUZ> {
    typedef _fp_encoding_traits<half> traits;
    constexpr static constant traits::encoding_type ifmax = 0x7B00;
    constexpr static constant traits::encoding_type ifmin = 0xFB00;
};

// TODO: Simplify. No need to support all fp8 variants immediately.
template <typename T, int variant = _fp8_variant_t::E4M3_FNUZ>
METAL_FUNC fp8_storage_t cast_to_fp8(T _x, bool clip = false, bool stoch = false, uint rng = 0) {
    typedef _fp_encoding_traits<T> traits;
    typedef typename traits::encoding_type bits;
    typedef numeric_limits<bits> limits;

    typedef _fp8_cast_traits<T> cast_traits;
    typedef _fp8_variant_traits<variant> variant_traits;
    typedef _fp8_variant_cast_traits<T, variant> variant_cast_traits;

    constexpr bool is_fnuz = variant_traits::is_fnuz;
    constexpr int we = variant_traits::we;
    constexpr int wm = variant_traits::wm;
    constexpr int mfmt = traits::exponent_shift;
    constexpr int bias = traits::exponent_bias;
    constexpr bits mask = cast_traits::mask;

    bits x = as_type<bits>(_x);

    bits head = x & cast_traits::head_mask;
    bits mantissa = x & cast_traits::mantissa_mask;
    int exponent = (head >> traits::exponent_shift) & traits::exponent_max;
    bits sign = head >> traits::sign_shift;

    bits signed_inf = 0;
    unsigned int nan = 0;
    if (is_fnuz) {
        signed_inf = clip ? ((sign << 7) + 0x7f) : 0x80;
        nan = 0x80;
    } else {
        if (we == 4) {
            signed_inf = (sign << 7) + (clip ? 0x7e : 0x7f);
        } else {
            signed_inf = (sign << 7) + (clip ? 0x7b : 0x7c);
        }
        nan = (sign << 7) + 0x7f;
    }
    constexpr bits ifmax = variant_cast_traits::ifmax;

    // Deal with inf and NaNs
    if ((x & traits::inf_mask) == traits::inf_mask) {
        if (is_fnuz || we == 4) return nan;  // fnuz and OCP E4M3 has no INF
        if (mantissa != 0) return nan;       // NaN
        return sign == 0 ? 0x7C : 0xFC;      // E5M2 Inf
    }

    if ((x & mask) > ifmax) {
        return signed_inf;
    }

    if (x == 0) {
        return 0;
    }

    // For IEEE bias mode, the bias is 2^(k-1) -1 where k is the width of exponent bits
    const int f8_bias = (1 << (we - 1)) - 1 + (is_fnuz ? 1 : 0);
    const int f8_denormal_act_exponent = 1 - f8_bias;  // actual exponent of f8 denormal

    int act_exponent, f8_exponent, exponent_diff;

    if (exponent == 0) {
        act_exponent = exponent - bias + 1;
        exponent_diff = f8_denormal_act_exponent - act_exponent;
    } else {
        act_exponent = exponent - bias;
        if (act_exponent <= f8_denormal_act_exponent) {
            exponent_diff = f8_denormal_act_exponent - act_exponent;
        } else {
            exponent_diff = 0;
        }
        mantissa += (1ull << mfmt);
    }

    bool midpoint = (mantissa & ((1ull << (mfmt - wm + exponent_diff)) - 1)) ==
        (1ull << (mfmt - wm + exponent_diff - 1));

    if (exponent_diff > 0)
        mantissa >>= exponent_diff;
    else if (exponent_diff == -1)
        mantissa <<= -exponent_diff;
    bool implicit_one = mantissa & (1ull << mfmt);
    f8_exponent =
        (act_exponent + exponent_diff) + f8_bias - (implicit_one ? 0 : 1);

    unsigned long drop_mask = (1ull << (mfmt - wm)) - 1;
    bool odd =
        mantissa & (1ull << (mfmt - wm));
    mantissa +=
        (stoch ? rng : (midpoint ? (odd ? mantissa : mantissa - 1ull) : mantissa)) & drop_mask;

    if (f8_exponent == 0) {
        if ((1ull << mfmt) & mantissa) {
            f8_exponent = 1;  // denormal overflow to become normal, promote exponent
        }
    } else {
        if ((1ull << (mfmt + 1)) & mantissa) {
            mantissa >>= 1;
            f8_exponent++;
        }
    }

    mantissa >>= (mfmt - wm);
    const int max_exp = (1 << we) - 1;
    if (f8_exponent > max_exp) {
        if (clip) {
            mantissa = (1 << wm) - 1;
            f8_exponent = max_exp;
        } else {
            return signed_inf;
        }
    }

    if (f8_exponent == 0 && mantissa == 0) return is_fnuz ? 0 : (sign << 7);
    mantissa &= (1 << wm) - 1;
    return (sign << 7) | (f8_exponent << wm) | mantissa;
}

template <int variant>
METAL_FUNC half fp8_to_half(fp8_storage_t x);

template <>
METAL_FUNC half fp8_to_half<_fp8_variant_t::E4M3>(fp8_storage_t x) {
    typedef _fp_encoding_traits<half> traits;
    typedef typename traits::encoding_type bits;

    bits ur = x << 8U;
    bits sign = ur & 0x8000U;
    bits exponent = (bits)(((ur & 0x7800U) >> 1U) + 0x2000U);
    bits mantissa = (ur & 0x0700U) >> 1U;
    unsigned char absx = 0x7FU & (unsigned char)x;

    if (absx == 0x7FU) {
        // return NaN
        ur = 0x7FFFU;
    } else if (exponent == 0x2000U) {
        if (mantissa != 0U) {
            // normalize
            mantissa = (bits)(mantissa << 1U);
            while ((mantissa & 0x0400U) == 0U) {
                mantissa = (bits)(mantissa << 1U);
                exponent = (bits)(exponent - 0x0400U);
            }
            // discard implicit leading bit
            mantissa &= 0x03FFU;
        } else {
            // zero
            exponent = 0U;
        }

        ur = (sign | exponent) | mantissa;
    } else {
        ur = (sign | exponent) | mantissa;
    }

    return as_type<half>(ur);
}

template <>
METAL_FUNC half fp8_to_half<_fp8_variant_t::E5M2>(fp8_storage_t x) {
    typedef _fp_encoding_traits<half> traits;
    typedef typename traits::encoding_type bits;

    bits ur = x << 8U;
    if ((x & 0x7FFFU) > 0x7C00U) {
        // return NaN
        ur = 0x7FFFU;
    }
    return as_type<half>(ur);
}

template <typename T, int variant>
METAL_FUNC T cast_fp8_to(fp8_storage_t x) {
    return static_cast<T>(fp8_to_half<variant>(x));
}

#define CAST_TO_FP8(name, T)                \
CAST_TO_FP8_VARIANT(name##_E4M3, T, E4M3)   \
CAST_TO_FP8_VARIANT(name##_E5M2, T, E5M2)

#define CAST_TO_FP8_VARIANT(name, T, FP8_VARIANT)               \
kernel void name(                                               \
    constant size_t &dim,                                       \
    device const T *input,                                      \
    device fp8_storage_t *output,                               \
    uint tid [[ thread_position_in_grid ]]                      \
) {                                                             \
    if (tid >= dim) {                                           \
        return;                                                 \
    }                                                           \
    output[tid] = cast_to_fp8<T, FP8_VARIANT>(input[tid]);       \
}                                                               \
kernel void name##_strided(                                     \
    constant size_t &dim,                                       \
    constant size_t &num_dims,                                  \
    constant size_t *dims,                                      \
    constant size_t *strides,                                   \
    device const T *input,                                      \
    device fp8_storage_t *output,                               \
    uint tid [[ thread_position_in_grid ]]                      \
) {                                                             \
    if (tid >= dim) {                                           \
        return;                                                 \
    }                                                           \
    output[tid] = cast_to_fp8<T, FP8_VARIANT>(                   \
        input[get_strided_index(tid, num_dims, dims, strides)]  \
    );                                                          \
}                                                               \

#define CAST_FROM_FP8(name, T)                \
CAST_FROM_FP8_VARIANT(name##_E4M3, T, E4M3)   \
CAST_FROM_FP8_VARIANT(name##_E5M2, T, E5M2)

#define CAST_FROM_FP8_VARIANT(name, T, FP8_VARIANT)             \
kernel void name(                                               \
    constant size_t &dim,                                       \
    device const fp8_storage_t *input,                          \
    device T *output,                                           \
    uint tid [[ thread_position_in_grid ]]                      \
) {                                                             \
    if (tid >= dim) {                                           \
        return;                                                 \
    }                                                           \
    output[tid] = cast_fp8_to<T, FP8_VARIANT>(input[tid]);      \
}                                                               \
kernel void name##_strided(                                     \
    constant size_t &dim,                                       \
    constant size_t &num_dims,                                  \
    constant size_t *dims,                                      \
    constant size_t *strides,                                   \
    device const fp8_storage_t *input,                          \
    device T *output,                                           \
    uint tid [[ thread_position_in_grid ]]                      \
) {                                                             \
    if (tid >= dim) {                                           \
        return;                                                 \
    }                                                           \
    output[tid] = cast_fp8_to<T, FP8_VARIANT>(                  \
        input[get_strided_index(tid, num_dims, dims, strides)]  \
    );                                                          \
}                                                               \

CAST_FROM_FP8(cast_fp8_f16, half)
CAST_FROM_FP8(cast_fp8_f32, float)
CAST_TO_FP8(cast_f32_fp8, float)
CAST_TO_FP8(cast_f16_fp8, half)


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

// u32
CAST(cast_u32_f32, cast_u32_f32_strided, uint32_t, float)
CAST(cast_u32_u8, cast_u32_u8_strided, uint32_t, uint8_t)
CAST(cast_u32_f16, cast_u32_f16_strided, uint32_t, half)
#if __METAL_VERSION__ >= 220
CAST(cast_u32_i64, cast_u32_i64_strided, uint32_t, int64_t)
#endif
#if defined(__HAVE_BFLOAT__)
CAST(cast_u32_bf16, cast_u32_bf16_strided, uint32_t, bfloat)
#endif

// u8
CAST(cast_u8_u32, cast_u8_u32_strided, uint8_t, uint32_t)
CAST(cast_u8_f32, cast_u8_f32_strided, uint8_t, float)
CAST(cast_u8_f16, cast_u8_f16_strided, uint8_t, half)
#if __METAL_VERSION__ >= 220
CAST(cast_u8_i64, cast_u8_i64_strided, uint8_t, int64_t)
#endif
#if defined(__HAVE_BFLOAT__)
CAST(cast_u8_bf16, cast_u8_bf16_strided, uint8_t, bfloat)
#endif

// f16
CAST(cast_f16_f32, cast_f16_f32_strided, half, float)
CAST(cast_f16_u8, cast_f16_u8_strided, half, uint8_t)
CAST(cast_f16_u32, cast_f16_u32_strided, half, uint32_t)
CAST(cast_f16_i64, cast_f16_i64_strided, half, int64_t)
#if defined(__HAVE_BFLOAT__)
CAST_THROUGH(cast_f16_bf16, cast_f16_bf16_strided, half, bfloat, float)
#endif

// i64
CAST(cast_i64_f32, cast_i64_f32_strided, int64_t, float)
CAST(cast_i64_u8, cast_i64_u8_strided, int64_t, uint8_t)
CAST(cast_i64_u32, cast_i64_u32_strided, int64_t, uint32_t)
CAST(cast_i64_f16, cast_i64_f16_strided, int64_t, half)
#if defined(__HAVE_BFLOAT__)
CAST_THROUGH(cast_i64_bf16, cast_i64_bf16_strided, int64_t, bfloat, float)
#endif

// f32
CAST(cast_f32_f16, cast_f32_f16_strided, float, half)
CAST(cast_f32_u32, cast_f32_u32_strided, float, uint32_t)
CAST(cast_f32_u8, cast_f32_u8_strided, float, uint8_t)
CAST(cast_f32_i64, cast_f32_i64_strided, float, int64_t)
#if defined(__HAVE_BFLOAT__)
CAST(cast_f32_bf16, cast_f32_bf16_strided, float, bfloat)
#endif

// bf16
#if defined(__HAVE_BFLOAT__)
CAST(cast_bf16_u32, cast_bf16_u32_strided, bfloat, uint32_t)
CAST(cast_bf16_i64, cast_bf16_i64_strided, bfloat, int64_t)
CAST(cast_bf16_f32, cast_bf16_f32_strided, bfloat, float)
CAST_THROUGH(cast_bf16_u8, cast_bf16_u8_strided, bfloat, uint8_t, float)
CAST_THROUGH(cast_bf16_f16, cast_bf16_f16_strided, bfloat, half, float)
#endif
