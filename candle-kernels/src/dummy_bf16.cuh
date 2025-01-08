#include<stdint.h>

__device__ __forceinline__ float bf16_to_f32(const uint16_t i)
{
    // If NaN, keep current mantissa but also set most significant mantissa bit
    if ((i & 0x7FFFu) > 0x7F80u) {
        // NaN path
        uint32_t tmp = ((static_cast<uint32_t>(i) | 0x0040u) << 16);
        union {
            uint32_t as_int;
            float as_float;
        } u;
        u.as_int = tmp;
        return u.as_float;
        // Alternatively:
        // return __int_as_float(((static_cast<uint32_t>(i) | 0x0040u) << 16));
    } else {
        // Normal path
        uint32_t tmp = (static_cast<uint32_t>(i) << 16);
        union {
            uint32_t as_int;
            float as_float;
        } u;
        u.as_int = tmp;
        return u.as_float;
        // Alternatively:
        // return __int_as_float(static_cast<uint32_t>(i) << 16);
    }
}

// Convert FP32 (float) to BF16 (unsigned short)
__device__ __forceinline__ uint16_t f32_to_bf16(const float value)
{
    // Reinterpret float bits as uint32_t
    union {
        float as_float;
        uint32_t as_int;
    } u;
    u.as_float = value;
    uint32_t x = u.as_int;

    // Check for NaN
    if ((x & 0x7FFF'FFFFu) > 0x7F80'0000u) {
        // Keep high part of current mantissa but also set most significant mantissa bit
        return static_cast<uint16_t>((x >> 16) | 0x0040u);
    }

    // Round and shift
    constexpr uint32_t round_bit = 0x0000'8000u;  // bit 15
    if (((x & round_bit) != 0) && ((x & (3 * round_bit - 1)) != 0)) {
        // Round half to even (or to odd) depends on your preference
        return static_cast<uint16_t>((x >> 16) + 1);
    } else {
        return static_cast<uint16_t>(x >> 16);
    }
}
