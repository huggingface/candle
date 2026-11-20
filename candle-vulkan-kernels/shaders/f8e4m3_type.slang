// FloatE4M3 scalar type and encode/decode, native or emulated.
//
// Compiled with -DFP8_NATIVE=1 on devices that enable the
// VK_EXT_shader_float8 extension and the shaderFloat8 device feature:
// `FP8` is the native Slang `FloatE4M3` type, stored directly in
// `StructuredBuffer<FloatE4M3>` / `RWStructuredBuffer<FloatE4M3>`. The
// 8-bit in-memory layout of `FloatE4M3` is exactly the
// `microfloat::f8e4m3` bit pattern, so the encode/decode below convert
// through the raw 8-bit pattern with `bit_cast` — the same microfloat
// crate algorithms the CPU backend uses, bit for bit.
//
// Otherwise (the default) `FP8` is the 8-bit bit pattern (`uint8_t`)
// and the same encode/decode functions apply to it directly.
//
// Format: 1 sign, 4 exponent (bias 7), 3 mantissa bits, IEEE-style NaN
// and infinity, max finite 240, overflow -> infinity.
module f8e4m3_type;

#if defined(FP8_NATIVE)
    public typealias FP8 = FloatE4M3;
#else
    public typealias FP8 = uint8_t;
#endif

// 2^e as float (microfloat exp2i), e in the f32-representable range.
float f8e4m3_f2i(int e)
{
    if (e < -149) { return 0.0f; }
    if (e > 127) { return bit_cast<float>(0x7f800000u); }
    if (e >= -126) { return bit_cast<float>(uint(e + 127) << 23); }
    return bit_cast<float>(1u << uint(e + 149));
}

// Round-to-nearest-even right shift (microfloat round_right_shift).
uint f8e4m3_rrs(uint64_t value, int shift)
{
    if (shift >= 64) { return 0u; }
    uint s = uint(shift);
    uint q = (uint)(value >> s);
    uint rem = (uint)(value & (((uint64_t)1u << s) - (uint64_t)1u));
    uint half = 1u << (s - 1u);
    if (rem > half || (rem == half && (q & 1u) == 1u)) { return q + 1u; }
    return q;
}

// Round value * 2^scale_exp to the nearest integer (microfloat round_to_integer).
uint64_t f8e4m3_rti(float value, int scale_exp)
{
    uint bits = bit_cast<uint>(value);
    int exp = int((bits >> 23) & 0xffu);
    uint fraction = bits & 0x7f_ffffu;
    uint64_t sig;
    int power;
    if (exp == 0) { sig = (uint64_t)fraction; power = -149; }
    else { sig = (uint64_t)((1u << 23) | fraction); power = exp - 150; }
    int shift = power - scale_exp;
    if (shift >= 0) { return sig << uint(shift); }
    return (uint64_t)f8e4m3_rrs(sig, -shift);
}

// floor(log2(v)) for positive finite v (microfloat floor_log2_f32).
int f8e4m3_fl2(float v)
{
    uint bits = bit_cast<uint>(v) & 0x7fff_ffffu;
    int exp = int((bits >> 23) & 0xffu);
    if (exp == 0) { return -200; }
    return exp - 127;
}

// f8e4m3 bit pattern -> float (microfloat decode_f32, bias 7, e4m3).
public float f8e4m3_decode_bits(uint8_t bits)
{
    uint b = (uint)bits;
    bool neg = (b & 0x80u) != 0u;
    if ((b & 0x78u) == 0x78u)
    {
        if ((b & 0x07u) != 0u)
        {
            return bit_cast<float>(neg ? 0xffc00000u : 0x7fc00000u);
        }
        return bit_cast<float>(neg ? 0xff800000u : 0x7f800000u);
    }
    uint mag = b & 0x7fu;
    float sign = neg ? -1.0f : 1.0f;
    if (mag == 0u) { return (sign < 0.0f) ? -0.0f : 0.0f; }
    uint exp = (b & 0x78u) >> 3;
    float mant = (float)(b & 0x07u);
    float scale = (exp == 0u) ? f8e4m3_f2i(1 - 7) : f8e4m3_f2i(int(exp) - 7);
    float significand = (exp == 0u) ? (mant / f8e4m3_f2i(3)) : (1.0f + mant / f8e4m3_f2i(3));
    return sign * significand * scale;
}

// float -> f8e4m3 bit pattern (microfloat encode_f32, bias 7, e4m3).
public uint8_t f8e4m3_encode_bits(float v)
{
    uint x = bit_cast<uint>(v);
    uint mag = x & 0x7fff_ffffu;
    bool neg = (x & 0x80000000u) != 0u;
    if (mag > 0x7f800000u)
    {
        uint base = 0x78u | (1u << 2);
        return (uint8_t)(neg ? (base | 0x80u) : base);
    }
    if (mag == 0x7f800000u) { return (uint8_t)(neg ? 0xf8u : 0x78u); }
    float abs = bit_cast<float>(mag);
    if (abs == 0.0f) { return (uint8_t)(neg ? 0x80u : 0x00u); }
    if (abs >= 248.0f) { return (uint8_t)(neg ? 0xf8u : 0x78u); }
    int exp = f8e4m3_fl2(abs);
    uint bits;
    if (exp < -6)
    {
        uint64_t scaled = f8e4m3_rti(abs, -9);
        bits = (uint)(scaled < (uint64_t)8u ? scaled : (uint64_t)8u);
    }
    else
    {
        int ve = exp;
        uint64_t sig = f8e4m3_rti(abs, ve - 3);
        if (sig >= (uint64_t)16u) { sig >>= 1; ve += 1; }
        uint exponent = uint(ve + 7);
        uint mantissa = (uint)(sig - (uint64_t)8u);
        bits = (exponent << 3) | mantissa;
    }
    return (uint8_t)(neg ? (bits | 0x80u) : bits);
}

#if defined(FP8_NATIVE)
    // float -> FloatE4M3 (native encode: round-to-nearest-even bit
    // pattern, bit_cast into the native type).
    public FloatE4M3 f8e4m3_encode(float v)
    {
        return bit_cast<FloatE4M3>(f8e4m3_encode_bits(v));
    }

    // FloatE4M3 -> float (native decode: bit_cast the pattern out, then
    // decode).
    public float f8e4m3_decode(FloatE4M3 b)
    {
        return f8e4m3_decode_bits(bit_cast<uint8_t>(b));
    }
#else
    public float f8e4m3_decode(uint8_t b)
    {
        return f8e4m3_decode_bits(b);
    }

    public uint8_t f8e4m3_encode(float v)
    {
        return f8e4m3_encode_bits(v);
    }
#endif
