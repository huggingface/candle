// BFloat16 scalar type and encode/decode, native or emulated.
//
// Compiled with -DBF16_NATIVE=1 on devices that enable the
// VK_KHR_shader_bfloat16 extension and the shaderBFloat16Type device
// feature: `BF16` is the native Slang `BFloat16` type, stored directly in
// `StructuredBuffer<BFloat16>` / `RWStructuredBuffer<BFloat16>`. The
// 16-bit in-memory layout of `BFloat16` is exactly the `half::bf16`
// bit pattern, so the encode/decode below convert through the raw
// 16-bit pattern with `bit_cast` — the same `half` crate algorithms
// the CPU backend uses, bit for bit.
//
// Otherwise (the default) `BF16` is the 16-bit bit pattern
// (`uint16_t`) and the same encode/decode functions apply to it
// directly.
module bf16_type;

#if defined(BF16_NATIVE)
    public typealias BF16 = BFloat16;
#else
    public typealias BF16 = uint16_t;
#endif

// bf16 bit pattern -> float (half crate bf16_to_f32, exact widening).
public float bf16_decode_bits(uint16_t b)
{
    uint x = (uint)b;
    if ((x & 0x7FFFu) > 0x7F80u) { return bit_cast<float>(x | 0x40u); }
    return bit_cast<float>(x << 16);
}

// float -> bf16 bit pattern (half crate f32_to_bf16, round-to-nearest-even).
public uint16_t bf16_encode_bits(float v)
{
    uint x = bit_cast<uint>(v);
    if ((x & 0x7FFFFFFFu) > 0x7F800000u) { return (uint16_t)((x >> 16) | 0x0040u); }
    uint rb = 0x8000u;
    if ((x & rb) != 0u && (x & (3u * rb - 1u)) != 0u) { return (uint16_t)((x >> 16) + 1u); }
    return (uint16_t)(x >> 16);
}

#if defined(BF16_NATIVE)
    // Native BFloat16 -> 16-bit bit pattern (half::bf16 layout).
    public uint16_t bf16_bits(BFloat16 b)
    {
        return bit_cast<uint16_t>(b);
    }

    // 16-bit bit pattern (half::bf16 layout) -> native BFloat16.
    public BFloat16 bf16_from_bits(uint16_t bits)
    {
        return bit_cast<BFloat16>(bits);
    }

    // float -> BFloat16 (native encode: round-to-nearest-even bit pattern,
    // bit_cast into the native type).
    public BFloat16 bf16_encode(float v)
    {
        return bit_cast<BFloat16>(bf16_encode_bits(v));
    }

    // BFloat16 -> float (native decode: bit_cast the pattern out, then
    // widen).
    public float bf16_decode(BFloat16 b)
    {
        return bf16_decode_bits(bit_cast<uint16_t>(b));
    }
#else
    public float bf16_decode(uint16_t b)
    {
        return bf16_decode_bits(b);
    }

    public uint16_t bf16_encode(float v)
    {
        return bf16_encode_bits(v);
    }
#endif
