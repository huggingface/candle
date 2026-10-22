// bf16 emulation module: encode/decode between the 16-bit bf16 bit pattern
// (uint16_t) and float, matching the `half` crate.
//
// Used by the bf16 compute kernels, which store bf16 data as raw bit
// patterns and perform arithmetic in float:
// from_f32(to_f32(a) OP to_f32(b)).
module bf16_emulation;

// bf16 -> float (half crate bf16_to_f32).
public float bf16_decode(uint16_t b)
{
    uint x = (uint)b;
    if ((x & 0x7FFFu) > 0x7F80u) { return bit_cast<float>(x | 0x40u); }
    return bit_cast<float>(x << 16);
}

// float -> bf16 (half crate f32_to_bf16, round-to-nearest-even).
public uint16_t bf16_encode(float v)
{
    uint x = bit_cast<uint>(v);
    if ((x & 0x7FFFFFFFu) > 0x7F800000u) { return (uint16_t)((x >> 16) | 0x0040u); }
    uint rb = 0x8000u;
    if ((x & rb) != 0u && (x & (3u * rb - 1u)) != 0u) { return (uint16_t)((x >> 16) + 1u); }
    return (uint16_t)(x >> 16);
}
