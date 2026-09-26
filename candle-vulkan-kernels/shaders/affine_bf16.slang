// affine (dst = src*mul + add) for bf16 (emulated): decode to float, compute, encode back.
StructuredBuffer<uint16_t> g_In;
RWStructuredBuffer<uint16_t> g_Out;
StructuredBuffer<float> g_Params; // [0]=total, [1]=mul, [2]=add

// bf16 -> float (half crate bf16_to_f32).
float bf16_decode(uint16_t b)
{
    uint x = (uint)b;
    if ((x & 0x7FFFu) > 0x7F80u) { return bit_cast<float>(x | 0x40u); }
    return bit_cast<float>(x << 16);
}

// float -> bf16 (half crate f32_to_bf16, round-to-nearest-even).
uint16_t bf16_encode(float v)
{
    uint x = bit_cast<uint>(v);
    if ((x & 0x7FFFFFFFu) > 0x7F800000u) { return (uint16_t)((x >> 16) | 0x0040u); }
    uint rb = 0x8000u;
    if ((x & rb) != 0u && (x & (3u * rb - 1u)) != 0u) { return (uint16_t)((x >> 16) + 1u); }
    return (uint16_t)(x >> 16);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_affine(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    if (i < uint(g_Params[0]))
    {
        float x = bf16_decode(g_In[i]);
        g_Out[i] = bf16_encode(x * g_Params[1] + g_Params[2]);
    }
}
