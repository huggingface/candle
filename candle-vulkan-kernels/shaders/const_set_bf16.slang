// Fills a bf16 (emulated u16) storage buffer with a constant value.
//
// Push constants: g_Params.count = element count, g_Params.value = fill value.
RWStructuredBuffer<uint16_t> g_Out;
struct Params
{
    uint64_t count;
    float value;
};
layout(push_constant) Params g_Params;
// bf16 helpers (half crate f32_to_bf16, round-to-nearest-even).
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
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        g_Out[idx] = bf16_encode(g_Params.value);
    }
}
