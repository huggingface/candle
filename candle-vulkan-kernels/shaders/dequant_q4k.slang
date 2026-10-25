// dequantize _q4k
import common_dequant;
import dequant_helpers;

StructuredBuffer<uint8_t> g_Src;
RWStructuredBuffer<float> g_Dst;
layout(push_constant) Params g_Params;

// Q4K/Q5K scale-min decoder (from candle quantized/utils.rs::get_scale_min_k4).
// `off` is the block base offset in g_Src; `soff` is the scale-array offset.
uint2 get_scale_min_k4(uint j, uint off, uint soff)
{
    if (j < 4u)
    {
        uint d = g_Src[off + soff + j] & 63u;
        uint m = g_Src[off + soff + j + 4u] & 63u;
        return uint2(d, m);
    }
    else
    {
        uint d = (g_Src[off + soff + j + 4u] & 0xFu) | ((g_Src[off + soff + j - 4u] >> 6) << 4);
        uint m = (g_Src[off + soff + j + 4u] >> 4) | ((g_Src[off + soff + j] >> 6) << 4);
        return uint2(d, m);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint b = dispatchThreadID.x;
    if (b >= g_Params.num_blocks)
    {
        return;
    }

    uint off = b * 144u;
    float d = f16_bits_to_f32(uint16_t(g_Src[off]) | (uint16_t(g_Src[off + 1]) << 8));
    float mn = f16_bits_to_f32(uint16_t(g_Src[off + 2]) | (uint16_t(g_Src[off + 3]) << 8));
    for (uint e = 0u; e < 256u; ++e)
    {
        uint jstep = e / 64u;
        uint within = e % 64u;
        bool first = within < 32u;
        uint i = within % 32u;
        uint ql_byte = off + 16u + 32u * jstep + i;
        uint qv = first ? (g_Src[ql_byte] & 0xFu) : (g_Src[ql_byte] >> 4);
        uint2 sm = get_scale_min_k4(2u * jstep + (first ? 0u : 1u), off, 4u);
        float d1 = d * float(sm.x);
        float m1 = mn * float(sm.y);
        g_Dst[b * 256u + e] = d1 * float(qv) - m1;
    }

}
