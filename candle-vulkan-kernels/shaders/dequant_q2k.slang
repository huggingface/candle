// dequantize _q2k
import dequant_helpers;

StructuredBuffer<uint8_t> g_Src;
RWStructuredBuffer<float> g_Dst;
StructuredBuffer<uint64_t> g_Params; // [0]=num_blocks

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
    if (b >= g_Params[0])
    {
        return;
    }

    uint off = b * 84u;
    float d = f16_bits_to_f32(uint16_t(g_Src[off + 80u]) | (uint16_t(g_Src[off + 81u]) << 8));
    float mn = f16_bits_to_f32(uint16_t(g_Src[off + 82u]) | (uint16_t(g_Src[off + 83u]) << 8));
    for (uint e = 0u; e < 256u; ++e)
    {
        uint is = e / 16u;
        uint isw = is % 8u;
        uint sc = g_Src[off + is];
        uint shift = 2u * (isw / 2u);
        uint qbyte = off + 16u + (e / 128u) * 32u + (e % 32u);
        uint v = (g_Src[qbyte] >> shift) & 3u;
        float dl = d * float(sc & 0xFu);
        float ml = mn * float((sc >> 4) & 0xFu);
        g_Dst[b * 256u + e] = dl * float(v) - ml;
    }

}
