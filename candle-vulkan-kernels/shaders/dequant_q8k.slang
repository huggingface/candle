// dequantize _q8k
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

    uint off = b * 292u;
    uint32_t dbits = uint32_t(g_Src[off]) | (uint32_t(g_Src[off + 1]) << 8)
        | (uint32_t(g_Src[off + 2]) << 16) | (uint32_t(g_Src[off + 3]) << 24);
    float d = asfloat(dbits);
    for (uint j = 0u; j < 256u; ++j)
    {
        int q = sbyte(g_Src[off + 4u + j]);
        g_Dst[b * 256u + j] = float(q) * d;
    }

}
