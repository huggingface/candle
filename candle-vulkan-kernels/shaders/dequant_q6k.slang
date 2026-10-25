// dequantize _q6k
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

    uint off = b * 210u;
    float d = f16_bits_to_f32(uint16_t(g_Src[off + 208u]) | (uint16_t(g_Src[off + 209u]) << 8));
    for (uint e = 0u; e < 256u; ++e)
    {
        uint idx = e / 128u;
        uint l = e % 32u;
        uint group = (e % 128u) / 32u;
        uint is = l / 16u;
        uint sc_idx = 8u * idx + is + 2u * group;
        uint ql_byte = off + 64u * idx + l + 32u * (group % 2u);
        uint qh_byte = off + 128u + 32u * idx + l;
        uint qlv = g_Src[ql_byte];
        uint qhv = g_Src[qh_byte];
        uint low4 = (group < 2u) ? (qlv & 0xFu) : (qlv >> 4);
        uint high2;
        if (group == 0u)
        {
            high2 = qhv & 3u;
        }
        else if (group == 1u)
        {
            high2 = (qhv >> 2) & 3u;
        }
        else if (group == 2u)
        {
            high2 = (qhv >> 4) & 3u;
        }
        else
        {
            high2 = (qhv >> 6) & 3u;
        }
        int q6 = int(low4 | (high2 << 4)) - 32;
        int scv = sbyte(g_Src[off + 192u + sc_idx]);
        g_Dst[b * 256u + e] = d * float(scv) * float(q6);
    }

}
