// dequantize _q8_0
import common_dequant;
import dequant_helpers;

StructuredBuffer<uint8_t> g_Src;
RWStructuredBuffer<float> g_Dst;
layout(push_constant) Params g_Params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint t = dispatchThreadID.x;
    uint b = t / 32u;
    uint e = t % 32u;
    if (b >= g_Params.num_blocks)
    {
        return;
    }

    uint off = b * 34u;
    float d = f16_bits_to_f32(uint16_t(g_Src[off]) | (uint16_t(g_Src[off + 1]) << 8));
    int q = sbyte(g_Src[off + 2u + e]);
    g_Dst[b * 32u + e] = float(q) * d;

}
