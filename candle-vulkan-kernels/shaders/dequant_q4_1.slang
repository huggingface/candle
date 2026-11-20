// dequantize _q4_1
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

    uint off = b * 20u;
    float d = f16_bits_to_f32(uint16_t(g_Src[off]) | (uint16_t(g_Src[off + 1]) << 8));
    float m = f16_bits_to_f32(uint16_t(g_Src[off + 2]) | (uint16_t(g_Src[off + 3]) << 8));
    if (e < 16u)
    {
        uint j = e;
        uint q = g_Src[off + 4u + j];
        float x0 = float(int(q & 0x0Fu));
        g_Dst[b * 32u + e] = x0 * d + m;
    }
    else
    {
        uint j = e - 16u;
        uint q = g_Src[off + 4u + j];
        float x1 = float(int((q >> 4) & 0x0Fu));
        g_Dst[b * 32u + e] = x1 * d + m;
    }

}
