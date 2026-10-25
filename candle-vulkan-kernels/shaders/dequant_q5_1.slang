// dequantize _q5_1
import common_dequant;
import dequant_helpers;

StructuredBuffer<uint8_t> g_Src;
RWStructuredBuffer<float> g_Dst;
layout(push_constant) Params g_Params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint b = dispatchThreadID.x;
    if (b >= g_Params.num_blocks)
    {
        return;
    }

    uint off = b * 24u;
    float d = f16_bits_to_f32(uint16_t(g_Src[off]) | (uint16_t(g_Src[off + 1]) << 8));
    float m = f16_bits_to_f32(uint16_t(g_Src[off + 2]) | (uint16_t(g_Src[off + 3]) << 8));
    uint32_t qh = uint32_t(g_Src[off + 4]) | (uint32_t(g_Src[off + 5]) << 8)
        | (uint32_t(g_Src[off + 6]) << 16) | (uint32_t(g_Src[off + 7]) << 24);
    for (uint j = 0u; j < 16u; ++j)
    {
        uint q = g_Src[off + 8u + j];
        uint xh0 = ((qh >> j) << 4u) & 0x10u;
        uint xh1 = (qh >> (j + 12u)) & 0x10u;
        float x0 = float(int((q & 0x0Fu) | xh0));
        float x1 = float(int(((q >> 4) & 0x0Fu) | xh1));
        g_Dst[b * 32u + j] = x0 * d + m;
        g_Dst[b * 32u + 16u + j] = x1 * d + m;
    }

}
