// dequantize _f32

import common_dequant;
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


    uint off = b * 4u;
    uint32_t bits = uint32_t(g_Src[off]) | (uint32_t(g_Src[off + 1]) << 8)
        | (uint32_t(g_Src[off + 2]) << 16) | (uint32_t(g_Src[off + 3]) << 24);
    g_Dst[b] = asfloat(bits);

}
