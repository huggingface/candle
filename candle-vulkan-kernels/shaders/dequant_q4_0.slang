// dequantize _q4_0
import dequant_helpers;

StructuredBuffer<uint8_t> g_Src;
RWStructuredBuffer<float> g_Dst;
StructuredBuffer<uint64_t> g_Params; // [0]=num_blocks

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint b = dispatchThreadID.x;
    if (b >= g_Params[0])
    {
        return;
    }

    uint off = b * 18u;
    float d = f16_bits_to_f32(uint16_t(g_Src[off]) | (uint16_t(g_Src[off + 1]) << 8));
    for (uint j = 0u; j < 16u; ++j)
    {
        uint q = g_Src[off + 2u + j];
        float x0 = float(int(q & 0x0Fu)) - 8.0;
        float x1 = float(int((q >> 4) & 0x0Fu)) - 8.0;
        g_Dst[b * 32u + j] = x0 * d;
        g_Dst[b * 32u + 16u + j] = x1 * d;
    }

}
