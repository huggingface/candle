// dequantize _q8_1
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

    // candle's Q8_1 to_float is just q * d (the `s` field is unused in dequantize).
    uint off = b * 36u;
    float d = f16_bits_to_f32(uint16_t(g_Src[off]) | (uint16_t(g_Src[off + 1]) << 8));
    for (uint j = 0u; j < 32u; ++j)
    {
        int q = sbyte(g_Src[off + 4u + j]);
        g_Dst[b * 32u + j] = float(q) * d;
    }

}
