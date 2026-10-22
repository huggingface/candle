// dequantize _f16
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

    uint off = b * 2u;
    uint16_t h = uint16_t(g_Src[off]) | (uint16_t(g_Src[off + 1]) << 8);
    g_Dst[b] = f16_bits_to_f32(h);

}
