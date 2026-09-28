// dequantize _f32

StructuredBuffer<uint8_t> g_Src;
RWStructuredBuffer<float> g_Dst;
StructuredBuffer<float> g_Params; // [0]=num_blocks

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint b = dispatchThreadID.x;
    if (b >= uint(g_Params[0]))
    {
        return;
    }


    uint off = b * 4u;
    uint32_t bits = uint32_t(g_Src[off]) | (uint32_t(g_Src[off + 1]) << 8)
        | (uint32_t(g_Src[off + 2]) << 16) | (uint32_t(g_Src[off + 3]) << 24);
    g_Dst[b] = asfloat(bits);

}
