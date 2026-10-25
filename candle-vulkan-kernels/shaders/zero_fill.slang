// zero_fill: writes zero to every byte of g_Dst.
//
// Used to initialize device buffers that are allocated uninitialized (see
// VulkanStorage::new). Byte-granular so it works for every element type and
// buffer size.
//
// Params: [0] = byte length of g_Dst.
import common_zero_fill;
RWStructuredBuffer<uint8_t> g_Dst;
layout(push_constant) Params g_Params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint64_t i = dispatchThreadID.x;
    if (i < g_Params.count)
    {
        g_Dst[uint(i)] = 0;
    }
}
