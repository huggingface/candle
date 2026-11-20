// const_fill_bytes: writes the 4-byte `pattern` (repeated byte-wise) to
// every byte of g_Dst.
//
// Used to finish constant fills of integer-typed regions whose byte length
// is not a multiple of 4 (the 1-3 trailing bytes the transfer commands
// cannot reach), and as the whole-region fallback for small constant fills.
import common_const_fill_bytes;
RWStructuredBuffer<uint8_t> g_Dst;
layout(push_constant) Params g_Params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint64_t i = dispatchThreadID.x;
    if (i < g_Params.count)
    {
        g_Dst[uint(i)] = (uint8_t)((g_Params.pattern >> (8u * (i & 3u))) & 0xFFu);
    }
}
