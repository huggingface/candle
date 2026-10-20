// u8 copy (raw byte pattern).
// 2D row-wise copy (values in elements):
// for row in 0..d1: dst[dst_o + row*dst_s .. +d2] = src[src_o + row*src_s .. +d2].
// Also used for copy_strided_src with a UniformBlocks source (dst_s = d2).
StructuredBuffer<uint8_t> g_Src;
RWStructuredBuffer<uint8_t> g_Dst;
StructuredBuffer<uint64_t> g_Params; // [0]=d1, [1]=d2, [2]=src_stride, [3]=dst_stride, [4]=src_offset, [5]=dst_offset

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    uint64_t d1 = g_Params[0];
    uint64_t d2 = g_Params[1];
    if (d2 == 0u || i >= d1 * d2)
    {
        return;
    }
    uint64_t row = i / d2;
    uint64_t col = i % d2;
    uint64_t si = g_Params[4] + row * g_Params[2] + col;
    uint64_t di = g_Params[5] + row * g_Params[3] + col;
    g_Dst[di] = g_Src[si];
}
