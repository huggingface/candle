// 2D row-wise copy (values in elements):
// for row in 0..d1: dst[dst_o + row*dst_s .. +d2] = src[src_o + row*src_s .. +d2].
// Also used for copy_strided_src with a UniformBlocks source (dst_s = d2).
import common_copy2d;
StructuredBuffer<double> g_Src;
RWStructuredBuffer<double> g_Dst;
layout(push_constant) Params g_Params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    uint64_t d1 = g_Params.d1;
    uint64_t d2 = g_Params.d2;
    if (d2 == 0u || i >= d1 * d2)
    {
        return;
    }
    uint64_t row = i / d2;
    uint64_t col = i % d2;
    uint64_t si = g_Params.src_offset + row * g_Params.src_stride + col;
    uint64_t di = g_Params.dst_offset + row * g_Params.dst_stride + col;
    g_Dst[di] = g_Src[si];
}
