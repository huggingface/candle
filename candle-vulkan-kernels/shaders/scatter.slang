// Scatter over F32 data, U32 index buffer:
// dst[left, idx, right] = src[left, i, right]
// params: [0]=dst_dim_len, [1]=dst_right_len, [2]=ids_dim_len,
//         [3]=ids_right_len, [4]=ids_left_len
import common_scatter;
StructuredBuffer<float> g_Src;
RWStructuredBuffer<float> g_Dst;
StructuredBuffer<uint> g_Ids;
layout(push_constant) Params g_Params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main_scatter(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint o = dispatchThreadID.x;
    uint64_t dst_dim_len = g_Params.dst_dim_len;
    uint64_t dst_right_len = g_Params.dst_right_len;
    uint64_t ids_dim_len = g_Params.ids_dim_len;
    uint64_t ids_right_len = g_Params.ids_right_len;
    uint64_t ids_left_len = g_Params.ids_left_len;
    uint64_t inner = ids_dim_len * ids_right_len;
    if (inner == 0u || o >= ids_left_len * inner)
    {
        return;
    }
    uint64_t left_i = o / inner;
    uint64_t rem = o - left_i * inner;
    uint64_t i = rem / ids_right_len;
    uint64_t right_i = rem - i * ids_right_len;
    uint idx = g_Ids[o];
    if (idx == 0xFFFFFFFFu || idx >= dst_dim_len)
    {
        return;
    }
    uint64_t dst_idx = left_i * dst_dim_len * dst_right_len + idx * dst_right_len + right_i;
    g_Dst[dst_idx] = g_Src[o];
}
