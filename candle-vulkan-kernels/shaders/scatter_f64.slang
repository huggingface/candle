// Scatter over F32 data, U32 index buffer:
// dst[left, idx, right] = src[left, i, right]
// params: [0]=dst_dim_len, [1]=dst_right_len, [2]=ids_dim_len,
//         [3]=ids_right_len, [4]=ids_left_len
StructuredBuffer<double> g_Src;
RWStructuredBuffer<double> g_Dst;
StructuredBuffer<uint> g_Ids;
StructuredBuffer<float> g_Params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main_scatter(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint o = dispatchThreadID.x;
    uint dst_dim_len = uint(g_Params[0]);
    uint dst_right_len = uint(g_Params[1]);
    uint ids_dim_len = uint(g_Params[2]);
    uint ids_right_len = uint(g_Params[3]);
    uint ids_left_len = uint(g_Params[4]);
    uint inner = ids_dim_len * ids_right_len;
    if (inner == 0u || o >= ids_left_len * inner)
    {
        return;
    }
    uint left_i = o / inner;
    uint rem = o - left_i * inner;
    uint i = rem / ids_right_len;
    uint right_i = rem - i * ids_right_len;
    uint idx = g_Ids[o];
    if (idx == 0xFFFFFFFFu || idx >= dst_dim_len)
    {
        return;
    }
    uint dst_idx = left_i * dst_dim_len * dst_right_len + idx * dst_right_len + right_i;
    g_Dst[dst_idx] = g_Src[o];
}
