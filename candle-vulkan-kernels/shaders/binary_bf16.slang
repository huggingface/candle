// Broadcast bf16 binary ops (add/sub/mul/div/maximum/minimum).
//
// bf16 is stored as its 16-bit bit pattern (uint16_t). Add/Sub/Mul/Div decode
// both operands to float, compute in float, and encode the result back to bf16,
// matching the half crate's `bf16` arithmetic: from_f32(to_f32(a) OP to_f32(b)).
// Maximum/Minimum compare the decoded floats and write back the *original* bit
// pattern of the winner (|v1 > v2 { v2 } else { v1 }), matching candle's CPU.
//
// Params layout: [0]=total, [1]=lhs_ndim, [2]=rhs_ndim,
//                [3..11]=lhs dims, [11..19]=rhs dims,
//                [19]=lhs_data_ndim, [20..28]=lhs_data_dims.
import bf16_type;

StructuredBuffer<BF16> g_In;
StructuredBuffer<BF16> g_Rhs;
RWStructuredBuffer<BF16> g_Out;
StructuredBuffer<uint64_t> g_Params;

uint64_t rhs_offset(uint i)
{
    uint64_t rhs_ndim = g_Params[2];
    if (rhs_ndim == 0u) { return 0u; }
    uint64_t lhs_ndim = g_Params[1];
    uint64_t idx[8];
    uint64_t rem = i;
    for (uint d = 0u; d < lhs_ndim; d += 1u)
    {
        uint64_t dim = g_Params[3u + (lhs_ndim - 1u - d)];
        idx[lhs_ndim - 1u - d] = rem % dim;
        rem = rem / dim;
    }
    uint64_t off = 0u;
    for (uint j = 0u; j < rhs_ndim; j += 1u)
    {
        uint64_t r = g_Params[11u + j];
        uint64_t lidx = idx[lhs_ndim - rhs_ndim + j];
        off = off * r + (r != 1u ? lidx : 0u);
    }
    return off;
}

uint64_t lhs_offset(uint i)
{
    // Map the output linear index to an index into the (possibly broadcast)
    // lhs data. Decompose over the output dims, then re-encode over the lhs
    // data dims (broadcast dims of 1 contribute 0).
    uint64_t lhs_ndim = g_Params[1];
    uint64_t lhs_data_ndim = g_Params[19];
    if (lhs_data_ndim == 0u) { return 0u; }
    uint64_t idx[8];
    uint64_t rem = i;
    for (uint d = 0u; d < lhs_ndim; d += 1u)
    {
        uint64_t dim = g_Params[3u + (lhs_ndim - 1u - d)];
        idx[lhs_ndim - 1u - d] = rem % dim;
        rem = rem / dim;
    }
    uint64_t off = 0u;
    for (uint j = 0u; j < lhs_data_ndim; j += 1u)
    {
        uint64_t l = g_Params[20u + j];
        uint64_t lidx = idx[lhs_ndim - lhs_data_ndim + j];
        off = off * l + (l != 1u ? lidx : 0u);
    }
    return off;
}
[shader("compute")]
[numthreads(256, 1, 1)]
void main_maximum(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < g_Params[0])
    {
        float a = bf16_decode(g_In[lhs_offset(idx)]);
        float b = bf16_decode(g_Rhs[rhs_offset(idx)]);
        g_Out[idx] = (a < b) ? g_Rhs[rhs_offset(idx)] : g_In[lhs_offset(idx)];
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_minimum(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < g_Params[0])
    {
        float a = bf16_decode(g_In[lhs_offset(idx)]);
        float b = bf16_decode(g_Rhs[rhs_offset(idx)]);
        g_Out[idx] = (a > b) ? g_Rhs[rhs_offset(idx)] : g_In[lhs_offset(idx)];
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_add(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < g_Params[0])
    {
        g_Out[idx] = bf16_encode(bf16_decode(g_In[lhs_offset(idx)]) + bf16_decode(g_Rhs[rhs_offset(idx)]));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_sub(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < g_Params[0])
    {
        g_Out[idx] = bf16_encode(bf16_decode(g_In[lhs_offset(idx)]) - bf16_decode(g_Rhs[rhs_offset(idx)]));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_mul(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < g_Params[0])
    {
        g_Out[idx] = bf16_encode(bf16_decode(g_In[lhs_offset(idx)]) * bf16_decode(g_Rhs[rhs_offset(idx)]));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_div(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < g_Params[0])
    {
        g_Out[idx] = bf16_encode(bf16_decode(g_In[lhs_offset(idx)]) / bf16_decode(g_Rhs[rhs_offset(idx)]));
    }
}
