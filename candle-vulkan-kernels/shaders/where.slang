// where_cond: out[i] = pred[i] != 0 ? true[t_off(i)] : false[f_off(i)].
// All three operands share the same logical shape; true/false may be
// broadcast views (stride 0 -> dim 1).
//
// Params: [0]=total, [1]=ndim, [2..6]=pred dims, [6..10]=true dims,
//         [10..14]=false dims.
StructuredBuffer<uint8_t> g_Pred;
StructuredBuffer<float> g_True;
StructuredBuffer<float> g_False;
RWStructuredBuffer<float> g_Out;
StructuredBuffer<float> g_Params;

// Offset into the operand whose dims start at params[base], for flat
// index `i` decomposed over the pred (full logical) dims.
uint operand_offset(uint i, uint base)
{
    uint nd = uint(g_Params[1]);
    uint idx[4];
    uint rem = i;
    for (uint d = 0u; d < nd; d += 1u)
    {
        uint dim = uint(g_Params[2u + (nd - 1u - d)]);
        idx[nd - 1u - d] = rem % dim;
        rem = rem / dim;
    }
    uint off = 0u;
    for (uint j = 0u; j < nd; j += 1u)
    {
        uint r = uint(g_Params[base + j]);
        uint lidx = idx[j];
        off = off * r + (r != 1u ? lidx : 0u);
    }
    return off;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0]))
    {
        if (g_Pred[idx] != 0u)
        {
            g_Out[idx] = g_True[operand_offset(idx, 6u)];
        }
        else
        {
            g_Out[idx] = g_False[operand_offset(idx, 10u)];
        }
    }
}
