// Tiled f32 GEMM: out[b, m, n] = lhs[b, m, k] @ rhs[b, k, n].
//
// The rhs may be stored transposed as (b, n, k); params[4]
// (rhs_transposed, 0/1) selects the indexing. 16x16 workgroup tiles
// over 16-wide slabs of the reduction dimension; m, n and k need not
// be multiples of 16 (out-of-range elements are zero-filled in the
// tiles).
//
// Barrier discipline: GroupMemoryBarrierWithGroupSync() is UB in
// divergent control flow (Vulkan spec 15.5.3), and RADV miscompiles
// it there. Every thread therefore executes both barriers on every
// slab iteration, regardless of whether its output element is in
// range; out-of-range threads still load/zero-fill their tile cells
// (so in-range readers see valid data) and skip only the reduction
// and the output store.
//
// Params: [0]=batch, [1]=m, [2]=n, [3]=k, [4]=rhs_transposed (0/1)
StructuredBuffer<float> g_Lhs;
StructuredBuffer<float> g_Rhs;
RWStructuredBuffer<float> g_Out;
StructuredBuffer<float> g_Params;
static groupshared float a_tile[16 * 16];
static groupshared float b_tile[16 * 16];
[shader("compute")]
[numthreads(16, 16, 1)]
void main_gemm(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID : SV_GroupID, uint3 localID : SV_GroupThreadID)
{
    uint batch = groupID.z;
    uint row = groupID.x * 16u + localID.y;
    uint col = groupID.y * 16u + localID.x;
    uint m = uint(g_Params[1]);
    uint n = uint(g_Params[2]);
    uint k = uint(g_Params[3]);
    uint rt = uint(g_Params[4]);
    bool in_range = row < m && col < n;
    float acc = 0.0;
    for (uint k0 = 0u; k0 < k; k0 += 16u)
    {
        uint a_k = k0 + localID.x;
        if (a_k < k && row < m)
        {
            a_tile[localID.y * 16u + localID.x] = g_Lhs[batch * m * k + row * k + a_k];
        }
        else
        {
            a_tile[localID.y * 16u + localID.x] = 0.0;
        }
        uint b_k = k0 + localID.y;
        if (b_k < k && col < n)
        {
            if (rt == 1u)
            {
                b_tile[localID.y * 16u + localID.x] = g_Rhs[batch * n * k + col * k + b_k];
            }
            else
            {
                b_tile[localID.y * 16u + localID.x] = g_Rhs[batch * k * n + b_k * n + col];
            }
        }
        else
        {
            b_tile[localID.y * 16u + localID.x] = 0.0;
        }
        GroupMemoryBarrierWithGroupSync();
        if (in_range)
        {
            for (uint kk = 0u; kk < 16u; kk += 1u)
            {
                acc += a_tile[localID.y * 16u + kk] * b_tile[kk * 16u + localID.x];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (in_range)
    {
        g_Out[batch * m * n + row * n + col] = acc;
    }
}
