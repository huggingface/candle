// Tiled f32 GEMM: out[b, m, n] = lhs[b, m, k] @ rhs[b, k, n].
//
// The rhs may be stored transposed as (b, n, k); params[4]
// (rhs_transposed, 0/1) selects the indexing. 16x16 workgroup tiles
// over 16-wide slabs of the reduction dimension; m, n and k need not
// be multiples of 16 (out-of-range elements are zero-filled in the
// tiles).
//
// 2x2 register micro-tile: the 8x8 workgroup threads each own a 2x2
// block of outputs, so every loaded lhs/rhs tile cell feeds four FMAs
// (2x the arithmetic intensity of one-output-per-thread). Reduction
// values are summed into four independent accumulators to further
// break the FMA dependency chain (ILP).
//
// Barrier discipline: GroupMemoryBarrierWithGroupSync() is UB in
// divergent control flow (Vulkan spec 15.5.3), and RADV miscompiles
// it there. Every thread therefore executes both barriers on every
// slab iteration, regardless of whether its outputs are in range;
// out-of-range threads still load/zero-fill their tile cells (so
// in-range readers see valid data) and skip only the output stores.
//
// Params: [0]=batch, [1]=m, [2]=n, [3]=k, [4]=rhs_transposed (0/1)
import common_gemm;
StructuredBuffer<float> g_Lhs;
StructuredBuffer<float> g_Rhs;
RWStructuredBuffer<float> g_Out;
layout(push_constant) Params g_Params;
static groupshared float a_tile[16 * 16];
static groupshared float b_tile[16 * 16];
[shader("compute")]
[numthreads(8, 8, 1)]
void main_gemm(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID : SV_GroupID, uint3 localID : SV_GroupThreadID)
{
    uint batch = groupID.z;
    uint64_t m = g_Params.m;
    uint64_t n = g_Params.n;
    uint64_t k = g_Params.k;
    uint64_t rt = g_Params.rhs_transposed;
    // This thread's 2x2 output block.
    uint row0 = groupID.x * 16u + localID.y * 2u;
    uint row1 = row0 + 1u;
    uint col0 = groupID.y * 16u + localID.x * 2u;
    uint col1 = col0 + 1u;
    // Four independent accumulators (2x2 outputs, one each).
    float acc00 = 0.0, acc01 = 0.0, acc10 = 0.0, acc11 = 0.0;
    for (uint k0 = 0u; k0 < k; k0 += 16u)
    {
        // Cooperative load: 64 threads fill the 16x16 lhs slab (4 cells
        // each); out-of-range cells are zero-filled.
        for (uint r = 0u; r < 2u; ++r)
        {
            uint tr = localID.y + r * 8u;
            uint grow = groupID.x * 16u + tr;
            for (uint c = 0u; c < 2u; ++c)
            {
                uint tc = localID.x + c * 8u;
                uint a_k = k0 + tc;
                a_tile[tr * 16u + tc] =
                    (grow < m && a_k < k) ? g_Lhs[batch * m * k + grow * k + a_k] : 0.0;
            }
        }
        // Cooperative load of the 16x16 rhs slab (4 cells each).
        for (uint r = 0u; r < 2u; ++r)
        {
            uint tr = localID.y + r * 8u;
            uint b_k = k0 + tr;
            for (uint c = 0u; c < 2u; ++c)
            {
                uint tc = localID.x + c * 8u;
                uint gcol = groupID.y * 16u + tc;
                if (b_k < k && gcol < n)
                {
                    if (rt == 1u)
                    {
                        b_tile[tr * 16u + tc] = g_Rhs[batch * n * k + gcol * k + b_k];
                    }
                    else
                    {
                        b_tile[tr * 16u + tc] = g_Rhs[batch * k * n + b_k * n + gcol];
                    }
                }
                else
                {
                    b_tile[tr * 16u + tc] = 0.0;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        // Reduction: the 2 rows x 2 cols owned by this thread.
        uint ar0 = localID.y * 2u;
        uint ar1 = ar0 + 1u;
        uint bc0 = localID.x * 2u;
        uint bc1 = bc0 + 1u;
        for (uint kk = 0u; kk < 16u; ++kk)
        {
            float a0 = a_tile[ar0 * 16u + kk];
            float a1 = a_tile[ar1 * 16u + kk];
            float b0 = b_tile[kk * 16u + bc0];
            float b1 = b_tile[kk * 16u + bc1];
            acc00 += a0 * b0;
            acc01 += a0 * b1;
            acc10 += a1 * b0;
            acc11 += a1 * b1;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    uint64_t ob = batch * m * n;
    if (row0 < m && col0 < n)
    {
        g_Out[ob + uint64_t(row0) * n + col0] = acc00;
    }
    if (row0 < m && col1 < n)
    {
        g_Out[ob + uint64_t(row0) * n + col1] = acc01;
    }
    if (row1 < m && col0 < n)
    {
        g_Out[ob + uint64_t(row1) * n + col0] = acc10;
    }
    if (row1 < m && col1 < n)
    {
        g_Out[ob + uint64_t(row1) * n + col1] = acc11;
    }
}