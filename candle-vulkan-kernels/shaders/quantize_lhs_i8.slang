// Quantize a single-row (k,) f32 activation to signed i8 for DP4A.
//
// One 256-thread workgroup computes the per-row scale (max |x| of the row)
// with a shared-memory tree reduction, quantizes every element to i8, and
// writes the dequant scale (max/127) to `g_Scale[0]`. Out-of-range threads
// still participate in the barriers (uniform, radv-safe).
import common_unary;
StructuredBuffer<float> g_Lhs; // (k,) f32
RWStructuredBuffer<uint8_t> g_LhsQ; // (k,) i8 stored as 2's-complement bytes
RWStructuredBuffer<float> g_Scale; // (1,) f32 dequant scale
layout(push_constant) Params g_Params; // [0]=k
static groupshared float gs_max[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 localID : SV_GroupThreadID)
{
    uint k = uint(g_Params.count);
    uint i = dispatchThreadID.x;
    // Local max over this thread's strided elements.
    float local_mx = 0.0;
    for (uint j = i; j < k; j += 256u)
    {
        local_mx = max(local_mx, abs(g_Lhs[j]));
    }
    gs_max[localID.x] = local_mx;
    GroupMemoryBarrierWithGroupSync();
    // Tree reduction of the per-thread maxima.
    for (uint s = 128u; s > 0u; s >>= 1u)
    {
        if (localID.x < s)
        {
            gs_max[localID.x] = max(gs_max[localID.x], gs_max[localID.x + s]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float mx = gs_max[0];
    float qscale = (mx > 1e-6) ? (127.0 / mx) : 1.0;
    // Quantize each element to i8 (clamped), stored as a 2's-complement byte.
    for (uint j = i; j < k; j += 256u)
    {
        float v = clamp(round(g_Lhs[j] * qscale), -128.0, 127.0);
        g_LhsQ[j] = uint8_t(int(v));
    }
    if (i == 0u)
    {
        g_Scale[0] = (mx > 1e-6) ? (mx / 127.0) : 1.0;
    }
}