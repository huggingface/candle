// Fused Q8_0 routed MoE GEMM:
// out[route, n] = lhs[route, k] @ dequant(W[ids[route], n, k]).
//
// The routed quantized MoE projection: for each route (a flattened
// (token, topk) pair), compute the dequantized GEMV of the route's input
// row against the selected expert's weights, reading the quantized bytes
// on the fly (no f32 materialization of the expert weights). Each thread
// owns one (route, output column); the dequantization happens in
// registers, so no shared memory or barriers are needed. W holds the raw
// Q8_0 bytes of all experts, (experts, n, k)-contiguous (32 elements per
// 34 bytes: a 2-byte f16 scale then 32 i8 quants). `weight_expert_bytes`
// is the byte size of one expert's (n, k) slice. k must be a multiple of
// 32; n need not be a multiple of 32 (out-of-range columns early-exit).
//
// Params: [0]=n, [1]=k, [2]=weight_expert_bytes.
import dequant_helpers;
import common_moe_gemm;

StructuredBuffer<float> g_Lhs; // (num_routes, k) f32
RWStructuredBuffer<float> g_Out; // (num_routes, n) f32
StructuredBuffer<uint8_t> g_W; // raw Q8_0 bytes, (experts, n, k)
StructuredBuffer<uint> g_Ids; // (num_routes,) u32 expert ids
layout(push_constant) Params g_Params;

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID : SV_GroupID)
{
    uint64_t n = g_Params.n;
    uint64_t k = g_Params.k;
    uint64_t wexp = g_Params.weight_expert_bytes;
    uint col = groupID.y * 32u + dispatchThreadID.x;
    if (col >= n)
    {
        return;
    }
    uint route = groupID.z;
    uint expert = g_Ids[route];
    uint64_t wbase = (uint64_t)expert * wexp;
    uint kblocks = (uint)k / 32u;
    uint lhs_base = route * (uint)k;
    // Four independent accumulators break the serial FMA chain (ILP).
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint bb = 0u; bb < kblocks; ++bb)
    {
        uint64_t off = wbase + (col * kblocks + bb) * 34u;
        float d =
            f16_bits_to_f32(uint16_t(g_W[off]) | (uint16_t(g_W[off + 1u]) << 8u));
        for (uint j = 0u; j < 32u; ++j)
        {
            int q = sbyte(g_W[off + 2u + j]);
            float val = g_Lhs[lhs_base + bb * 32u + j] * (float(q) * d);
            uint r = j & 3u;
            if (r == 0u)
            {
                a0 += val;
            }
            else if (r == 1u)
            {
                a1 += val;
            }
            else if (r == 2u)
            {
                a2 += val;
            }
            else
            {
                a3 += val;
            }
        }
    }
    g_Out[route * (uint)n + col] = a0 + a1 + a2 + a3;
}
