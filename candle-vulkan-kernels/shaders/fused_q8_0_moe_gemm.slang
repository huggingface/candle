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
    uint wbase = expert * (uint)wexp;
    uint kblocks = (uint)k / 32u;
    uint lhs_base = route * (uint)k;
    float acc = 0.0;
    for (uint bb = 0u; bb < kblocks; ++bb)
    {
        uint off = wbase + (col * kblocks + bb) * 34u;
        float d =
            f16_bits_to_f32(uint16_t(g_W[off]) | (uint16_t(g_W[off + 1u]) << 8u));
        for (uint j = 0u; j < 32u; ++j)
        {
            int q = sbyte(g_W[off + 2u + j]);
            acc += g_Lhs[lhs_base + bb * 32u + j] * (float(q) * d);
        }
    }
    g_Out[route * (uint)n + col] = acc;
}
