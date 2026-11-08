// Fused Q2_K routed MoE GEMM:
// out[route, n] = lhs[route, k] @ dequant(W[ids[route], n, k]).
//
// The routed quantized MoE projection: for each route (a flattened
// (token, topk) pair), compute the dequantized GEMV of the route's input
// row against the selected expert's weights, reading the quantized bytes
// on the fly (no f32 materialization of the expert weights). Each thread
// owns one (route, output column); the dequantization happens in
// registers, so no shared memory or barriers are needed. W holds the raw
// Q2_K bytes of all experts, (experts, n, k)-contiguous (256 elements
// per 84 bytes: 16 4-bit scale/min bytes at [0..16), 128 2-bit quants at
// [16..84), an f16 super-scale at [80..82), an f16 super-min at
// [82..84)). `weight_expert_bytes` is the byte size of one expert's
// (n, k) slice. k must be a multiple of 256; n need not be a multiple of
// 32 (out-of-range columns early-exit).
//
// The per-element decode is the Q2_K layout from candle
// quantized/k_quants.rs BlockQ2K::to_float.
//
// Params: [0]=n, [1]=k, [2]=weight_expert_bytes.
import dequant_helpers;
import common_moe_gemm;

StructuredBuffer<float> g_Lhs; // (num_routes, k) f32
RWStructuredBuffer<float> g_Out; // (num_routes, n) f32
StructuredBuffer<uint8_t> g_W; // raw Q2_K bytes, (experts, n, k)
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
    uint kblocks = (uint)k / 256u;
    uint lhs_base = route * (uint)k;
    float acc = 0.0;
    for (uint bb = 0u; bb < kblocks; ++bb)
    {
        uint64_t off = wbase + (col * kblocks + bb) * 84u;
        float d =
            f16_bits_to_f32(uint16_t(g_W[off + 80u]) | (uint16_t(g_W[off + 81u]) << 8u));
        float mn =
            f16_bits_to_f32(uint16_t(g_W[off + 82u]) | (uint16_t(g_W[off + 83u]) << 8u));
        for (uint e = 0u; e < 256u; ++e)
        {
            uint is = e / 16u;
            uint isw = is % 8u;
            uint sc = g_W[off + is];
            uint shift = 2u * (isw / 2u);
            uint64_t qbyte = off + 16u + (e / 128u) * 32u + (e % 32u);
            uint v = (g_W[qbyte] >> shift) & 3u;
            float dl = d * float(sc & 0xFu);
            float ml = mn * float((sc >> 4) & 0xFu);
            acc += g_Lhs[lhs_base + bb * 256u + e] * (dl * float(v) - ml);
        }
    }
    g_Out[route * (uint)n + col] = acc;
}
