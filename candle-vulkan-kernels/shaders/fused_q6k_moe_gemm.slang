// Fused Q6_K routed MoE GEMM:
// out[route, n] = lhs[route, k] @ dequant(W[ids[route], n, k]).
//
// The routed quantized MoE projection: for each route (a flattened
// (token, topk) pair), compute the dequantized GEMV of the route's input
// row against the selected expert's weights, reading the quantized bytes
// on the fly (no f32 materialization of the expert weights). Each thread
// owns one (route, output column); the dequantization happens in
// registers, so no shared memory or barriers are needed. W holds the raw
// Q6_K bytes of all experts, (experts, n, k)-contiguous (256 elements
// per 210 bytes: 128 4-bit low quants at [0..128), 64 2-bit high quants
// at [128..192), 16 i8 scales at [192..208), an f16 super-scale at
// [208..210)). `weight_expert_bytes` is the byte size of one expert's
// (n, k) slice. k must be a multiple of 256; n need not be a multiple of
// 32 (out-of-range columns early-exit).
//
// The per-element decode is the Q6_K layout from candle
// quantized/k_quants.rs BlockQ6K::to_float.
//
// Params: [0]=n, [1]=k, [2]=weight_expert_bytes.
import dequant_helpers;
import common_moe_gemm;

StructuredBuffer<float> g_Lhs; // (num_routes, k) f32
RWStructuredBuffer<float> g_Out; // (num_routes, n) f32
StructuredBuffer<uint8_t> g_W; // raw Q6_K bytes, (experts, n, k)
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
    // Four independent accumulators break the serial FMA chain (ILP).
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint bb = 0u; bb < kblocks; ++bb)
    {
        uint64_t off = wbase + (col * kblocks + bb) * 210u;
        float d =
            f16_bits_to_f32(uint16_t(g_W[off + 208u]) | (uint16_t(g_W[off + 209u]) << 8u));
        for (uint idx = 0u; idx < 2u; ++idx)
        {
            for (uint l = 0u; l < 32u; ++l)
            {
                uint is = l / 16u;
                uint ql1 = g_W[off + 64u * idx + l];
                uint ql2 = g_W[off + 64u * idx + l + 32u];
                uint qh = g_W[off + 128u + 32u * idx + l];
                int q1 = int((ql1 & 0xFu) | ((qh & 3u) << 4)) - 32;
                int q2 = int((ql2 & 0xFu) | (((qh >> 2u) & 3u) << 4)) - 32;
                int q3 = int((ql1 >> 4u) | (((qh >> 4u) & 3u) << 4)) - 32;
                int q4 = int((ql2 >> 4u) | (((qh >> 6u) & 3u) << 4)) - 32;
                uint64_t sc = off + 192u + 8u * idx + is;
                float s1 = d * float(sbyte(g_W[sc]));
                float s2 = d * float(sbyte(g_W[sc + 2u]));
                float s3 = d * float(sbyte(g_W[sc + 4u]));
                float s4 = d * float(sbyte(g_W[sc + 6u]));
                uint k0 = bb * 256u + 128u * idx + l;
                a0 += g_Lhs[lhs_base + k0] * (s1 * float(q1));
                a1 += g_Lhs[lhs_base + k0 + 32u] * (s2 * float(q2));
                a2 += g_Lhs[lhs_base + k0 + 64u] * (s3 * float(q3));
                a3 += g_Lhs[lhs_base + k0 + 96u] * (s4 * float(q4));
            }
        }
    }
    g_Out[route * (uint)n + col] = a0 + a1 + a2 + a3;
}
