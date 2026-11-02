// Fused Q3_K routed MoE GEMM:
// out[route, n] = lhs[route, k] @ dequant(W[ids[route], n, k]).
//
// The routed quantized MoE projection: for each route (a flattened
// (token, topk) pair), compute the dequantized GEMV of the route's input
// row against the selected expert's weights, reading the quantized bytes
// on the fly (no f32 materialization of the expert weights). Each thread
// owns one (route, output column); the dequantization happens in
// registers, so no shared memory or barriers are needed. W holds the raw
// Q3_K bytes of all experts, (experts, n, k)-contiguous (256 elements per
// 110 bytes: 32 high-quant bytes at [0..32), 64 2-bit quants at
// [32..96), 12 packed scale bytes at [96..108), an f16 super-scale at
// [108..110)). `weight_expert_bytes` is the byte size of one expert's
// (n, k) slice. k must be a multiple of 256; n need not be a multiple of
// 32 (out-of-range columns early-exit).
//
// The per-element decode is the Q3_K layout from candle
// quantized/k_quants.rs BlockQ3K::to_float.
//
// Params: [0]=n, [1]=k, [2]=weight_expert_bytes.
import dequant_helpers;
import common_moe_gemm;

StructuredBuffer<float> g_Lhs; // (num_routes, k) f32
RWStructuredBuffer<float> g_Out; // (num_routes, n) f32
StructuredBuffer<uint8_t> g_W; // raw Q3_K bytes, (experts, n, k)
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
    uint kblocks = (uint)k / 256u;
    uint lhs_base = route * (uint)k;
    float acc = 0.0;
    for (uint bb = 0u; bb < kblocks; ++bb)
    {
        uint off = wbase + (col * kblocks + bb) * 110u;
        float d =
            f16_bits_to_f32(uint16_t(g_W[off + 108u]) | (uint16_t(g_W[off + 109u]) << 8u));
        uint KMASK1 = 0x03030303u;
        uint KMASK2 = 0x0F0F0F0Fu;
        uint a0 = uint32_t(g_W[off + 96u]) | (uint32_t(g_W[off + 97u]) << 8)
            | (uint32_t(g_W[off + 98u]) << 16) | (uint32_t(g_W[off + 99u]) << 24);
        uint a1 = uint32_t(g_W[off + 100u]) | (uint32_t(g_W[off + 101u]) << 8)
            | (uint32_t(g_W[off + 102u]) << 16) | (uint32_t(g_W[off + 103u]) << 24);
        uint a2 = uint32_t(g_W[off + 104u]) | (uint32_t(g_W[off + 105u]) << 8)
            | (uint32_t(g_W[off + 106u]) << 16) | (uint32_t(g_W[off + 107u]) << 24);
        uint tmp = a2;
        a2 = ((a0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        uint a3 = ((a1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        a0 = (a0 & KMASK2) | (((tmp) & KMASK1) << 4);
        a1 = (a1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
        for (uint e = 0u; e < 256u; ++e)
        {
            uint yc = e / 128u;
            uint sl = e % 128u;
            uint s = sl / 32u;
            uint w32 = sl % 32u;
            uint scidx = w32 / 16u;
            uint ii = w32 % 16u;
            uint is = yc * 8u + s * 2u + scidx;
            uint shift = 2u * s;
            uint m = 1u << (yc * 4u + s);
            uint qbyte = off + 32u + yc * 32u + ii + 16u * scidx;
            uint hbyte = ii + 16u * scidx;
            uint v = (g_W[qbyte] >> shift) & 3u;
            uint correction = (g_W[off + hbyte] & m) == 0u ? 4u : 0u;
            uint word =
                (is / 4u) == 0u ? a0 : (is / 4u) == 1u ? a1 : (is / 4u) == 2u ? a2 : a3;
            int scale_val = sbyte((word >> (8u * (is % 4u))) & 0xFFu);
            acc +=
                g_Lhs[lhs_base + bb * 256u + e] *
                (d * float(scale_val - 32) * float(int(v) - int(correction)));
        }
    }
    g_Out[route * (uint)n + col] = acc;
}
