// Fused Q4_K routed MoE GEMM:
// out[route, n] = lhs[route, k] @ dequant(W[ids[route], n, k]).
//
// The routed quantized MoE projection for the decode/prefill path: for
// each route (a flattened (token, topk) pair), compute the dequantized
// GEMV of the route's input row against the selected expert's weights,
// reading the quantized bytes on the fly (no f32 materialization of the
// expert weights). Each thread owns one (route, output column); the
// dequantization happens in registers, so no shared memory or barriers
// are needed. W holds the raw Q4_K bytes of all experts, (experts, n,
// k)-contiguous; `weight_expert_bytes` is the byte size of one expert's
// (n, k) slice. k must be a multiple of 256; n need not be a multiple of
// 32 (out-of-range columns early-exit).
//
// The weight is read through a uint (4-byte) view of the buffer: the 144
// byte Q4_K blocks are 4-byte aligned, so a block is 36 uints. Loading the
// 6-bit scale/min pairs and the 256 4-bit quants in 4-byte chunks cuts the
// per-value byte-load count by 4x, and the accumulator is split across four
// independent registers to break the long serial FMA dependency chain.
//
// Params: [0]=n, [1]=k, [2]=weight_expert_bytes.
import dequant_helpers;
import common_moe_gemm;

StructuredBuffer<float> g_Lhs; // (num_routes, k) f32
RWStructuredBuffer<float> g_Out; // (num_routes, n) f32
StructuredBuffer<uint> g_W; // raw Q4_K bytes (uint view), (experts, n, k)
StructuredBuffer<uint> g_Ids; // (num_routes,) u32 expert ids
layout(push_constant) Params g_Params;

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID : SV_GroupID)
{
    uint n = (uint)g_Params.n;
    uint k = (uint)g_Params.k;
    uint wexp = (uint)g_Params.weight_expert_bytes;
    uint col = groupID.y * 32u + dispatchThreadID.x;
    if (col >= n)
    {
        return;
    }
    uint route = groupID.z;
    uint expert = g_Ids[route];
    uint wbase = expert * wexp;
    uint kblocks = k / 256u;
    uint lhs_base = route * k;
    // Four independent accumulators break the serial FMA chain (ILP).
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint bb = 0u; bb < kblocks; ++bb)
    {
        uint u = (wbase + (col * kblocks + bb) * 144u) / 4u; // uint index of the block
        uint u0 = g_W[u];
        float d = f16_bits_to_f32(uint16_t(u0 & 0xFFFFu));
        float mn = f16_bits_to_f32(uint16_t((u0 >> 16u) & 0xFFFFu));
        uint u1 = g_W[u + 1u]; // bytes 4-7  : d1[0..4] (6-bit)
        uint u2 = g_W[u + 2u]; // bytes 8-11 : m1[0..4] (6-bit)
        uint u3 = g_W[u + 3u]; // bytes 12-15: d1[4..8]/m1[4..8] high bits
        for (uint s = 0u; s < 8u; ++s)
        {
            float d1s;
            float m1s;
            if (s < 4u)
            {
                d1s = float((u1 >> (8u * s)) & 63u);
                m1s = float((u2 >> (8u * s)) & 63u);
            }
            else
            {
                uint t = s - 4u;
                // Isolate each target byte before shifting; the unmasked 32-bit
                // values would otherwise leak higher bytes into the 6-bit scales.
                uint b1 = (u1 >> (8u * t)) & 0xFFu; // block byte 4 + t
                uint b2 = (u2 >> (8u * t)) & 0xFFu; // block byte 8 + t
                uint b3 = (u3 >> (8u * t)) & 0xFFu; // block byte 12 + t
                d1s = float((b3 & 0xFu) | ((b1 >> 6u) << 4u));
                m1s = float(((b3 >> 4u) & 0xFu) | ((b2 >> 6u) << 4u));
            }
            float d1 = d * d1s;
            float m1 = mn * m1s;
            uint k0 = (bb * 8u + s) * 32u; // lhs offset of this 32-value group
            uint qbase = 4u + 8u * (s >> 1u); // uint index (in block) of the group's 32 q-bytes
            uint q0 = g_W[u + qbase];
            uint q1 = g_W[u + qbase + 1u];
            uint q2 = g_W[u + qbase + 2u];
            uint q3 = g_W[u + qbase + 3u];
            uint q4 = g_W[u + qbase + 4u];
            uint q5 = g_W[u + qbase + 5u];
            uint q6 = g_W[u + qbase + 6u];
            uint q7 = g_W[u + qbase + 7u];
            for (uint i = 0u; i < 32u; ++i)
            {
                uint j = i >> 2u;
                uint qq;
                if (j < 4u)
                {
                    qq = (j == 0u) ? q0 : (j == 1u) ? q1 : (j == 2u) ? q2 : q3;
                }
                else
                {
                    qq = (j == 4u) ? q4 : (j == 5u) ? q5 : (j == 6u) ? q6 : q7;
                }
                uint bit = 8u * (i & 3u) + 4u * (s & 1u);
                uint qval = (qq >> bit) & 0xFu;
                float val = d1 * float(qval) - m1;
                float l = g_Lhs[lhs_base + k0 + i];
                uint r = i & 3u;
                if (r == 0u)
                {
                    a0 += l * val;
                }
                else if (r == 1u)
                {
                    a1 += l * val;
                }
                else if (r == 2u)
                {
                    a2 += l * val;
                }
                else
                {
                    a3 += l * val;
                }
            }
        }
    }
    g_Out[route * n + col] = a0 + a1 + a2 + a3;
}
