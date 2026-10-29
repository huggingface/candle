// Fused Q6_K decode GEMM: out[n] = lhs[k] @ dequant(W[n, k]).
//
// Single-row lhs (decode): each thread owns one output column and
// dequantizes its Q6_K column in registers while accumulating the dot
// product, so no shared memory or barriers are needed and every thread
// does useful work. W holds raw Q6_K blocks (256 elements per 210 bytes:
// 128 bytes of 4-bit low quants at [0..128), 64 bytes of 2-bit high quants
// at [128..192), 16 i8 scales (one per 16 elements) at [192..208), and an
// f16 super-scale d at [208..210)) stored (n, k)-contiguous. m must be 1
// and k a multiple of 256; n need not be a multiple of 32 (out-of-range
// columns early-exit).
//
// Decode layout (from candle quantized/k_quants.rs BlockQ6K::to_float):
// within block half idx (0/1) and 32-wide group l, the four 6-bit quants
// for elements 128*idx + {l, l+32, l+64, l+96} come from low nibbles of
// ql[64*idx + {l, l+32}], high nibbles of the same bytes, and the 2-bit
// groups of qh[32*idx + l] (bits 0,2,4,6); element e uses scale
// scales[8*idx + (l/16) + 2*group] and value (q6 - 32).
//
// Params: [0]=m (must be 1), [1]=n, [2]=k.
import dequant_helpers;
import common_fused_gemm;

StructuredBuffer<float> g_Lhs; // (k,) f32
RWStructuredBuffer<float> g_Out; // (n,) f32
StructuredBuffer<uint8_t> g_W; // raw Q6_K bytes, (n, k)
layout(push_constant) Params g_Params;

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID : SV_GroupID)
{
    uint64_t n = g_Params.n;
    uint64_t k = g_Params.k;
    uint col = groupID.y * 32u + dispatchThreadID.x;
    if (col >= n)
    {
        return;
    }
    uint kblocks = k / 256u;
    float acc = 0.0;
    for (uint bb = 0u; bb < kblocks; ++bb)
    {
        uint off = (col * kblocks + bb) * 210u;
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
                uint sc = off + 192u + 8u * idx + is;
                float s1 = d * float(sbyte(g_W[sc]));
                float s2 = d * float(sbyte(g_W[sc + 2u]));
                float s3 = d * float(sbyte(g_W[sc + 4u]));
                float s4 = d * float(sbyte(g_W[sc + 6u]));
                uint k0 = bb * 256u + 128u * idx + l;
                acc += g_Lhs[k0] * (s1 * float(q1));
                acc += g_Lhs[k0 + 32u] * (s2 * float(q2));
                acc += g_Lhs[k0 + 64u] * (s3 * float(q3));
                acc += g_Lhs[k0 + 96u] * (s4 * float(q4));
            }
        }
    }
    g_Out[col] = acc;
}
