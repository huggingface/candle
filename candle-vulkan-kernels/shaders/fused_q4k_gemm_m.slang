// Fused Q4_K prefill GEMM: out[row, col] = lhs[row, k] @ dequant(W[col, k]).
//
// Multi-row lhs (prefill): one thread per output element. The dispatch is
// (1, ceil(n/32), m) workgroups of 32 threads: groupID.z selects the lhs row,
// groupID.y * 32 + dispatchThreadID.x the weight column. Each thread
// dequantizes its column in registers (no shared memory or barriers). W holds
// raw Q4_K blocks (256 elements per 144 bytes: 2 f16 super scale/min, 12 bytes of 6-bit scale/min pairs, 128 bytes of 4-bit quants) stored (n, k)-contiguous. k must be a multiple
// of 256; n need not be a multiple of 32 (out-of-range columns early-exit).
//
// Params: [0]=m, [1]=n, [2]=k.
import dequant_helpers;

StructuredBuffer<float> g_Lhs; // (m, k) f32
RWStructuredBuffer<float> g_Out; // (m, n) f32
StructuredBuffer<uint8_t> g_W; // raw Q4_K bytes, (n, k)
StructuredBuffer<uint64_t> g_Params;

// Q4K/Q5K scale-min decoder (from candle quantized/utils.rs::get_scale_min_k4).
// `off` is the block base offset in g_W; `soff` is the scale-array offset.
uint2 get_scale_min_k4(uint j, uint off, uint soff)
{
    if (j < 4u)
    {
        uint d = g_W[off + soff + j] & 63u;
        uint m = g_W[off + soff + j + 4u] & 63u;
        return uint2(d, m);
    }
    else
    {
        uint d = (g_W[off + soff + j + 4u] & 0xFu) | ((g_W[off + soff + j - 4u] >> 6) << 4);
        uint m = (g_W[off + soff + j + 4u] >> 4) | ((g_W[off + soff + j] >> 6) << 4);
        return uint2(d, m);
    }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID : SV_GroupID)
{
    uint64_t n = g_Params[1];
    uint64_t k = g_Params[2];
    uint col = groupID.y * 32u + dispatchThreadID.x;
    if (col >= n)
    {
        return;
    }
    uint row = groupID.z;
    uint row_base = uint(row) * uint(k);
    uint kblocks = k / 256u;
    float acc = 0.0;
    for (uint bb = 0u; bb < kblocks; ++bb)
    {
        uint off = (col * kblocks + bb) * 144u;
        float d =
            f16_bits_to_f32(uint16_t(g_W[off]) | (uint16_t(g_W[off + 1u]) << 8u));
        float mn =
            f16_bits_to_f32(uint16_t(g_W[off + 2u]) | (uint16_t(g_W[off + 3u]) << 8u));
        for (uint s = 0u; s < 8u; ++s)
        {
            uint2 sm = get_scale_min_k4(s, off, 4u);
            float d1 = d * float(sm.x);
            float m1 = mn * float(sm.y);
            uint qbyte0 = off + 16u + 32u * (s / 2u);
            uint k0 = (bb * 8u + s) * 32u;
            if ((s & 1u) == 0u)
            {
                for (uint i = 0u; i < 32u; ++i)
                {
                    acc += g_Lhs[row_base + k0 + i] * (d1 * float(g_W[qbyte0 + i] & 0xFu) - m1);
                }
            }
            else
            {
                for (uint i = 0u; i < 32u; ++i)
                {
                    acc += g_Lhs[row_base + k0 + i] * (d1 * float((g_W[qbyte0 + i] >> 4u) & 0xFu) - m1);
                }
            }
        }
    }
    g_Out[uint(row) * uint(n) + col] = acc;
}
