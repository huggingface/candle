// Fused Q4_K decode GEMM: out[n] = lhs[k] @ dequant(W[n, k]).
//
// Single-row lhs (decode): each thread owns one output column and
// dequantizes its Q4_K column in registers while accumulating the dot
// product, so no shared memory or barriers are needed and every thread
// does useful work. W holds raw Q4_K blocks (256 elements per 144 bytes:
// 2 f16 super scale/min at bytes [0..4), 12 bytes of 6-bit scale/min
// pairs at [4..16), 128 bytes of 4-bit quants at [16..144)) stored
// (n, k)-contiguous. m must be 1 and k a multiple of 256; n need not be
// a multiple of 32 (out-of-range columns early-exit).
//
// The 32-wide slab s (within-block slab 0..7) of a Q4_K block holds 32
// quants at bytes [16 + 32*(s/2) .. 16 + 32*(s/2) + 32): low nibbles for
// even s, high nibbles for odd s, 6-bit scale/min pair j = s. The super
// scale/min is decoded once per block and reused by all 8 slabs.
//
// Params: [0]=m (must be 1), [1]=n, [2]=k.
import dequant_helpers;

StructuredBuffer<float> g_Lhs; // (k,) f32
RWStructuredBuffer<float> g_Out; // (n,) f32
StructuredBuffer<uint8_t> g_W; // raw Q4_K bytes, (n, k)
StructuredBuffer<uint64_t> g_Params;

// 6-bit scale/min pair index j (0..7) of the Q4_K block at `off`
// (from candle quantized/utils.rs::get_scale_min_k4, soff=4).
uint2 get_scale_min_k4(uint j, uint off)
{
    if (j < 4u)
    {
        uint d = g_W[off + 4u + j] & 63u;
        uint m = g_W[off + 8u + j] & 63u;
        return uint2(d, m);
    }
    else
    {
        uint d = (g_W[off + 12u + (j - 4u)] & 0xFu) | ((g_W[off + 4u + (j - 4u)] >> 6) << 4);
        uint m = (g_W[off + 12u + (j - 4u)] >> 4) | ((g_W[off + 8u + (j - 4u)] >> 6) << 4);
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
            uint2 sm = get_scale_min_k4(s, off);
            float d1 = d * float(sm.x);
            float m1 = mn * float(sm.y);
            uint qbyte0 = off + 16u + 32u * (s / 2u);
            uint k0 = (bb * 8u + s) * 32u;
            if ((s & 1u) == 0u)
            {
                for (uint i = 0u; i < 32u; ++i)
                {
                    acc += g_Lhs[k0 + i] * (d1 * float(g_W[qbyte0 + i] & 0xFu) - m1);
                }
            }
            else
            {
                for (uint i = 0u; i < 32u; ++i)
                {
                    acc += g_Lhs[k0 + i] * (d1 * float((g_W[qbyte0 + i] >> 4) & 0xFu) - m1);
                }
            }
        }
    }
    g_Out[col] = acc;
}
