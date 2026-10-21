// Fused Q4_0 prefill GEMM: out[row, col] = lhs[row, k] @ dequant(W[col, k]).
//
// Multi-row lhs (prefill): one thread per output element. The dispatch is
// (1, ceil(n/32), m) workgroups of 32 threads: groupID.z selects the lhs row,
// groupID.y * 32 + dispatchThreadID.x the weight column. Each thread
// dequantizes its column in registers (no shared memory or barriers). W holds
// raw Q4_0 blocks (32 elements per 18 bytes: 2-byte f16 scale, then 16 bytes of 4-bit quants) stored (n, k)-contiguous. k must be a multiple
// of 32; n need not be a multiple of 32 (out-of-range columns early-exit).
//
// Params: [0]=m, [1]=n, [2]=k.
StructuredBuffer<float> g_Lhs; // (m, k) f32
RWStructuredBuffer<float> g_Out; // (m, n) f32
StructuredBuffer<uint8_t> g_W; // raw Q4_0 bytes, (n, k)
StructuredBuffer<uint64_t> g_Params;

// Reinterpret an IEEE-754 half (16 bits) as a float (half crate).
float f16_bits_to_f32(uint16_t h)
{
    uint sign = (h >> 15) & 1u;
    uint exp = (h >> 10) & 0x1Fu;
    uint mant = h & 0x3FFu;
    uint32_t f;
    if (exp == 0u)
    {
        if (mant == 0u)
        {
            f = sign << 31; // signed zero
        }
        else
        {
            uint m = mant;
            uint shift = 0u;
            while ((m & 0x400u) == 0u)
            {
                m <<= 1;
                shift += 1;
            }
            m &= 0x3FFu;
            f = (sign << 31) | ((113u - shift) << 23) | (m << 13);
        }
    }
    else if (exp == 31u)
    {
        f = (sign << 31) | 0x7F800000u | (mant << 13);
    }
    else
    {
        f = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
    }
    return asfloat(f);
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
    uint kblocks = k / 32u;
    float acc = 0.0;
    for (uint bb = 0u; bb < kblocks; ++bb)
    {
        uint off = (col * kblocks + bb) * 18u;
        float d =
            f16_bits_to_f32(uint16_t(g_W[off]) | (uint16_t(g_W[off + 1u]) << 8u));
        for (uint j = 0u; j < 16u; ++j)
        {
            uint q = g_W[off + 2u + j];
            float x0 = float(int(q & 0x0Fu)) - 8.0;
            float x1 = float(int((q >> 4u) & 0x0Fu)) - 8.0;
            acc += g_Lhs[row_base + bb * 32u + j] * (x0 * d);
            acc += g_Lhs[row_base + bb * 32u + 16u + j] * (x1 * d);
        }
    }
    g_Out[uint(row) * uint(n) + col] = acc;
}
