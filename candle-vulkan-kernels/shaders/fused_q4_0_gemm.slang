// Fused Q4_0 GEMM: out[m, n] = lhs[m, k] @ dequant(W[n, k]).
//
// W holds raw Q4_0 blocks (32 elements per 18 bytes: 2-byte f16 scale,
// then 16 bytes holding 32 4-bit quants — low nibble element j, high
// nibble element 16+j) stored (n, k)-contiguous. Each workgroup tile
// dequantizes the Q4_0 column of its output element on the fly, so no f32
// weight materialization is needed. 16x16 workgroup tiles over 32-wide
// slabs of the reduction dimension (one Q4_0 block per slab); m, n and k
// need not be multiples of 16, k must be a multiple of 32 (out-of-range
// elements are zero-filled in the tiles).
//
// Barrier discipline: GroupMemoryBarrierWithGroupSync() is UB in divergent
// control flow (Vulkan spec 15.5.3), and RADV miscompiles it there. Every
// thread therefore executes both barriers on every slab iteration,
// regardless of whether its output element is in range; out-of-range
// threads still zero-fill their tile cells (so in-range readers see valid
// data) and skip only the dequantize/reduction and the output store.
//
// Params: [0]=m, [1]=n, [2]=k.
StructuredBuffer<float> g_Lhs; // (m, k) f32
RWStructuredBuffer<float> g_Out; // (m, n) f32
StructuredBuffer<uint8_t> g_W; // raw Q4_0 bytes, (n, k)
StructuredBuffer<uint64_t> g_Params;

static groupshared float a_tile[16 * 32];

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
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID : SV_GroupID, uint3 localID : SV_GroupThreadID)
{
    uint64_t m = g_Params[0];
    uint64_t n = g_Params[1];
    uint64_t k = g_Params[2];
    uint64_t kblocks = k / 32u;
    uint row = groupID.x * 16u + localID.y;
    uint col = groupID.y * 16u + localID.x;
    bool in_range = row < m && col < n;

    float acc = 0.0;
    for (uint b = 0u; b < kblocks; b++)
    {
        // Load the 16x32 lhs slab into shared memory: 2 values per thread.
        uint k0 = b * 32u;
        uint a_k = k0 + localID.x;
        if (a_k < k && row < m)
        {
            a_tile[localID.y * 32u + localID.x] = g_Lhs[row * k + a_k];
        }
        else
        {
            a_tile[localID.y * 32u + localID.x] = 0.0;
        }
        uint a_k2 = k0 + 16u + localID.x;
        if (a_k2 < k && row < m)
        {
            a_tile[localID.y * 32u + 16u + localID.x] = g_Lhs[row * k + a_k2];
        }
        else
        {
            a_tile[localID.y * 32u + 16u + localID.x] = 0.0;
        }
        GroupMemoryBarrierWithGroupSync();
        if (in_range)
        {
            // Dequantize this thread's Q4_0 column block (32 elements).
            uint off = (col * kblocks + b) * 18u;
            float d =
                f16_bits_to_f32(uint16_t(g_W[off]) | (uint16_t(g_W[off + 1u]) << 8u));
            for (uint j = 0u; j < 16u; ++j)
            {
                uint q = g_W[off + 2u + j];
                float x0 = float(int(q & 0x0Fu)) - 8.0;
                float x1 = float(int((q >> 4u) & 0x0Fu)) - 8.0;
                acc += a_tile[localID.y * 32u + j] * (x0 * d);
                acc += a_tile[localID.y * 32u + 16u + j] * (x1 * d);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (in_range)
    {
        g_Out[row * n + col] = acc;
    }
}
