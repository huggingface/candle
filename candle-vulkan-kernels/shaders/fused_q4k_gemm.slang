// Fused Q4_K GEMM: out[m, n] = lhs[m, k] @ dequant(W[n, k]).
//
// W holds raw Q4_K blocks (256 elements per 144 bytes: 2 f16 super scale/min
// at bytes [0..4), 12 bytes of 6-bit scale/min pairs at [4..16), 128 bytes of
// 4-bit quants at [16..144)) stored (n, k)-contiguous. Each workgroup tile
// dequantizes the Q4_K column of its output element on the fly, so no f32
// weight materialization is needed. 16x16 workgroup tiles over 32-wide slabs
// of the reduction dimension; m, n need not be multiples of 16, k must be a
// multiple of 256 (out-of-range elements are zero-filled in the tiles).
//
// The 32-wide slab `b % 8` (within-block slab 0..7) of a Q4_K block holds
// 32 quants at bytes [16 + 32*((b%8)/2) .. 16 + 32*((b%8)/2) + 32): low
// nibbles for even b, high nibbles for odd b, 6-bit scale/min pair j = b%8.
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
StructuredBuffer<uint8_t> g_W; // raw Q4_K bytes, (n, k)
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
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID : SV_GroupID, uint3 localID : SV_GroupThreadID)
{
    uint64_t m = g_Params[0];
    uint64_t n = g_Params[1];
    uint64_t k = g_Params[2];
    uint64_t kslabs = k / 32u;
    uint row = groupID.x * 16u + localID.y;
    uint col = groupID.y * 16u + localID.x;
    bool in_range = row < m && col < n;

    float acc = 0.0;
    for (uint b = 0u; b < kslabs; b++)
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
            // Dequantize this thread's Q4_K slab (32 elements).
            uint block = col * (k / 256u) + b / 8u;
            uint off = block * 144u;
            float d =
                f16_bits_to_f32(uint16_t(g_W[off]) | (uint16_t(g_W[off + 1u]) << 8u));
            float mn =
                f16_bits_to_f32(uint16_t(g_W[off + 2u]) | (uint16_t(g_W[off + 3u]) << 8u));
            uint2 sm = get_scale_min_k4(b % 8u, off);
            float d1 = d * float(sm.x);
            float m1 = mn * float(sm.y);
            uint qbyte0 = off + 16u + 32u * ((b % 8u) / 2u);
            if ((b & 1u) == 0u)
            {
                for (uint i = 0u; i < 32u; ++i)
                {
                    float w = d1 * float(g_W[qbyte0 + i] & 0xFu) - m1;
                    acc += a_tile[localID.y * 32u + i] * w;
                }
            }
            else
            {
                for (uint i = 0u; i < 32u; ++i)
                {
                    float w = d1 * float((g_W[qbyte0 + i] >> 4) & 0xFu) - m1;
                    acc += a_tile[localID.y * 32u + i] * w;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (in_range)
    {
        g_Out[row * n + col] = acc;
    }
}
