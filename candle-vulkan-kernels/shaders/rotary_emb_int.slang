// Rotary embedding (interleaved):
// For each pair (i, i+1) in the head dimension:
//   dst[i]   = src[i]   * cos[rope_i] - src[i+1] * sin[rope_i]
//   dst[i+1] = src[i]   * sin[rope_i] + src[i+1] * cos[rope_i]
//
// Tensor layout: (b, h, t, d) where d is the head dimension.
// cos/sin layout: (t, d/2) or (b, t, d/2) depending on unbatched_rope.
//
// Params: [0]=b, [1]=h, [2]=t, [3]=d, [4]=unbatched (0 or 1)

StructuredBuffer<float> g_Src;
RWStructuredBuffer<float> g_Dst;
StructuredBuffer<float> g_Cos;
StructuredBuffer<float> g_Sin;
StructuredBuffer<uint64_t> g_Params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    uint64_t b = g_Params[0];
    uint64_t h = g_Params[1];
    uint64_t t = g_Params[2];
    uint64_t d = g_Params[3];
    uint64_t unbatched = g_Params[4];

    uint64_t el_count = b * h * t * d;
    if (i >= el_count)
    {
        return;
    }

    // Only process even-indexed elements (the first of each pair)
    if ((i & 1u) != 0u)
    {
        return;
    }

    // Compute the position within the (b, h, t, d) tensor
    uint64_t d_idx = i % d;
    uint64_t rem = i / d;
    uint64_t t_idx = rem % t;
    uint64_t rem2 = rem / t;
    uint64_t h_idx = rem2 % h;
    uint64_t b_idx = rem2 / h;

    // Compute the rope index
    uint64_t rope_i;
    if (unbatched != 0u)
    {
        rope_i = d_idx / 2u + b_idx * (d / 2u);
    }
    else
    {
        rope_i = d_idx / 2u;
    }

    float cos_val = g_Cos[rope_i];
    float sin_val = g_Sin[rope_i];
    float x0 = g_Src[i];
    float x1 = g_Src[i + 1];

    g_Dst[i] = x0 * cos_val - x1 * sin_val;
    g_Dst[i + 1] = x0 * sin_val + x1 * cos_val;
}
