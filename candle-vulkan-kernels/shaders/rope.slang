// Rotary embedding (contiguous, non-interleaved, 2D cos/sin).
// src (b, h, t, d), cos (t, d/2), sin (t, d/2).
// For each (bh_i, i_t, i_d): i1 = bh_i*t*d + i_t*d + i_d, i2 = i1 + d/2,
// i_cs = i_t*(d/2) + i_d;
//   dst[i1] = src[i1]*cos[i_cs] - src[i2]*sin[i_cs]
//   dst[i2] = src[i1]*sin[i_cs] + src[i2]*cos[i_cs]
StructuredBuffer<float> g_Src;
StructuredBuffer<float> g_Cos;
StructuredBuffer<float> g_Sin;
RWStructuredBuffer<float> g_Dst;
StructuredBuffer<uint64_t> g_Params; // [0]=b*h, [1]=t, [2]=d
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    uint64_t bh = g_Params[0];
    uint64_t t = g_Params[1];
    uint64_t d = g_Params[2];
    uint64_t half = d / 2u;
    uint64_t per_bh = t * half;
    if (idx >= bh * per_bh || half == 0u)
    {
        return;
    }
    uint64_t bh_i = idx / per_bh;
    uint64_t rem = idx % per_bh;
    uint64_t i_t = rem / half;
    uint64_t i_d = rem % half;
    uint64_t i1 = bh_i * t * d + i_t * d + i_d;
    uint64_t i2 = i1 + half;
    uint64_t i_cs = i_t * half + i_d;
    float s1 = g_Src[i1];
    float s2 = g_Src[i2];
    float c = g_Cos[i_cs];
    float s = g_Sin[i_cs];
    g_Dst[i1] = s1 * c - s2 * s;
    g_Dst[i2] = s1 * s + s2 * c;
}
