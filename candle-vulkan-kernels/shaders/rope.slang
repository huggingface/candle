// Rotary embedding (contiguous, non-interleaved, 2D cos/sin).
// src (b, h, t, d), cos (t, d/2), sin (t, d/2).
// For each (bh_i, i_t, i_d): i1 = bh_i*t*d + i_t*d + i_d, i2 = i1 + d/2,
// i_cs = i_t*(d/2) + i_d;
//   dst[i1] = src[i1]*cos[i_cs] - src[i2]*sin[i_cs]
//   dst[i2] = src[i1]*sin[i_cs] + src[i2]*cos[i_cs]
import common_rope;
StructuredBuffer<float> g_Src;
StructuredBuffer<float> g_Cos;
StructuredBuffer<float> g_Sin;
RWStructuredBuffer<float> g_Dst;
layout(push_constant) Params g_Params;
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    uint64_t bh = g_Params.bh;
    uint64_t t = g_Params.t;
    uint64_t d = g_Params.d;
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

// Batched rotary embedding: src (b, h, t, d), cos/sin (b, t, d/2).
// b_i = bh_i / h; i_cs = i_t*(d/2) + i_d + b_i*cs_stride.
[shader("compute")]
[numthreads(1, 1, 1)]
void main_batched(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    uint64_t bh = g_Params.bh;
    uint64_t h = max(g_Params.h, 1u);
    uint64_t t = g_Params.t;
    uint64_t d = g_Params.d;
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
    uint64_t b_i = bh_i / h;
    uint64_t i1 = bh_i * t * d + i_t * d + i_d;
    uint64_t i2 = i1 + half;
    uint64_t i_cs = i_t * half + i_d + b_i * g_Params.cs_stride;
    float s1 = g_Src[i1];
    float s2 = g_Src[i2];
    float c = g_Cos[i_cs];
    float s = g_Sin[i_cs];
    g_Dst[i1] = s1 * c - s2 * s;
    g_Dst[i2] = s1 * s + s2 * c;
}

// T/H/D-contiguous rotary embedding: src (b, t, h, d), cos/sin (t, d/2)
// or (b, t, d/2).
[shader("compute")]
[numthreads(1, 1, 1)]
void main_thd(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    uint64_t b = g_Params.bh;
    uint64_t h = max(g_Params.h, 1u);
    uint64_t t = g_Params.t;
    uint64_t d = g_Params.d;
    uint64_t half = d / 2u;
    uint64_t per_b = t * h * half;
    if (idx >= b * per_b || half == 0u)
    {
        return;
    }
    uint64_t b_i = idx / per_b;
    uint64_t rem = idx % per_b;
    uint64_t i_t = rem / (h * half);
    uint64_t rem2 = rem % (h * half);
    uint64_t i_h = rem2 / half;
    uint64_t i_d = rem2 % half;
    uint64_t i1 = b_i * t * h * d + i_t * h * d + i_h * d + i_d;
    uint64_t i2 = i1 + half;
    uint64_t i_cs = i_t * half + i_d + b_i * g_Params.cs_stride;
    float s1 = g_Src[i1];
    float s2 = g_Src[i2];
    float c = g_Cos[i_cs];
    float s = g_Sin[i_cs];
    g_Dst[i1] = s1 * c - s2 * s;
    g_Dst[i2] = s1 * s + s2 * c;
}
