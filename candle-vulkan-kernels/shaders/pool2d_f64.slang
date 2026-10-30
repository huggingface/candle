// 2D pooling over NCHW data (b*c treated as planes):
//   out[plane, ih, iw] = avg/max of src[plane, ih*s_h + m, iw*s_w + n]
// over m in 0..k_h, n in 0..k_w.
import common_pool2d;
StructuredBuffer<double> g_Src;
RWStructuredBuffer<double> g_Out;
layout(push_constant) Params g_Params;
// [0]=planes(b*c), [1]=h, [2]=w, [3]=k_h, [4]=k_w, [5]=s_h, [6]=s_w,
// [7]=h_out, [8]=w_out

[shader("compute")]
[numthreads(256, 1, 1)]
void main_avg_pool2d(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint o = dispatchThreadID.x;
    uint64_t planes = g_Params.planes;
    uint64_t h = g_Params.h;
    uint64_t w = g_Params.w;
    uint64_t k_h = g_Params.k_h;
    uint64_t k_w = g_Params.k_w;
    uint64_t s_h = g_Params.s_h;
    uint64_t s_w = g_Params.s_w;
    uint64_t h_out = g_Params.h_out;
    uint64_t w_out = g_Params.w_out;
    if (o >= planes * h_out * w_out)
    {
        return;
    }
    uint64_t hw = h_out * w_out;
    uint64_t plane = o / hw;
    uint64_t r = o - plane * hw;
    uint64_t ih = r / w_out;
    uint64_t iw = r - ih * w_out;
    uint64_t base = plane * w * h;
    double sum = 0.0;
    for (uint m = 0u; m < k_h; m += 1u)
    {
        uint64_t sr = ih * s_h + m;
        for (uint n = 0u; n < k_w; n += 1u)
        {
            sum += g_Src[base + sr * w + iw * s_w + n];
        }
    }
    g_Out[o] = sum / (double)(k_h * k_w);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_max_pool2d(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint o = dispatchThreadID.x;
    uint64_t planes = g_Params.planes;
    uint64_t h = g_Params.h;
    uint64_t w = g_Params.w;
    uint64_t k_h = g_Params.k_h;
    uint64_t k_w = g_Params.k_w;
    uint64_t s_h = g_Params.s_h;
    uint64_t s_w = g_Params.s_w;
    uint64_t h_out = g_Params.h_out;
    uint64_t w_out = g_Params.w_out;
    if (o >= planes * h_out * w_out)
    {
        return;
    }
    uint64_t hw = h_out * w_out;
    uint64_t plane = o / hw;
    uint64_t r = o - plane * hw;
    uint64_t ih = r / w_out;
    uint64_t iw = r - ih * w_out;
    uint64_t base = plane * h * w;
    double best = g_Src[base + ih * s_h * w + iw * s_w];
    for (uint m = 0u; m < k_h; m += 1u)
    {
        uint64_t sr = ih * s_h + m;
        for (uint n = 0u; n < k_w; n += 1u)
        {
            double v = g_Src[base + sr * w + iw * s_w + n];
            if (v > best)
            {
                best = v;
            }
        }
    }
    g_Out[o] = best;
}
