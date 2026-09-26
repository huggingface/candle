// 2D pooling over NCHW data (b*c treated as planes):
//   out[plane, ih, iw] = avg/max of src[plane, ih*s_h + m, iw*s_w + n]
// over m in 0..k_h, n in 0..k_w.
StructuredBuffer<half> g_Src;
RWStructuredBuffer<half> g_Out;
StructuredBuffer<float> g_Params;
// [0]=planes(b*c), [1]=h, [2]=w, [3]=k_h, [4]=k_w, [5]=s_h, [6]=s_w,
// [7]=h_out, [8]=w_out

[shader("compute")]
[numthreads(256, 1, 1)]
void main_avg_pool2d(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint o = dispatchThreadID.x;
    uint planes = uint(g_Params[0]);
    uint h = uint(g_Params[1]);
    uint w = uint(g_Params[2]);
    uint k_h = uint(g_Params[3]);
    uint k_w = uint(g_Params[4]);
    uint s_h = uint(g_Params[5]);
    uint s_w = uint(g_Params[6]);
    uint h_out = uint(g_Params[7]);
    uint w_out = uint(g_Params[8]);
    if (o >= planes * h_out * w_out)
    {
        return;
    }
    uint hw = h_out * w_out;
    uint plane = o / hw;
    uint r = o - plane * hw;
    uint ih = r / w_out;
    uint iw = r - ih * w_out;
    uint base = plane * w * h;
    float sum = 0.0;
    for (uint m = 0u; m < k_h; m += 1u)
    {
        uint sr = ih * s_h + m;
        for (uint n = 0u; n < k_w; n += 1u)
        {
            sum += (float)g_Src[base + sr * w + iw * s_w + n];
        }
    }
    g_Out[o] = (half)(sum / (float(k_h * k_w)));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_max_pool2d(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint o = dispatchThreadID.x;
    uint planes = uint(g_Params[0]);
    uint h = uint(g_Params[1]);
    uint w = uint(g_Params[2]);
    uint k_h = uint(g_Params[3]);
    uint k_w = uint(g_Params[4]);
    uint s_h = uint(g_Params[5]);
    uint s_w = uint(g_Params[6]);
    uint h_out = uint(g_Params[7]);
    uint w_out = uint(g_Params[8]);
    if (o >= planes * h_out * w_out)
    {
        return;
    }
    uint hw = h_out * w_out;
    uint plane = o / hw;
    uint r = o - plane * hw;
    uint ih = r / w_out;
    uint iw = r - ih * w_out;
    uint base = plane * h * w;
    float best = (float)g_Src[base + ih * s_h * w + iw * s_w];
    for (uint m = 1u; m < k_h; m += 1u)
    {
        uint sr = ih * s_h + m;
        for (uint n = 0u; n < k_w; n += 1u)
        {
            float v = (float)g_Src[base + sr * w + iw * s_w + n];
            if (v > best)
            {
                best = v;
            }
        }
    }
    g_Out[o] = (half)best;
}
