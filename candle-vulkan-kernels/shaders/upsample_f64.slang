// Upsampling over NCHW-style data (b*c treated as planes):
// main_upsample_nearest1d: (b, c, src_sz) -> (b, c, dst_sz)
// main_upsample_nearest2d: (b, c, src_h, src_w) -> (b, c, dst_h, dst_w)
// main_upsample_bilinear2d: (b, c, src_h, src_w) -> (b, c, dst_h, dst_w)
StructuredBuffer<double> g_Src;
RWStructuredBuffer<double> g_Out;
StructuredBuffer<float> g_Params;
// nearest1d: [0]=planes, [1]=src_sz, [2]=dst_sz
// nearest2d: [0]=planes, [1]=src_h, [2]=src_w, [3]=dst_h, [4]=dst_w
// bilinear : [0]=planes, [1]=src_h, [2]=src_w, [3]=dst_h, [4]=dst_w,
//            [5]=scale_h, [6]=scale_w, [7]=align_corners

[shader("compute")]
[numthreads(256, 1, 1)]
void main_upsample_nearest1d(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint o = dispatchThreadID.x;
    uint planes = uint(g_Params[0]);
    uint src_sz = uint(g_Params[1]);
    uint dst_sz = uint(g_Params[2]);
    if (o >= planes * dst_sz || src_sz == 0u)
    {
        return;
    }
    uint plane = o / dst_sz;
    uint d = o - plane * dst_sz;
    uint s = min(src_sz - 1u, (d * src_sz) / dst_sz);
    g_Out[o] = g_Src[plane * src_sz + s];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_upsample_nearest2d(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint o = dispatchThreadID.x;
    uint planes = uint(g_Params[0]);
    uint src_h = uint(g_Params[1]);
    uint src_w = uint(g_Params[2]);
    uint dst_h = uint(g_Params[3]);
    uint dst_w = uint(g_Params[4]);
    if (o >= planes * dst_h * dst_w || src_h == 0u || src_w == 0u)
    {
        return;
    }
    uint hw = dst_h * dst_w;
    uint plane = o / hw;
    uint r = o - plane * hw;
    uint dh = r / dst_w;
    uint dw = r - dh * dst_w;
    uint sh = min(src_h - 1u, (dh * src_h) / dst_h);
    uint sw = min(src_w - 1u, (dw * src_w) / dst_w);
    g_Out[o] = g_Src[plane * src_h * src_w + sh * src_w + sw];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_upsample_bilinear2d(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint o = dispatchThreadID.x;
    uint planes = uint(g_Params[0]);
    uint src_h = uint(g_Params[1]);
    uint src_w = uint(g_Params[2]);
    uint dst_h = uint(g_Params[3]);
    uint dst_w = uint(g_Params[4]);
    float scale_h = g_Params[5];
    float scale_w = g_Params[6];
    float align = g_Params[7];
    if (o >= planes * dst_h * dst_w || src_h == 0u || src_w == 0u)
    {
        return;
    }
    uint hw = dst_h * dst_w;
    uint plane = o / hw;
    uint r = o - plane * hw;
    uint dh = r / dst_w;
    uint dw = r - dh * dst_w;

    float src_hf = align != 0.0 ? scale_h * float(dh) : scale_h * (float(dh) + 0.5) - 0.5;
    float src_wf = align != 0.0 ? scale_w * float(dw) : scale_w * (float(dw) + 0.5) - 0.5;
    src_hf = max(src_hf, 0.0);
    src_wf = max(src_wf, 0.0);
    uint h0 = uint(floor(src_hf));
    uint w0 = uint(floor(src_wf));
    uint h1 = min(h0 + 1u, src_h - 1u);
    uint w1 = min(w0 + 1u, src_w - 1u);
    float wh = clamp(src_hf - float(h0), 0.0, 1.0);
    float wv = clamp(src_wf - float(w0), 0.0, 1.0);
    uint base = plane * src_h * src_w;
    double v00 = g_Src[base + h0 * src_w + w0];
    double v10 = g_Src[base + h0 * src_w + w1];
    double v01 = g_Src[base + h1 * src_w + w0];
    double v11 = g_Src[base + h1 * src_w + w1];
    double v_top = v00 * (double)(1.0 - wv) + v10 * (double)wv;
    double v_bottom = v01 * (double)(1.0 - wv) + v11 * (double)wv;
    g_Out[o] = v_top * (double)(1.0 - wh) + v_bottom * (double)wh;
}
