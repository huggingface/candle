// Upsampling over NCHW-style data (b*c treated as planes):
// main_upsample_nearest1d: (b, c, src_sz) -> (b, c, dst_sz)
// main_upsample_nearest2d: (b, c, src_h, src_w) -> (b, c, dst_h, dst_w)
// main_upsample_bilinear2d: (b, c, src_h, src_w) -> (b, c, dst_h, dst_w)
import common_upsample;

StructuredBuffer<half> g_Src;
RWStructuredBuffer<half> g_Out;
layout(push_constant) Params g_Params;
// nearest1d: planes, d0=src_sz, d1=dst_sz
// nearest2d: planes, d0=src_h, d1=src_w, d2=dst_h, d3=dst_w
// bilinear : planes, d0=src_h, d1=src_w, d2=dst_h, d3=dst_w,
//            scale_h, scale_w, align_corners

[shader("compute")]
[numthreads(256, 1, 1)]
void main_upsample_nearest1d(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint o = dispatchThreadID.x;
    uint64_t planes = g_Params.planes;
    uint64_t src_sz = g_Params.d0;
    uint64_t dst_sz = g_Params.d1;
    if (o >= planes * dst_sz || src_sz == 0u)
    {
        return;
    }
    uint64_t plane = o / dst_sz;
    uint64_t d = o - plane * dst_sz;
    uint64_t s = min(src_sz - 1u, (d * src_sz) / dst_sz);
    g_Out[o] = g_Src[plane * src_sz + s];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_upsample_nearest2d(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint o = dispatchThreadID.x;
    uint64_t planes = g_Params.planes;
    uint64_t src_h = g_Params.d0;
    uint64_t src_w = g_Params.d1;
    uint64_t dst_h = g_Params.d2;
    uint64_t dst_w = g_Params.d3;
    if (o >= planes * dst_h * dst_w || src_h == 0u || src_w == 0u)
    {
        return;
    }
    uint64_t hw = dst_h * dst_w;
    uint64_t plane = o / hw;
    uint64_t r = o - plane * hw;
    uint64_t dh = r / dst_w;
    uint64_t dw = r - dh * dst_w;
    uint64_t sh = min(src_h - 1u, (dh * src_h) / dst_h);
    uint64_t sw = min(src_w - 1u, (dw * src_w) / dst_w);
    g_Out[o] = g_Src[plane * src_h * src_w + sh * src_w + sw];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_upsample_bilinear2d(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint o = dispatchThreadID.x;
    uint64_t planes = g_Params.planes;
    uint64_t src_h = g_Params.d0;
    uint64_t src_w = g_Params.d1;
    uint64_t dst_h = g_Params.d2;
    uint64_t dst_w = g_Params.d3;
    float scale_h = float(g_Params.scale_h);
    float scale_w = float(g_Params.scale_w);
    float align = float(g_Params.align);
    if (o >= planes * dst_h * dst_w || src_h == 0u || src_w == 0u)
    {
        return;
    }
    uint64_t hw = dst_h * dst_w;
    uint64_t plane = o / hw;
    uint64_t r = o - plane * hw;
    uint64_t dh = r / dst_w;
    uint64_t dw = r - dh * dst_w;

    float src_hf = align != 0.0 ? scale_h * float(dh) : scale_h * (float(dh) + 0.5) - 0.5;
    float src_wf = align != 0.0 ? scale_w * float(dw) : scale_w * (float(dw) + 0.5) - 0.5;
    src_hf = max(src_hf, 0.0);
    src_wf = max(src_wf, 0.0);
    uint h0 = uint(floor(src_hf));
    uint w0 = uint(floor(src_wf));
    uint h1 = min(h0 + 1u, uint(src_h - 1u));
    uint w1 = min(w0 + 1u, uint(src_w - 1u));
    float wh = clamp(src_hf - float(h0), 0.0, 1.0);
    float wv = clamp(src_wf - float(w0), 0.0, 1.0);
    uint64_t base = plane * src_h * src_w;
    float v00 = (float)g_Src[base + h0 * src_w + w0];
    float v10 = (float)g_Src[base + h0 * src_w + w1];
    float v01 = (float)g_Src[base + h1 * src_w + w0];
    float v11 = (float)g_Src[base + h1 * src_w + w1];
    float v_top = v00 * (1.0 - wv) + v10 * wv;
    float v_bottom = v01 * (1.0 - wv) + v11 * wv;
    g_Out[o] = (half)(v_top * (1.0 - wh) + v_bottom * wh);
}
