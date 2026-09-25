// Convolution kernels, F32, NCHW/NCL data, single-group semantics
// (grouped convs are chunked into single-group calls by the Tensor API).
//
// main_conv1d: in (b, c_in, l_in), w (c_out, c_in, k), out (b, c_out, l_out)
//   params: [0]=b, [1]=c_in, [2]=c_out, [3]=k_size, [4]=l_in, [5]=l_out,
//           [6]=padding, [7]=stride, [8]=dilation
// main_conv2d: in (b, c_in, i_h, i_w), w (c_out, c_in, k_h, k_w),
//   out (b, c_out, o_h, o_w)
//   params: [0]=b, [1]=c_in, [2]=c_out, [3]=k_h, [4]=k_w, [5]=i_h, [6]=i_w,
//           [7]=o_h, [8]=o_w, [9]=padding, [10]=stride, [11]=dilation
// main_conv_transpose1d: in (b, c_in, l_in), w (c_in, c_out, k), out (b, c_out, l_out)
//   params: [0]=b, [1]=c_in, [2]=c_out, [3]=k_size, [4]=l_in, [5]=l_out,
//           [6]=padding, [7]=stride, [8]=dilation
// main_conv_transpose2d: in (b, c_in, i_h, i_w), w (c_in, c_out, k_h, k_w),
//   out (b, c_out, o_h, o_w)
//   params: [0]=b, [1]=c_in, [2]=c_out, [3]=k_h, [4]=k_w, [5]=i_h, [6]=i_w,
//           [7]=o_h, [8]=o_w, [9]=padding, [10]=stride, [11]=dilation
StructuredBuffer<float> g_In;
StructuredBuffer<float> g_W;
RWStructuredBuffer<float> g_Out;
StructuredBuffer<float> g_Params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main_conv1d(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint o = dispatchThreadID.x;
    uint b = uint(g_Params[0]);
    uint c_in = uint(g_Params[1]);
    uint c_out = uint(g_Params[2]);
    uint k_size = uint(g_Params[3]);
    uint l_in = uint(g_Params[4]);
    uint l_out = uint(g_Params[5]);
    int padding = int(g_Params[6]);
    int stride = int(g_Params[7]);
    int dilation = int(g_Params[8]);
    if (o >= b * c_out * l_out)
    {
        return;
    }
    uint plane = o / l_out;
    uint lo = o - plane * l_out;
    uint batch = plane / c_out;
    uint co = plane - batch * c_out;
    uint in_base = batch * c_in * l_in;
    uint w_base = co * c_in * k_size;
    float acc = 0.0;
    for (uint ci = 0u; ci < c_in; ci++)
    {
        for (uint k = 0u; k < k_size; k++)
        {
            int pos = int(lo) * stride - padding + int(k) * dilation;
            if (pos >= 0 && uint(pos) < l_in)
            {
                acc += g_In[in_base + ci * l_in + uint(pos)] * g_W[w_base + ci * k_size + k];
            }
        }
    }
    g_Out[o] = acc;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_conv2d(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint o = dispatchThreadID.x;
    uint b = uint(g_Params[0]);
    uint c_in = uint(g_Params[1]);
    uint c_out = uint(g_Params[2]);
    uint k_h = uint(g_Params[3]);
    uint k_w = uint(g_Params[4]);
    uint i_h = uint(g_Params[5]);
    uint i_w = uint(g_Params[6]);
    uint o_h = uint(g_Params[7]);
    uint o_w = uint(g_Params[8]);
    int padding = int(g_Params[9]);
    int stride = int(g_Params[10]);
    int dilation = int(g_Params[11]);
    if (o >= b * c_out * o_h * o_w)
    {
        return;
    }
    uint hw = o_h * o_w;
    uint plane = o / hw;
    uint r = o - plane * hw;
    uint oh = r / o_w;
    uint ow = r - oh * o_w;
    uint batch = plane / c_out;
    uint co = plane - batch * c_out;
    uint in_base = batch * c_in * i_h * i_w;
    uint w_base = co * c_in * k_h * k_w;
    float acc = 0.0;
    for (uint ci = 0u; ci < c_in; ci++)
    {
        uint ci_base = ci * i_h * i_w;
        uint ci_w_base = ci * k_h * k_w;
        for (uint kh = 0u; kh < k_h; kh++)
        {
            int ph = int(oh) * stride - padding + int(kh) * dilation;
            if (ph < 0 || uint(ph) >= i_h)
            {
                continue;
            }
            uint ih = uint(ph);
            for (uint kw = 0u; kw < k_w; kw++)
            {
                int pw = int(ow) * stride - padding + int(kw) * dilation;
                if (pw < 0 || uint(pw) >= i_w)
                {
                    continue;
                }
                acc += g_In[in_base + ci_base + ih * i_w + uint(pw)] *
                    g_W[w_base + ci_w_base + kh * k_w + kw];
            }
        }
    }
    g_Out[o] = acc;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_conv_transpose1d(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint o = dispatchThreadID.x;
    uint b = uint(g_Params[0]);
    uint c_in = uint(g_Params[1]);
    uint c_out = uint(g_Params[2]);
    uint k_size = uint(g_Params[3]);
    uint l_in = uint(g_Params[4]);
    uint l_out = uint(g_Params[5]);
    int padding = int(g_Params[6]);
    int stride = int(g_Params[7]);
    int dilation = int(g_Params[8]);
    if (o >= b * c_out * l_out)
    {
        return;
    }
    uint plane = o / l_out;
    uint po = o - plane * l_out;
    uint batch = plane / c_out;
    uint co = plane - batch * c_out;
    uint in_base = batch * c_in * l_in;
    uint w_base = co * k_size;
    float acc = 0.0;
    for (uint ci = 0u; ci < c_in; ci++)
    {
        for (uint k = 0u; k < k_size; k++)
        {
            int num = int(po) + padding - int(k) * dilation;
            if (num < 0 || num % stride != 0)
            {
                continue;
            }
            uint li = uint(num / stride);
            if (li >= l_in)
            {
                continue;
            }
            acc += g_In[in_base + ci * l_in + li] * g_W[(ci * c_out + co) * k_size + k];
        }
    }
    g_Out[o] = acc;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_conv_transpose2d(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint o = dispatchThreadID.x;
    uint b = uint(g_Params[0]);
    uint c_in = uint(g_Params[1]);
    uint c_out = uint(g_Params[2]);
    uint k_h = uint(g_Params[3]);
    uint k_w = uint(g_Params[4]);
    uint i_h = uint(g_Params[5]);
    uint i_w = uint(g_Params[6]);
    uint o_h = uint(g_Params[7]);
    uint o_w = uint(g_Params[8]);
    int padding = int(g_Params[9]);
    int stride = int(g_Params[10]);
    int dilation = int(g_Params[11]);
    if (o >= b * c_out * o_h * o_w)
    {
        return;
    }
    uint hw = o_h * o_w;
    uint plane = o / hw;
    uint r = o - plane * hw;
    uint oh = r / o_w;
    uint ow = r - oh * o_w;
    uint batch = plane / c_out;
    uint co = plane - batch * c_out;
    uint in_base = batch * c_in * i_h * i_w;
    float acc = 0.0;
    for (uint ci = 0u; ci < c_in; ci++)
    {
        uint ci_base = ci * i_h * i_w;
        uint ci_w_base = ci * c_out * k_h * k_w;
        for (uint kh = 0u; kh < k_h; kh++)
        {
            int num_h = int(oh) + padding - int(kh) * dilation;
            if (num_h < 0 || num_h % stride != 0)
            {
                continue;
            }
            uint ih = uint(num_h / stride);
            if (ih >= i_h)
            {
                continue;
            }
            for (uint kw = 0u; kw < k_w; kw++)
            {
                int num_w = int(ow) + padding - int(kw) * dilation;
                if (num_w < 0 || num_w % stride != 0)
                {
                    continue;
                }
                uint iw = uint(num_w / stride);
                if (iw >= i_w)
                {
                    continue;
                }
                acc += g_In[in_base + ci_base + ih * i_w + iw] *
                    g_W[(ci_w_base + co * k_h * k_w) + kh * k_w + kw];
            }
        }
    }
    g_Out[o] = acc;
}
