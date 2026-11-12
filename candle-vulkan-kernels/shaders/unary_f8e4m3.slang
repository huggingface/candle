// Unary f8e4m3 ops (emulated). Each entry point maps g_In[i] -> g_Out[i] for i < g_Params[0],
// decoding to float, applying the op in float, and encoding back to f8e4m3.
import common_unary;
import f8e4m3_type;

StructuredBuffer<FP8> g_In;
RWStructuredBuffer<FP8> g_Out;
layout(push_constant) Params g_Params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main_log(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = f8e4m3_decode(g_In[idx]);
        g_Out[idx] = f8e4m3_encode(log(x));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_abs(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = f8e4m3_decode(g_In[idx]);
        g_Out[idx] = f8e4m3_encode(abs(x));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_recip(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = f8e4m3_decode(g_In[idx]);
        g_Out[idx] = f8e4m3_encode(1.0 / x);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_sqr(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = f8e4m3_decode(g_In[idx]);
        g_Out[idx] = f8e4m3_encode(x * x);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_gelu(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = f8e4m3_decode(g_In[idx]);
        g_Out[idx] = f8e4m3_encode(0.5 * x * (1.0 + tanh(0.79788456080286535587989211986876373 * x * (1.0 + 0.044715 * x * x))));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_gelu_erf(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = f8e4m3_decode(g_In[idx]);
        g_Out[idx] = f8e4m3_encode((erf_approx(x * 0.7071067811865476) + 1.0) * 0.5 * x);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_erf(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = f8e4m3_decode(g_In[idx]);
        g_Out[idx] = f8e4m3_encode(erf_approx(x));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_relu(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = f8e4m3_decode(g_In[idx]);
        g_Out[idx] = f8e4m3_encode(max(x, 0.0));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_tanh(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = f8e4m3_decode(g_In[idx]);
        g_Out[idx] = f8e4m3_encode(tanh(x));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_floor(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = f8e4m3_decode(g_In[idx]);
        g_Out[idx] = f8e4m3_encode(floor(x));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_ceil(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = f8e4m3_decode(g_In[idx]);
        g_Out[idx] = f8e4m3_encode(ceil(x));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_round(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        // candle uses f8e4m3::round_ties_even (round-half-to-even); Slang round()
        // is half-to-even, so it matches directly.
        float x = f8e4m3_decode(g_In[idx]);
        g_Out[idx] = f8e4m3_encode(round(x));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_sign(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = f8e4m3_decode(g_In[idx]);
        g_Out[idx] = f8e4m3_encode((x > 0.0 ? 1.0 : 0.0) - (x < 0.0 ? 1.0 : 0.0));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_exp(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = f8e4m3_decode(g_In[idx]);
        g_Out[idx] = f8e4m3_encode(exp(x));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_silu(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = f8e4m3_decode(g_In[idx]);
        g_Out[idx] = f8e4m3_encode(x / (1.0 + exp(-x)));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_sqrt(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = f8e4m3_decode(g_In[idx]);
        g_Out[idx] = f8e4m3_encode(sqrt(x));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_sin(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = f8e4m3_decode(g_In[idx]);
        g_Out[idx] = f8e4m3_encode(sin(x));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_cos(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = f8e4m3_decode(g_In[idx]);
        g_Out[idx] = f8e4m3_encode(cos(x));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_neg(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = f8e4m3_decode(g_In[idx]);
        g_Out[idx] = f8e4m3_encode(-x);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_pow(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    if (i < uint(g_Params.count))
    {
        float x = f8e4m3_decode(g_In[i]);
        g_Out[i] = f8e4m3_encode(pow(x, float(g_Params.extra)));
    }
}
[shader("compute")]
[numthreads(256, 1, 1)]
void main_elu(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    if (i < uint(g_Params.count))
    {
        float x = f8e4m3_decode(g_In[i]);
        float alpha = (float)g_Params.extra;
        g_Out[i] = f8e4m3_encode(x >= 0.0 ? x : alpha * (exp(x) - 1.0));
    }
}
