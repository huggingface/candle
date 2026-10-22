// Unary bf16 ops (emulated). Each entry point maps g_In[i] -> g_Out[i] for i < g_Params[0],
// decoding to float, applying the op in float, and encoding back to bf16.
import bf16_emulation;
import common_unary;

StructuredBuffer<uint16_t> g_In;
RWStructuredBuffer<uint16_t> g_Out;
layout(push_constant) Params g_Params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main_log(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = bf16_decode(g_In[idx]);
        g_Out[idx] = bf16_encode(log(x));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_abs(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = bf16_decode(g_In[idx]);
        g_Out[idx] = bf16_encode(abs(x));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_recip(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = bf16_decode(g_In[idx]);
        g_Out[idx] = bf16_encode(1.0 / x);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_sqr(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = bf16_decode(g_In[idx]);
        g_Out[idx] = bf16_encode(x * x);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_gelu(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = bf16_decode(g_In[idx]);
        g_Out[idx] = bf16_encode(0.5 * x * (1.0 + tanh(0.79788456080286535587989211986876373 * x * (1.0 + 0.044715 * x * x))));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_gelu_erf(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = bf16_decode(g_In[idx]);
        g_Out[idx] = bf16_encode((erf_approx(x * 0.7071067811865476) + 1.0) * 0.5 * x);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_erf(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = bf16_decode(g_In[idx]);
        g_Out[idx] = bf16_encode(erf_approx(x));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_relu(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = bf16_decode(g_In[idx]);
        g_Out[idx] = bf16_encode(max(x, 0.0));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_tanh(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = bf16_decode(g_In[idx]);
        g_Out[idx] = bf16_encode(tanh(x));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_floor(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = bf16_decode(g_In[idx]);
        g_Out[idx] = bf16_encode(floor(x));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_ceil(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = bf16_decode(g_In[idx]);
        g_Out[idx] = bf16_encode(ceil(x));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_round(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = bf16_decode(g_In[idx]);
        g_Out[idx] = bf16_encode(x > 0.0 ? floor(x + 0.5) : ceil(x - 0.5));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_sign(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = bf16_decode(g_In[idx]);
        g_Out[idx] = bf16_encode((x > 0.0 ? 1.0 : 0.0) - (x < 0.0 ? 1.0 : 0.0));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_exp(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = bf16_decode(g_In[idx]);
        g_Out[idx] = bf16_encode(exp(x));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_silu(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = bf16_decode(g_In[idx]);
        g_Out[idx] = bf16_encode(x / (1.0 + exp(-x)));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_sqrt(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = bf16_decode(g_In[idx]);
        g_Out[idx] = bf16_encode(sqrt(x));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_sin(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = bf16_decode(g_In[idx]);
        g_Out[idx] = bf16_encode(sin(x));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_cos(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = bf16_decode(g_In[idx]);
        g_Out[idx] = bf16_encode(cos(x));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_neg(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        float x = bf16_decode(g_In[idx]);
        g_Out[idx] = bf16_encode(-x);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_pow(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    if (i < uint(g_Params.count))
    {
        float x = bf16_decode(g_In[i]);
        g_Out[i] = bf16_encode(pow(x, float(g_Params.extra)));
    }
}
[shader("compute")]
[numthreads(256, 1, 1)]
void main_elu(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    if (i < uint(g_Params.count))
    {
        float x = bf16_decode(g_In[i]);
        float alpha = (float)g_Params.extra;
        g_Out[i] = bf16_encode(x >= 0.0 ? x : alpha * (exp(x) - 1.0));
    }
}
