// Unary bf16 ops (emulated). Each entry point maps g_In[i] -> g_Out[i] for i < g_Params[0],
// decoding to float, applying the op in float, and encoding back to bf16.
StructuredBuffer<uint16_t> g_In;
RWStructuredBuffer<uint16_t> g_Out;
StructuredBuffer<float> g_Params; // [0] = element count

// bf16 -> float (half crate bf16_to_f32).
float bf16_decode(uint16_t b)
{
    uint x = (uint)b;
    if ((x & 0x7FFFu) > 0x7F80u) { return bit_cast<float>(x | 0x40u); }
    return bit_cast<float>(x << 16);
}

// float -> bf16 (half crate f32_to_bf16, round-to-nearest-even).
uint16_t bf16_encode(float v)
{
    uint x = bit_cast<uint>(v);
    if ((x & 0x7FFFFFFFu) > 0x7F800000u) { return (uint16_t)((x >> 16) | 0x0040u); }
    uint rb = 0x8000u;
    if ((x & rb) != 0u && (x & (3u * rb - 1u)) != 0u) { return (uint16_t)((x >> 16) + 1u); }
    return (uint16_t)(x >> 16);
}


// Abramowitz-Stegun 7.1.26 approximation (|err| < 1.5e-7),
// matching libm::erff (what candle's CPU path uses) to f32 precision.
float erf_approx(float x)
{
    float sign = x < 0.0 ? -1.0 : 1.0;
    float ax = abs(x);
    float t = 1.0 / (1.0 + 0.3275911 * ax);
    float poly = 0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)));
    return sign * (1.0 - t * poly * exp(-ax * ax));
}


[shader("compute")]
[numthreads(256, 1, 1)]
void main_log(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0]))
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
    if (idx < uint(g_Params[0]))
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
    if (idx < uint(g_Params[0]))
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
    if (idx < uint(g_Params[0]))
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
    if (idx < uint(g_Params[0]))
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
    if (idx < uint(g_Params[0]))
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
    if (idx < uint(g_Params[0]))
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
    if (idx < uint(g_Params[0]))
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
    if (idx < uint(g_Params[0]))
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
    if (idx < uint(g_Params[0]))
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
    if (idx < uint(g_Params[0]))
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
    if (idx < uint(g_Params[0]))
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
    if (idx < uint(g_Params[0]))
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
    if (idx < uint(g_Params[0]))
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
    if (idx < uint(g_Params[0]))
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
    if (idx < uint(g_Params[0]))
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
    if (idx < uint(g_Params[0]))
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
    if (idx < uint(g_Params[0]))
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
    if (idx < uint(g_Params[0]))
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
    if (i < uint(g_Params[0]))
    {
        float x = bf16_decode(g_In[i]);
        g_Out[i] = bf16_encode(pow(x, g_Params[1]));
    }
}
[shader("compute")]
[numthreads(256, 1, 1)]
void main_elu(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    if (i < uint(g_Params[0]))
    {
        float x = bf16_decode(g_In[i]);
        float alpha = g_Params[1];
        g_Out[i] = bf16_encode(x >= 0.0 ? x : alpha * (exp(x) - 1.0));
    }
}
