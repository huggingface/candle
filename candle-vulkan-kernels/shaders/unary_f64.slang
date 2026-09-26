// Unary f64 ops. Each entry point maps `g_In[i] -> g_Out[i]` for `i < g_Params[0]`.
StructuredBuffer<double> g_In;
RWStructuredBuffer<double> g_Out;
StructuredBuffer<float> g_Params; // [0] = element count, [1] = powf exponent (main_pow only)

// Abramowitz-Stegun 7.1.26 approximation (|err| < 1.5e-7),
// matching libm::erff (what candle's CPU path uses) to f64 precision.
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
    // radv (testing-use-only) double-transcendental bug: compute in float.
    if (idx < uint(g_Params[0])) { g_Out[idx] = (double)log((float)g_In[idx]); }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_abs(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0])) { g_Out[idx] = abs(g_In[idx]); }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_recip(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0])) { g_Out[idx] = (double)1.0 / g_In[idx]; }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_sqr(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0])) { g_Out[idx] = g_In[idx] * g_In[idx]; }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_gelu(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0]))
    {
        // Computed in float (radv double-arithmetic crash workaround),
        // matching candle's Gelu::f32.
        float x = (float)g_In[idx];
        g_Out[idx] = (double)(0.5 * x * (1.0 + tanh(0.79788456080286535587989211986876373 * x * (1.0 + 0.044715 * x * x))));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_gelu_erf(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0]))
    {
        // (erf(x / sqrt(2)) + 1) * 0.5 * x, matching candle's GeluErf::f32.
        float x = (float)g_In[idx];
        g_Out[idx] = (double)((erf_approx(x * 0.7071067811865476) + 1.0) * 0.5 * x);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_erf(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0])) { g_Out[idx] = (double)erf_approx((float)g_In[idx]); }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_relu(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0])) { g_Out[idx] = max(g_In[idx], (double)0.0); }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_tanh(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    // radv (testing-use-only) double-transcendental bug: compute in float.
    if (idx < uint(g_Params[0])) { g_Out[idx] = (double)tanh((float)g_In[idx]); }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_floor(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0])) { g_Out[idx] = floor(g_In[idx]); }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_ceil(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0])) { g_Out[idx] = ceil(g_In[idx]); }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_round(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0]))
    {
        // candle's Round is round-half-away-from-zero (Slang's round()
        // rounds half to even).
        float x = (float)g_In[idx];
        g_Out[idx] = (double)(x > 0.0 ? floor(x + 0.5) : ceil(x - 0.5));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_sign(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0]))
    {
        float x = (float)g_In[idx];
        g_Out[idx] = (double)((x > 0.0 ? 1.0 : 0.0) - (x < 0.0 ? 1.0 : 0.0));
    }
}
[shader("compute")]
[numthreads(256, 1, 1)]
void main_pow(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    if (i < uint(g_Params[0]))
    {
        g_Out[i] = (double)pow((float)g_In[i], g_Params[1]);
    }
}
[shader("compute")]
[numthreads(256, 1, 1)]
void main_elu(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    if (i < uint(g_Params[0]))
    {
        float x = (float)g_In[i];
        float alpha = g_Params[1];
        g_Out[i] = (double)(x >= 0.0 ? x : alpha * (exp(x) - 1.0));
    }
}
[shader("compute")]
[numthreads(256, 1, 1)]
void main_exp(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    // radv (testing-use-only) double-transcendental bug: compute in float.
    if (idx < uint(g_Params[0])) { g_Out[idx] = (double)exp((float)g_In[idx]); }
}
[shader("compute")]
[numthreads(256, 1, 1)]
void main_silu(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0]))
    {
        float x = (float)g_In[idx];
        g_Out[idx] = (double)(x / (1.0 + exp(-x)));
    }
}
[shader("compute")]
[numthreads(256, 1, 1)]
void main_sqrt(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    // radv (testing-use-only) double-transcendental bug: compute in float.
    if (idx < uint(g_Params[0])) { g_Out[idx] = (double)sqrt((float)g_In[idx]); }
}
[shader("compute")]
[numthreads(256, 1, 1)]
void main_sin(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0])) { g_Out[idx] = sin(g_In[idx]); }
}
[shader("compute")]
[numthreads(256, 1, 1)]
void main_cos(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0])) { g_Out[idx] = cos(g_In[idx]); }
}
[shader("compute")]
[numthreads(256, 1, 1)]
void main_neg(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0])) { g_Out[idx] = -g_In[idx]; }
}
