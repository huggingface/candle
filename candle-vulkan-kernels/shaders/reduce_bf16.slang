// Last-axis reductions, BF16 (stored as the 16-bit bit pattern).
// out[row] = sum_c in[row*cols + c]                      (main_reduce_sum)
// out[row] = max_c in[row*cols + c]                      (main_reduce_max)
// out[row] = min_c in[row*cols + c]                      (main_reduce_min)
// out[row] = argmin_c in[row*cols + c]                   (main_reduce_argmin)
// out[row] = argmax_c in[row*cols + c]                   (main_reduce_argmax)
StructuredBuffer<uint16_t> g_In;
RWStructuredBuffer<uint16_t> g_OutF;
RWStructuredBuffer<uint> g_OutI;
StructuredBuffer<float> g_Params; // [0]=rows, [1]=cols

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

[shader("compute")]
[numthreads(1, 1, 1)]
void main_reduce_min(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint row = dispatchThreadID.x;
    uint rows = uint(g_Params[0]);
    uint cols = uint(g_Params[1]);
    if (row >= rows || cols == 0u)
    {
        return;
    }
    float acc = bf16_decode(g_In[row * cols]);
    for (uint c = 1u; c < cols; c += 1u)
    {
        float v = bf16_decode(g_In[row * cols + c]);
        if (v < acc)
        {
            acc = v;
        }
    }
    g_OutF[row] = bf16_encode(acc);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void main_reduce_argmin(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint row = dispatchThreadID.x;
    uint rows = uint(g_Params[0]);
    uint cols = uint(g_Params[1]);
    if (row >= rows || cols == 0u)
    {
        return;
    }
    float best = bf16_decode(g_In[row * cols]);
    uint besti = 0u;
    for (uint c = 1u; c < cols; c += 1u)
    {
        float v = bf16_decode(g_In[row * cols + c]);
        if (v < best)
        {
            best = v;
            besti = c;
        }
    }
    g_OutI[row] = besti;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void main_reduce_argmax(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint row = dispatchThreadID.x;
    uint rows = uint(g_Params[0]);
    uint cols = uint(g_Params[1]);
    if (row >= rows || cols == 0u)
    {
        return;
    }
    float best = bf16_decode(g_In[row * cols]);
    uint besti = 0u;
    for (uint c = 1u; c < cols; c += 1u)
    {
        float v = bf16_decode(g_In[row * cols + c]);
        if (v > best)
        {
            best = v;
            besti = c;
        }
    }
    g_OutI[row] = besti;
}
[shader("compute")]
[numthreads(1, 1, 1)]
void main_reduce_sum(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint row = dispatchThreadID.x;
    uint rows = uint(g_Params[0]);
    uint cols = uint(g_Params[1]);
    if (row >= rows || cols == 0u)
    {
        return;
    }
    float acc = 0.0;
    for (uint c = 0u; c < cols; c += 1u)
    {
        acc += bf16_decode(g_In[row * cols + c]);
    }
    g_OutF[row] = bf16_encode(acc);
}
[shader("compute")]
[numthreads(1, 1, 1)]
void main_reduce_max(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint row = dispatchThreadID.x;
    uint rows = uint(g_Params[0]);
    uint cols = uint(g_Params[1]);
    if (row >= rows || cols == 0u)
    {
        return;
    }
    float acc = -3.402823466e+38;
    for (uint c = 0u; c < cols; c += 1u)
    {
        float v = bf16_decode(g_In[row * cols + c]);
        if (v > acc)
        {
            acc = v;
        }
    }
    g_OutF[row] = bf16_encode(acc);
}
