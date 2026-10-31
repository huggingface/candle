// RMSNorm over the last dim.
// out[i][j] = x[i][j] / m[i] * alpha[j]
// m[i] = sqrt(sum_j(x[i][j]^2) / n + eps)
import common_norm;

StructuredBuffer<float> g_Src;
StructuredBuffer<float> g_Alpha;
RWStructuredBuffer<float> g_Dst;
layout(push_constant) Params g_Params;
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    if (i >= uint(g_Params.n_rows))
    {
        return;
    }
    uint64_t n = g_Params.n;
    uint64_t base = uint64_t(i) * n;
    float sum2 = 0.0f;
    for (uint64_t j = 0; j < n; j += 1)
    {
        float v = g_Src[base + j];
        // This shader is compiled with `-fp-mode precise` (see
        // `PRECISE_FP_SHADERS` in build.rs) so this multiply-add is not
        // contracted into an FMA; the slow reference path
        // (`rms_norm_slow`) squares with the unary kernel and adds with
        // the reduce kernel, each a separately rounded operation, so
        // the sums must match bit for bit.
        float sq = v * v;
        sum2 += sq;
    }
    // Reproduce `rms_norm_slow` bit for bit: the sum is the same f32
    // left fold, `sum / n` and `+ eps` are the two affine steps (f64
    // multiply / add, each rounded to f32), then f32 sqrt and a true
    // f32 division (not a reciprocal multiply).
    float normed = float(double(sum2) * (1.0 / double(n)));
    float m = sqrt(float(double(normed) + g_Params.eps));
    for (uint64_t j = 0; j < n; j += 1)
    {
        g_Dst[base + j] = (g_Src[base + j] / m) * g_Alpha[j];
    }
}
