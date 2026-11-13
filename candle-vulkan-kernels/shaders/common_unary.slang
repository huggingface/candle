// Shared push-constant layout and helpers for the unary op kernels:
// unary, unary_bf16, unary_f16, unary_f64, unary_f8e4m3.
module common_unary;

public struct Params
{
    public uint64_t count;
    public double extra;
};

// Abramowitz-Stegun 7.1.26 approximation (|err| < 1.5e-7),
// matching libm::erff (what candle's CPU path uses) to f32 precision.
public float erf_approx(float x)
{
    float sign = x < 0.0 ? -1.0 : 1.0;
    float ax = abs(x);
    float t = 1.0 / (1.0 + 0.3275911 * ax);
    float poly = 0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)));
    return sign * (1.0 - t * poly * exp(-ax * ax));
}
