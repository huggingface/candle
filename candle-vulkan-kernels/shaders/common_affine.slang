// Shared push-constant layout for the affine kernels (dst = src * mul + add):
// affine, affine_bf16, affine_f16, affine_f64, affine_f8e4m3.
module common_affine;

public struct Params
{
    public uint64_t total;
    public double mul;
    public double add;
};
