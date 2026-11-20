// Shared push-constant layout for the normalization kernels:
// layer_norm, rms_norm.
module common_norm;

public struct Params
{
    public uint64_t n_rows;
    public uint64_t n;
    public double eps;
};
