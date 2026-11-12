// Shared push-constant layout for the 2D pooling kernels:
// pool2d, pool2d_f16, pool2d_f64.
module common_pool2d;

public struct Params
{
    public uint64_t planes;
    public uint64_t h;
    public uint64_t w;
    public uint64_t k_h;
    public uint64_t k_w;
    public uint64_t s_h;
    public uint64_t s_w;
    public uint64_t h_out;
    public uint64_t w_out;
};
