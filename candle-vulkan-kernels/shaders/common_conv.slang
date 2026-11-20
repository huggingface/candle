// Shared push-constant layout for the convolution kernels (1D and 2D,
// forward and transposed): conv, conv_bf16, conv_f16, conv_f64.
// Slots 3/4 are the kernel extents (k_size for 1D, k_h/k_w for 2D),
// 5/6 the input extents, 7/8 the output extents.
module common_conv;

public struct Params
{
    public uint64_t b;
    public uint64_t c_in;
    public uint64_t c_out;
    public uint64_t k1;
    public uint64_t k2;
    public uint64_t i1;
    public uint64_t i2;
    public uint64_t o1;
    public uint64_t o2;
    public uint64_t padding;
    public uint64_t stride;
    public uint64_t dilation;
};
