// Shared push-constant layout for the upsample kernels:
// upsample, upsample_f16, upsample_f64.
module common_upsample;

public struct Params
{
    public uint64_t planes;
    public uint64_t d0;
    public uint64_t d1;
    public uint64_t d2;
    public uint64_t d3;
    public double scale_h;
    public double scale_w;
    public uint64_t align;
};
