// Shared push-constant layout for the scatter kernels:
// scatter, scatter_f16, scatter_f64.
module common_scatter;

public struct Params
{
    public uint64_t dst_dim_len;
    public uint64_t dst_right_len;
    public uint64_t ids_dim_len;
    public uint64_t ids_right_len;
    public uint64_t ids_left_len;
};
