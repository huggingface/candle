// Shared push-constant layout for the last-axis reduction kernels:
// reduce_ops, reduce_f16, reduce_f64, reduce_bf16.
module common_reduce;

public struct Params
{
    public uint64_t rows;
    public uint64_t cols;
};
