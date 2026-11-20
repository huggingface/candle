// Shared push-constant layout for the last-axis argsort kernels.
module common_argsort;

public struct Params
{
    public uint64_t rows;
    public uint64_t cols;
};
