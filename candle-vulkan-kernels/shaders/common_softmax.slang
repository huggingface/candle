// Shared push-constant layout for the softmax kernel.
module common_softmax;

public struct Params
{
    public uint64_t n_rows;
    public uint64_t n;
};
