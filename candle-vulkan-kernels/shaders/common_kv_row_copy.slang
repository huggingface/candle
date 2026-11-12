// Shared push-constant layout for the kv_row_copy kernel.
module common_kv_row_copy;

public struct Params
{
    public uint64_t heads;
    public uint64_t head_dim;
    public uint64_t row;
    public uint64_t stride;
};
