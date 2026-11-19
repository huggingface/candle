// Shared push-constant layout for the where (select) kernels:
// where, where_bf16, where_f16, where_f64, where_f8e4m3.
module common_where;

public struct Params
{
    public uint64_t count;
    public uint64_t ndim;
    public uint64_t pred_dims[4];
    public uint64_t true_dims[4];
    public uint64_t false_dims[4];
};
