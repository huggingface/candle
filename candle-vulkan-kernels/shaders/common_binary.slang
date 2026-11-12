// Shared push-constant layout for the broadcast binary-op kernels:
// binary, binary_bf16, binary_f16, binary_f64, binary_f8e4m3,
// binary_i16, binary_i32, binary_i64, binary_u32, binary_u8.
module common_binary;

public struct Params
{
    public uint64_t count;
    public uint64_t lhs_ndim;
    public uint64_t rhs_ndim;
    public uint64_t lhs_dims[8];
    public uint64_t rhs_dims[8];
    public uint64_t lhs_data_ndim;
    public uint64_t lhs_data_dims[8];
};
