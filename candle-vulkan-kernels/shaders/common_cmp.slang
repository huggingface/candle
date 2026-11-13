// Shared push-constant layout for the broadcast comparison kernels:
// cmp, cmp_bf16, cmp_f16, cmp_f64, cmp_f8e4m3, cmp_i16,
// cmp_i32, cmp_i64, cmp_u32, cmp_u8.
module common_cmp;

public struct Params
{
    public uint64_t count;
    public uint64_t lhs_ndim;
    public uint64_t rhs_ndim;
    public uint64_t lhs_dims[4];
    public uint64_t rhs_dims[4];
};
