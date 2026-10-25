// Shared push-constant layout for the index_add kernels:
// index_add_f16_u32, index_add_f16_i64, index_add_f16_u8,
// index_add_f32_u32, index_add_f32_i64, index_add_f32_u8,
// index_add_f64_u32, index_add_f64_i64, index_add_f64_u8.
module common_index_add;

public struct Params
{
    public uint64_t count;
    public uint64_t post_dim;
    public uint64_t nsrc;
    public uint64_t max_idx;
};
