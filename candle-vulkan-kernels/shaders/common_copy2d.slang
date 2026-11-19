// Shared push-constant layout for the 2D row-wise copy kernels:
// copy2d, copy2d_bf16, copy2d_f16, copy2d_f64, copy2d_i64,
// copy2d_u32, copy2d_u8.
module common_copy2d;

public struct Params
{
    public uint64_t d1;
    public uint64_t d2;
    public uint64_t src_stride;
    public uint64_t dst_stride;
    public uint64_t src_offset;
    public uint64_t dst_offset;
};
