// Shared push-constant layout for the index-select gather kernels:
// gather_idx, gather_idx_bf16, gather_idx_f16, gather_idx_f64,
// gather_idx_i64, gather_idx_u32, gather_idx_u8.
// The `main` entry reads only count/block_len/emb_h; the
// gather-rows/index-select entries also read dst_offset.
module common_gather_idx;

public struct Params
{
    public uint64_t count;
    public uint64_t block_len;
    public uint64_t emb_h;
    public uint64_t dst_offset;
};
