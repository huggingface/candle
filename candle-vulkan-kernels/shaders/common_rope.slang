// Shared push-constant layout for the rotary-embedding kernel (rope).
module common_rope;

public struct Params
{
    public uint64_t bh;
    public uint64_t t;
    public uint64_t d;
};
