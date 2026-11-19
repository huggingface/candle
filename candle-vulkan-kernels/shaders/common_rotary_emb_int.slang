// Shared push-constant layout for the int rotary-embedding kernel.
module common_rotary_emb_int;

public struct Params
{
    public uint64_t b;
    public uint64_t h;
    public uint64_t t;
    public uint64_t d;
    public uint64_t unbatched;
};
