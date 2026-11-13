// Shared push-constant layout for the byte-granular const-fill kernel.
module common_const_fill_bytes;

public struct Params
{
    public uint64_t count;
    public uint32_t pattern;
};
