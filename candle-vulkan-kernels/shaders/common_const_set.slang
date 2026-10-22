// Shared push-constant layout for the const_set fill kernels:
// const_set, const_set_bf16, const_set_f16.
module common_const_set;

public struct Params
{
    public uint64_t count;
    public float value;
};
