// Shared push-constant layout for the const_set fill kernels:
// const_set, const_set_bf16, const_set_f16.
module common_const_set;

public struct Params
{
    public uint64_t count;
    public float value;
};

// Native-width variants for the integer/f64 fills, so `Tensor::full` does not
// round-trip the constant through an f32 (which only keeps 24 significant bits).
public struct I64Params
{
    public uint64_t count;
    public int64_t value;
};

public struct U32Params
{
    public uint64_t count;
    public uint32_t value;
};

public struct F64Params
{
    public uint64_t count;
    public double value;
};
