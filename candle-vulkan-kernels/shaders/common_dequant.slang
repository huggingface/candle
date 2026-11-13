// Shared push-constant layout for the quantized dequantize kernels:
// dequant_bf16, dequant_f16, dequant_f32, dequant_q2k, dequant_q3k,
// dequant_q4_0, dequant_q4_1, dequant_q4k, dequant_q5_0, dequant_q5_1,
// dequant_q5k, dequant_q6k, dequant_q8_0, dequant_q8_1, dequant_q8k.
module common_dequant;

public struct Params
{
    public uint64_t num_blocks;
};
