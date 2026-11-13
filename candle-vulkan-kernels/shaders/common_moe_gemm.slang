// Shared push-constant layout for the fused routed-MoE dequant+GEMM
// kernels: fused_q2k_moe_gemm, fused_q4k_moe_gemm, fused_q6k_moe_gemm,
// fused_q8_0_moe_gemm.
//
// `n` is the per-expert output dim, `k` the per-expert input dim, and
// `weight_expert_bytes` the byte size of one expert's (n, k) quantized
// slice (used to offset into the (experts, n, k) weight buffer by the
// route's selected expert).
module common_moe_gemm;

public struct Params
{
    public uint64_t n;
    public uint64_t k;
    public uint64_t weight_expert_bytes;
};
