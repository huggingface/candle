// Shared push-constant layout for the fused dequant+GEMM kernels:
// fused_q4_0_gemm, fused_q4k_gemm, fused_q4_0_gemm_m, fused_q4k_gemm_m,
// fused_q5_0_gemm, fused_q5_1_gemm, fused_q5k_gemm, fused_q5k_gemm_m,
// fused_q8_0_gemm, fused_q8_1_gemm, fused_q8k_gemm, fused_q8k_gemm_m.
module common_fused_gemm;

public struct Params
{
    public uint64_t m;
    public uint64_t n;
    public uint64_t k;
};
