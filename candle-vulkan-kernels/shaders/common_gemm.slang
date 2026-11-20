// Shared push-constant layout for the tiled GEMM and the affine
// cooperative-matrix kernels: gemm, gemm_f16, gemm_f64,
// gemm_coopmat_f16, gemm_coopmat_bf16, affine_coopmat_f16,
// affine_coopmat_bf16. The classic tiled entry points use the first five
// fields; the cooperative-matrix entry points use all seven (`sk` and `sn`
// are dedicated stride copies of `k` and `n`, see gemm_params).
module common_gemm;

public struct Params
{
    public uint64_t batch;
    public uint64_t m;
    public uint64_t n;
    public uint64_t k;
    public uint64_t rhs_transposed;
    public uint64_t sk;
    public uint64_t sn;
};
