// Tiled bf16 GEMM via cooperative matrices:
// out[b, m, n] = lhs[b, m, k] @ rhs[b, k, n].
//
// One 64-thread workgroup (one subgroup) computes a 16x16 output tile;
// the K dimension is reduced in 16-wide slabs. The product is
// accumulated in the f16 accumulator matrix (the device reports f16
// output for the 16-bit input cooperative-matrix shapes); each output
// tile is converted to BFloat16 element by element on the way to the
// output buffer. The tile row strides (k for lhs, n for rhs/out) must
// satisfy the cooperative-matrix load/store alignment: a multiple of 8
// elements (16 bytes) for the 16-bit element types. m, n and k need
// not be multiples of 16: out-of-range elements load as zero and
// out-of-range stores are dropped.
//
// The rhs may be stored transposed as (b, n, k); params[4]
// (rhs_transposed, 0/1) selects the load layout.
//
// The cooperative-matrix load base and stride are computed in 32-bit
// unsigned arithmetic: some drivers (RADV) miscompile the load when
// the base/stride are passed as 64-bit values.
//
// Params: [0]=batch, [1]=m, [2]=n, [3]=k, [4]=rhs_transposed (0/1),
// [5]=k (cooperative-matrix stride copy), [6]=n (cooperative-matrix
// stride copy).
//
// Some drivers (RADV) miscompile the cooperative-matrix load when the
// Stride operand is the same value as one feeding the k loop
// condition: the stride is read from dedicated params slots
// (params[5]=k, params[6]=n) so it is a distinct SSA value.
typealias MatA_t =
    linalg.CoopMat<BFloat16, MemoryScope.Subgroup, 16, 16,
                   linalg.CoopMatMatrixUse.MatrixA>;
typealias MatB_t =
    linalg.CoopMat<BFloat16, MemoryScope.Subgroup, 16, 16,
                   linalg.CoopMatMatrixUse.MatrixB>;
typealias MatC_t =
    linalg.CoopMat<half, MemoryScope.Subgroup, 16, 16,
                   linalg.CoopMatMatrixUse.MatrixAccumulator>;

StructuredBuffer<BFloat16> g_Lhs;
StructuredBuffer<BFloat16> g_Rhs;
RWStructuredBuffer<BFloat16> g_Out;
StructuredBuffer<uint64_t> g_Params;

[shader("compute")]
[numthreads(64, 1, 1)]
void main_gemm(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID : SV_GroupID,
               uint3 localID : SV_GroupThreadID)
{
    uint batch = groupID.z;
    uint row = groupID.x * 16u;
    uint col = groupID.y * 16u;
    uint m = (uint)g_Params[1];
    uint n = (uint)g_Params[2];
    uint k = (uint)g_Params[3];
    uint rt = (uint)g_Params[4];
    uint sk = (uint)g_Params[5];
    uint sn = (uint)g_Params[6];

    MatC_t matC;
    matC.fill(0.0h);
    for (uint k0 = 0u; k0 < k; k0 += 16u)
    {
        MatA_t matA = MatA_t.Load<linalg.CoopMatMatrixLayout.RowMajor>(
            g_Lhs, batch * m * k + row * k + k0, sk);
        MatB_t matB;
        if (rt == 1u)
        {
            matB = MatB_t.Load<linalg.CoopMatMatrixLayout.ColumnMajor>(
                g_Rhs, batch * n * k + col * k + k0, sk);
        }
        else
        {
            matB = MatB_t.Load<linalg.CoopMatMatrixLayout.RowMajor>(
                g_Rhs, batch * k * n + k0 * n + col, sn);
        }
        matC = linalg.coopMatMulAdd<half, false, BFloat16, BFloat16, half,
            MemoryScope.Subgroup, 16, 16, 16>(matA, matB, matC);
    }
    // Convert the f16 accumulator tile to BFloat16 and store it; the
    // accumulator's Store requires an f16 buffer, so the conversion
    // happens here instead.
    uint outBase = batch * m * n + row * n + col;
    for (uint i = localID.x; i < 256u; i += 64u)
    {
        uint r = i / 16u;
        uint c = i % 16u;
        if (row + r < m && col + c < n)
        {
            g_Out[outBase + r * n + c] = BFloat16(matC[i]);
        }
    }
}
