// Tiled f16 affine (matrix product + bias) via cooperative matrices:
// out[b, m, n] = lhs[b, m, k] @ rhs[b, k, n] + bias[n].
//
// Same tiling as the cooperative-matrix GEMM kernel: one 64-thread
// workgroup (one subgroup) computes a 16x16 output tile and the K
// dimension is reduced in 16-wide slabs, accumulated in the f16
// accumulator matrix (the device reports f16 output for the 16-bit
// input cooperative-matrix shapes). The bias tile seeds the
// accumulator through a zero-row-stride load: the same 16 bias values
// are loaded into all 16 rows of the tile, so the bias is added once
// to the accumulated product per output column.
//
// The tile row strides (k for lhs, n for rhs/out) must satisfy the
// cooperative-matrix load/store alignment: a multiple of 8 elements
// (16 bytes) for f16. m, n and k need not be multiples of 16:
// out-of-range elements load as zero and out-of-range stores are
// dropped. The bias load is always in range: the bias has n elements
// and n is a multiple of 8.
//
// The rhs may be stored transposed as (b, n, k); params[4]
// (rhs_transposed, 0/1) selects the load layout.
//
// The cooperative-matrix load/store base and stride are computed in
// 32-bit unsigned arithmetic: some drivers (RADV) miscompile the
// load/store when the base/stride are passed as 64-bit values.
//
// Params: [0]=batch, [1]=m, [2]=n, [3]=k, [4]=rhs_transposed (0/1)
typealias MatA_t =
    linalg.CoopMat<half, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixA>;
typealias MatB_t =
    linalg.CoopMat<half, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixB>;
typealias MatC_t =
    linalg.CoopMat<half, MemoryScope.Subgroup, 16, 16,
                   linalg.CoopMatMatrixUse.MatrixAccumulator>;

StructuredBuffer<half> g_Lhs;
StructuredBuffer<half> g_Rhs;
StructuredBuffer<half> g_Bias;
RWStructuredBuffer<half> g_Out;
StructuredBuffer<uint64_t> g_Params;

[shader("compute")]
[numthreads(64, 1, 1)]
void main_affine(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID : SV_GroupID)
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

    // Zero row-stride load: bias[col..col+16] replicated across all
    // 16 rows of the tile.
    MatC_t matC = MatC_t.Load<linalg.CoopMatMatrixLayout.RowMajor>(
        g_Bias, col, 0);
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
        matC = linalg.coopMatMulAdd<half, false, half, half, half,
            MemoryScope.Subgroup, 16, 16, 16>(matA, matB, matC);
    }
    matC.Store<linalg.CoopMatMatrixLayout.RowMajor>(
        g_Out, batch * m * n + row * n + col, sn);
}
