// Probe: C = A @ B via a single cooperative-matrix multiply-add
// (16x16x16, f16 in / f16 accumulate), then Store C. With A the
// identity the result must equal B.
typealias MatA_t = linalg.CoopMat<half, MemoryScope.Subgroup, 16, 16,
                                  linalg.CoopMatMatrixUse.MatrixA>;
typealias MatB_t = linalg.CoopMat<half, MemoryScope.Subgroup, 16, 16,
                                  linalg.CoopMatMatrixUse.MatrixB>;
typealias MatC_t = linalg.CoopMat<half, MemoryScope.Subgroup, 16, 16,
                                  linalg.CoopMatMatrixUse.MatrixAccumulator>;

StructuredBuffer<half> g_A;
StructuredBuffer<half> g_B;
RWStructuredBuffer<half> g_C;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 groupID : SV_GroupID)
{
    MatA_t matA = MatA_t.Load<linalg.CoopMatMatrixLayout.RowMajor>(g_A, 0, 16);
    MatB_t matB = MatB_t.Load<linalg.CoopMatMatrixLayout.RowMajor>(g_B, 0, 16);
    MatC_t matC;
    matC.fill(0.0h);
    matC = linalg.coopMatMulAdd<half, false, half, half, half,
        MemoryScope.Subgroup, 16, 16, 16>(matA, matB, matC);
    matC.Store<linalg.CoopMatMatrixLayout.RowMajor>(g_C, 0, 16);
}
