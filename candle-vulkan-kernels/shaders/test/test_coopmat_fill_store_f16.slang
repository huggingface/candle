// Probe: fill a 16x16 f16 cooperative-matrix accumulator with 1.0h and
// Store it to g_Out. If the Store path works, g_Out[i] == 1.0 for all i.
typealias MatC_t = linalg.CoopMat<half, MemoryScope.Subgroup, 16, 16,
                                  linalg.CoopMatMatrixUse.MatrixAccumulator>;

RWStructuredBuffer<half> g_Out;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 groupID : SV_GroupID)
{
    MatC_t matC;
    matC.fill(1.0h);
    matC.Store<linalg.CoopMatMatrixLayout.RowMajor>(g_Out, 0, 16);
}
