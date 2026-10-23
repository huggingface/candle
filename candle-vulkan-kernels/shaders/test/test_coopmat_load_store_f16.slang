// Probe: Load a 16x16 f16 tile from g_In and Store it unchanged to
// g_Out (row-major, row stride 16). A working Load/Store pair round-
// trips: g_Out == g_In.
typealias MatA_t = linalg.CoopMat<half, MemoryScope.Subgroup, 16, 16,
                                  linalg.CoopMatMatrixUse.MatrixA>;

StructuredBuffer<half> g_In;
RWStructuredBuffer<half> g_Out;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 groupID : SV_GroupID)
{
    MatA_t matA = MatA_t.Load<linalg.CoopMatMatrixLayout.RowMajor>(g_In, 0, 16);
    matA.Store<linalg.CoopMatMatrixLayout.RowMajor>(g_Out, 0, 16);
}
