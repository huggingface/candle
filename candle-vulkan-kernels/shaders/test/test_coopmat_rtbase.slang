// Probe: cooperative-matrix Load/Store round-trip with a RUNTIME
// (non-constant) base and stride read from g_Params: out = Load(in,
// g_Params[0], g_Params[1]) stored at (g_Params[0], g_Params[1]).
StructuredBuffer<half> g_In;
RWStructuredBuffer<half> g_Out;
StructuredBuffer<uint64_t> g_Params;
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 threadIdx : SV_DispatchThreadID)
{
    uint base = (uint)g_Params[0];
    uint stride = (uint)g_Params[1];
    linalg.CoopMat<half, MemoryScope.Subgroup, 16, 16,
                   linalg.CoopMatMatrixUse.MatrixA> matA =
        linalg.CoopMat<half, MemoryScope.Subgroup, 16, 16,
                       linalg.CoopMatMatrixUse.MatrixA>.Load<
            linalg.CoopMatMatrixLayout.RowMajor>(g_In, base, stride);
    matA.Store<linalg.CoopMatMatrixLayout.RowMajor>(g_Out, base, stride);
}
