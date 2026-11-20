// DP4A probe: g_Out[i] = dot4add_u8packed(g_A[i], g_B[i], 0).
//
// Verifies the VK_KHR_shader_integer_dot_product path (OpUDot with
// PackedVectorFormat4x8Bit / DotProductInput4x8BitPacked capability)
// actually compiles and runs on the device. Used by the integration test
// to prove the hardware integer-dot-product instruction is reachable.
StructuredBuffer<uint> g_A; // 4 packed i8 activations per u32
StructuredBuffer<uint> g_B; // 4 packed i8 weights per u32
RWStructuredBuffer<uint> g_Out;
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    g_Out[i] = dot4add_u8packed(g_A[i], g_B[i], 0u);
}