// Test shader: fills a float (32-bit) storage buffer with 256 distinct values,
// some exceeding the f16 maximum finite value (65504), so the result cannot be
// an f16. Thread `i` writes `(i + 1) * 1000.0`.
RWStructuredBuffer<float> g_OutputBuffer;
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx >= 256) return;
    g_OutputBuffer[idx] = float(idx + 1u) * 1000.0;
}
