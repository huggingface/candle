// Test shader: fills a uint (32-bit) storage buffer with 256 distinct values,
// each exceeding the 16-bit unsigned range (65535), so the result cannot be a
// truncated u16/u8. Thread `i` writes `(i + 1) * 100000 + 65536`.
RWStructuredBuffer<uint> g_OutputBuffer;
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx >= 256) return;
    g_OutputBuffer[idx] = (idx + 1u) * 100000u + 65536u;
}
