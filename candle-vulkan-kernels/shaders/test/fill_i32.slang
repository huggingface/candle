// Test shader: fills an int (32-bit) storage buffer with 256 distinct signed
// values that straddle the 16-bit range (below -32768 and above 32767), so the
// result cannot be a truncated i16. Thread `i` writes `(i - 128) * 100000`.
RWStructuredBuffer<int> g_OutputBuffer;
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx >= 256) return;
    int s = (int)idx - 128;
    g_OutputBuffer[idx] = s * 100000;
}
