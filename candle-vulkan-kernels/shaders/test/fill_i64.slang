// Test shader: fills an int64_t storage buffer with 256 distinct values, each
// exceeding the 32-bit signed range (+-2,147,483,648), so the result cannot be
// a truncated i32. Thread `i` writes `(i - 128) * 1,000,000,000,000`.
RWStructuredBuffer<int64_t> g_OutputBuffer;
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx >= 256) return;
    int64_t s = (int64_t)idx - (int64_t)128;
    g_OutputBuffer[idx] = s * (int64_t)1000000000000;
}
