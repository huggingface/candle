// Test shader: fills a double (64-bit) storage buffer with 256 distinct values
// that are not exactly representable in f32 (each needs more than 24
// significant bits), so the result cannot be an f32. Thread `i` writes
// `(i + 1) * 16,777,217.0` (16,777,217 = 2^24 + 1).
RWStructuredBuffer<double> g_OutputBuffer;
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx >= 256) return;
    // Multiply by an exact integer literal (promoted to double), avoiding a
    // float32 literal that would already be rounded below 2^24 + 1.
    g_OutputBuffer[idx] = double(idx + 1u) * 16777217;
}
