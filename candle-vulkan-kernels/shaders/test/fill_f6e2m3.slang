// Test shader: fills a 6-bit E2M3 storage buffer with all 64 possible
// values. Thread `i` writes byte `i` (for `i` in 0..64). Each element is
// stored in a byte (the low bits hold the value). Used to validate F6E2M3
// VRAM round-trips end to end.
RWStructuredBuffer<uint8_t> g_OutputBuffer;
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx >= 64) return;
    g_OutputBuffer[idx] = (uint8_t)idx;
}
