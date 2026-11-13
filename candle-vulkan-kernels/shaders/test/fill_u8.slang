// Test shader: fills a uint8_t storage buffer with all 256 possible byte
// values. Thread `i` writes `i` (for `i` in 0..256). Used to validate u8
// VRAM round-trips end to end.
RWStructuredBuffer<uint8_t> g_OutputBuffer;
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx >= 256) return;
    g_OutputBuffer[idx] = (uint8_t)idx;
}
