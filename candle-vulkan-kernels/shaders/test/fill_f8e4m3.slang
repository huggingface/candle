// Test shader: fills a FP8 E4M3 (emulated as raw bytes) storage buffer with all 256 possible
// values. Thread `i` writes byte `i` (for `i` in 0..256). Each element is
// stored in a byte (the low bits hold the value). Used to validate F8E4M3
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
