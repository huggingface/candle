// Test shader: fills an int16_t storage buffer with all 65,536 possible
// 16-bit bit patterns. Thread `i` writes the bit pattern of `i` reinterpreted
// as a signed 16-bit integer. Used to validate i16 VRAM round-trips.
RWStructuredBuffer<int16_t> g_OutputBuffer;
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx >= 65536) return;
    g_OutputBuffer[idx] = bit_cast<int16_t>((uint16_t)idx);
}
