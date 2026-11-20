// Test shader (emulated bfloat16): fills a 16-bit storage buffer with all
// 65,536 possible bit patterns using raw uint16_t stores. For devices without
// native bfloat16 the buffer is the same 16-bit footprint, so the bits written
// here map exactly to the sequential index positions. Thread `i` stores `i`.
RWStructuredBuffer<uint16_t> g_OutputBuffer;
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    uint idx = threadId.x;
    if (idx >= 65536) return;
    g_OutputBuffer[idx] = uint16_t(idx);
}
