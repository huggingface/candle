// Test shader (native bfloat16): fills a BFloat16 storage buffer with all
// 65,536 possible 16-bit bit patterns, using the device's native bfloat16
// support. Thread `i` stores the f32 whose top 16 bits are `i`, cast to
// BFloat16, so element `i` holds the bit pattern `i`.
RWStructuredBuffer<BFloat16> g_OutputBuffer;
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    uint idx = threadId.x;
    if (idx >= 65536) return;
    // Shift the 16-bit pattern into the upper 16 bits of an f32 bit pattern.
    uint f32_bits = idx << 16;
    float f32_val = asfloat(f32_bits);
    // Cast the float to native BFloat16; the top 16 bits are exactly `idx`.
    g_OutputBuffer[idx] = BFloat16(f32_val);
}
