// Test shader (native bfloat16): fills a BFloat16 storage buffer with all
// 65,536 possible 16-bit bit patterns, using the device's native bfloat16
// support. Thread `i` bit-casts the raw 16-bit pattern `i` into a native
// BFloat16 element, so element `i` holds the exact bit pattern `i`
// (including NaN patterns, which value-conversions would canonicalize).
RWStructuredBuffer<BFloat16> g_OutputBuffer;
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    uint idx = threadId.x;
    if (idx >= 65536) return;
    g_OutputBuffer[idx] = bit_cast<BFloat16>((uint16_t)idx);
}
