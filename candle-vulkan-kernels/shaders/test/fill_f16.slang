// Define a structured buffer to hold the resulting f16 (half) values
RWStructuredBuffer<half> g_OutputBuffer;

[shader("compute")]
[numthreads(64, 1, 1)] // 64 threads per thread group
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // The total number of possible 16-bit combinations is 65,536
    uint threadIdx = dispatchThreadID.x;
    
    if (threadIdx >= 65536)
        return;

    // 1. Interpret the thread index as a 16-bit unsigned integer bit pattern
    uint16_t bitPattern = (uint16_t)threadIdx;

    // 2. Reinterpret those exact bits as a 16-bit floating-point number (f16/half)
    half f16Value = bit_cast<half>(bitPattern);

    // 3. Store the value into the buffer at the corresponding index
    g_OutputBuffer[threadIdx] = f16Value;
}
