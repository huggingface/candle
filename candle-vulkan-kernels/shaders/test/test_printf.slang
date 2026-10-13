// Test kernel: validates slang `printf()` debug statements for every data
// type supported by the backend.
//
// g_In and g_Out each hold 18 64-bit words. For data type d (0-based),
// word 2*d holds the bit pattern of the lowest representable value and
// word 2*d+1 holds the bit pattern of the highest representable value:
//
//   d = 0: u8, 1: u32, 2: i16, 3: i32, 4: i64,
//   d = 5: bf16, 6: f16, 7: f32, 8: f64
//
// The kernel copies every word from g_In to g_Out (verifiable on the CPU
// side) and prints one "Expected ..., actual ..." line per value. When
// the Khronos validation layer is enabled (VK_LAYER_PRINTF_ENABLE), the
// output appears on stdout (VK_LAYER_PRINTF_TO_STDOUT).
//
// Format specifiers follow the validation layer's restricted printf
// format: 8-bit and 16-bit integers print with the 32-bit specifiers
// (%u / %d), 64-bit integers with %ld, and 16/32/64-bit floats with %f.

StructuredBuffer<uint64_t> g_In;
RWStructuredBuffer<uint64_t> g_Out;

// bf16 -> float (half crate bf16_to_f32 bit pattern).
float bf16_decode(uint16_t b)
{
    uint x = (uint)b;
    if ((x & 0x7FFFu) > 0x7F80u) { return bit_cast<float>(x | 0x40u); }
    return bit_cast<float>(x << 16);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u) { return; }

    // u8: 0 .. 255
    g_Out[0] = g_In[0];
    printf("Expected u8 lowest value '0', actual value '%u'\n", (uint)(g_In[0] & 0xFFu));
    g_Out[1] = g_In[1];
    printf("Expected u8 highest value '255', actual value '%u'\n", (uint)(g_In[1] & 0xFFu));

    // u32: 0 .. 4294967295
    g_Out[2] = g_In[2];
    printf("Expected u32 lowest value '0', actual value '%u'\n", (uint)g_In[2]);
    g_Out[3] = g_In[3];
    printf("Expected u32 highest value '4294967295', actual value '%u'\n", (uint)g_In[3]);

    // i16: -32768 .. 32767
    g_Out[4] = g_In[4];
    printf("Expected i16 lowest value '-32768', actual value '%d'\n", (int)(int16_t)(g_In[4] & 0xFFFFu));
    g_Out[5] = g_In[5];
    printf("Expected i16 highest value '32767', actual value '%d'\n", (int)(int16_t)(g_In[5] & 0xFFFFu));

    // i32: -2147483648 .. 2147483647
    g_Out[6] = g_In[6];
    printf("Expected i32 lowest value '-2147483648', actual value '%d'\n", (int)g_In[6]);
    g_Out[7] = g_In[7];
    printf("Expected i32 highest value '2147483647', actual value '%d'\n", (int)g_In[7]);

    // i64: -9223372036854775808 .. 9223372036854775807
    g_Out[8] = g_In[8];
    printf("Expected i64 lowest value '-9223372036854775808', actual value '%ld'\n", (int64_t)g_In[8]);
    g_Out[9] = g_In[9];
    printf("Expected i64 highest value '9223372036854775807', actual value '%ld'\n", (int64_t)g_In[9]);

    // bf16: -3.3895313892515355e+38 .. 3.3895313892515355e+38
    g_Out[10] = g_In[10];
    printf("Expected bf16 lowest value '-3.3895313892515355e+38', actual value '%f'\n", bf16_decode((uint16_t)(g_In[10] & 0xFFFFu)));
    g_Out[11] = g_In[11];
    printf("Expected bf16 highest value '3.3895313892515355e+38', actual value '%f'\n", bf16_decode((uint16_t)(g_In[11] & 0xFFFFu)));

    // f16: -65504 .. 65504
    g_Out[12] = g_In[12];
    printf("Expected f16 lowest value '-65504', actual value '%f'\n", bit_cast<half>((uint16_t)(g_In[12] & 0xFFFFu)));
    g_Out[13] = g_In[13];
    printf("Expected f16 highest value '65504', actual value '%f'\n", bit_cast<half>((uint16_t)(g_In[13] & 0xFFFFu)));

    // f32: -3.4028234663852886e+38 .. 3.4028234663852886e+38
    g_Out[14] = g_In[14];
    printf("Expected f32 lowest value '-3.4028234663852886e+38', actual value '%f'\n", bit_cast<float>((uint)g_In[14]));
    g_Out[15] = g_In[15];
    printf("Expected f32 highest value '3.4028234663852886e+38', actual value '%f'\n", bit_cast<float>((uint)g_In[15]));

    // f64: -1.7976931348623157e+308 .. 1.7976931348623157e+308
    g_Out[16] = g_In[16];
    printf("Expected f64 lowest value '-1.7976931348623157e+308', actual value '%f'\n", bit_cast<double>(g_In[16]));
    g_Out[17] = g_In[17];
    printf("Expected f64 highest value '1.7976931348623157e+308', actual value '%f'\n", bit_cast<double>(g_In[17]));
}
