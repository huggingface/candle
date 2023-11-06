#include <metal_stdlib>

using namespace metal;

template <typename T>
kernel void unary_cos(device const T *input, device T *output,                       uint index [[thread_position_in_grid]])
{
    output[index] = cos(input[index]);
}

#define UNARY(FN, TYPENAME, FN_NAME) \
kernel void FN_NAME(device const TYPENAME *input, device TYPENAME *output, uint index [[thread_position_in_grid]]) \
{ \
    output[index] = FN(input[index]);\
}

UNARY(cos, float, cos_float);
UNARY(cos, half, cos_half);

#if __METAL_VERSION__ >= 310
UNARY(cos, half, cos_half);
#endif
