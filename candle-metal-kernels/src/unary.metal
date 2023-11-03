#include <metal_stdlib>

using namespace metal;

struct Input {
    device float *input;
    device float *output;
};

kernel void cos(device Input& args [[ buffer(0) ]],                       uint index [[thread_position_in_grid]])
{
    args.output[index] = cos(args.input[index]);
}

