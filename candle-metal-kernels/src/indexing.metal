#include <metal_stdlib>
#include <metal_config>
#define METAL_FUNC inline __attribute__((__always_inline__))
using namespace metal;

struct fault_counter {
    uint counter;
    uint tolerance;

    fault_counter(uint tolerance) {
        this->counter = 0;
        this->tolerance = tolerance;
    }

    bool quit() {
        counter += 1;
        return (counter > tolerance);
    }
};

constant uint IDS_DIM_SIZE [[function_constant(0)]];
constant uint SRC_DIM_SIZE [[function_constant(1)]];
constant uint DST_DIM_SIZE [[function_constant(2)]];
constant uint LEFT_SIZE [[function_constant(3)]];
constant uint RIGHT_SIZE [[function_constant(4)]];
constant uint NUMEL = LEFT_SIZE * RIGHT_SIZE;

kernel void index_add(
    device uint *ids [[buffer(0)]],
    device float *inp [[buffer(1)]],
    device float *out [[buffer(2)]],

    uint grid_size [[threadgroups_per_grid]],           // gridDim
    uint gid [[thread_position_in_grid]],               // blockIdx
    uint num_threads [[threads_per_grid]],              // blockDim
    uint thread_index [[thread_index_in_threadgroup]]   // threadIdx
) {
    for (uint i = gid * num_threads + thread_index; i < NUMEL; i += num_threads * grid_size) {
        const uint pre = i / RIGHT_SIZE;
        const uint post = i % RIGHT_SIZE;

        for (uint j = 0; j < IDS_DIM_SIZE; j++) {
            const uint idx = ids[j];
            const uint src_i = (pre * IDS_DIM_SIZE + j) * RIGHT_SIZE + post;
            const uint dst_i = (pre * DST_DIM_SIZE + idx) * RIGHT_SIZE + post;
            out[dst_i] += inp[src_i];
        }
    }
}