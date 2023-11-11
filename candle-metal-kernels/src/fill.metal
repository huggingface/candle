#include <metal_stdlib>
using namespace metal;

template<typename T>
void fill(
    device T *buffer [[buffer(0)]],
    constant T &value,
    constant size_t &numel,
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= numel) return;
    buffer[gid] = value;
}

#define FILL_OP(TYPENAME, FN_NAME) \
kernel void FN_NAME( \
    device TYPENAME *buffer [[buffer(0)]], \
    constant TYPENAME &value, \
    constant size_t &numel, \
    uint gid [[thread_position_in_grid]] \
) { fill<TYPENAME>(buffer, value, numel, gid); } \

FILL_OP(uint8_t, fill_u8)
FILL_OP(uint32_t, fill_u32)
FILL_OP(int64_t, fill_i64)
FILL_OP(half, fill_f16)
FILL_OP(float, fill_f32)

#if __METAL_VERSION__ >= 310
FILL_OP(bfloat, fill_bf16)
#endif
