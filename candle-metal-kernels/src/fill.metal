#include <metal_stdlib>

using namespace metal;

template<typename T> METAL_FUNC void fill_with(
    device T *out,
    constant float &value,
    constant size_t &numel,
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= numel) {
        return;
    }
    out[tid] = static_cast<T>(value);
}

#define FILL_OP(NAME, T)                                \
kernel void fill_##NAME(                                \
    device T *out,                                      \
    constant float &value,                              \
    constant size_t &numel,                              \
    uint tid [[thread_position_in_grid]]                \
) {                                                     \
    fill_with<T>(out, value, numel, tid);              \
}                                                       \


#define FILL_OPS(NAME, T) \
FILL_OP(NAME, T)          \

FILL_OPS(u8, uchar)
FILL_OPS(u32, uint)
FILL_OPS(i64, long)
FILL_OPS(f16, half)
FILL_OPS(f32, float)

#if __METAL_VERSION__ >= 310
FILL_OPS(bf16, bfloat)
#endif
