// Imported from https://github.com/ggerganov/llama.cpp/blob/master/ggml-metal.metal
#include <metal_stdlib>
using namespace metal;

#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }
#define SORT_ASC 1
#define SORT_DESC 0

// bitonic sort implementation following the CUDA kernels as reference
typedef void (argsort_t)(
        device const float * x,
        device     int32_t * dst,
        constant   int64_t & ncols,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]]);

template<int order, typename T>
METAL_FUNC void argsort(
        device const T        * x,
        device       uint32_t * dst,
        constant     int64_t & ncols,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]]) {
    // bitonic sort
    int col = tpitg[0];
    int row = tgpig[1];

    if (col >= ncols) return;

    device const T        * x_row   = x   + row * ncols;
    device       uint32_t * dst_row = dst + row * ncols;

    // initialize indices
    if (col < ncols) {
        dst_row[col] = col;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k = 2; k <= ncols; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (order == SORT_ASC ? x_row[dst_row[col]] > x_row[dst_row[ixj]] : x_row[dst_row[col]] < x_row[dst_row[ixj]]) {
                        SWAP(dst_row[col], dst_row[ixj]);
                    }
                } else {
                    if (order == SORT_ASC ? x_row[dst_row[col]] < x_row[dst_row[ixj]] : x_row[dst_row[col]] > x_row[dst_row[ixj]]) {
                        SWAP(dst_row[col], dst_row[ixj]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

#define ARGSORT(T, RUST_T) \
kernel void asort_asc_##RUST_T( \
    device const T        * x, \
    device       uint32_t * dst, \
    constant     int64_t & ncols, \
    uint3 tgpig[[threadgroup_position_in_grid]], \
    uint3 tpitg[[thread_position_in_threadgroup]] \
) {  \
    argsort<SORT_ASC, T>(x, dst, ncols, tgpig, tpitg); \
} \
kernel void asort_desc_##RUST_T( \
    device const T        * x, \
    device       uint32_t * dst, \
    constant     int64_t & ncols, \
    uint3 tgpig[[threadgroup_position_in_grid]], \
    uint3 tpitg[[thread_position_in_threadgroup]] \
) {  \
    argsort<SORT_DESC, T>(x, dst, ncols, tgpig, tpitg); \
} \

ARGSORT(float, f32)
ARGSORT(half, f16)
ARGSORT(uint8_t, u8)
ARGSORT(uint32_t, u32)

#if __METAL_VERSION__ >= 220
ARGSORT(int64_t, i64)
#endif
#if defined(__HAVE_BFLOAT__)
ARGSORT(bfloat, bf16)
#endif
