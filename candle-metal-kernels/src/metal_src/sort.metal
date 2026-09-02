// Imported from https://github.com/ggerganov/llama.cpp/blob/master/ggml-metal.metal
#include <metal_stdlib>
using namespace metal;

#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }
#define SORT_ASC 1
#define SORT_DESC 0

template<int order, typename T>
METAL_FUNC void argsort(
        device const T        * x,
        device       uint32_t * dst,
        constant     int64_t & ncols,
        constant     int64_t & ncols_pad,
        threadgroup uint32_t  * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]]) {
    int col = tpitg[0];
    int row = tgpig[1];

    if (col >= ncols_pad) return;

    device const T        * x_row   = x + row * ncols;
    threadgroup uint32_t  * dst_row = shared_values;

    // initialize indices
    dst_row[col] = col;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (dst_row[col] >= ncols ||
                        (dst_row[ixj] < ncols && (order == SORT_ASC ?
                            x_row[dst_row[col]] > x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] < x_row[dst_row[ixj]]))
                    ) {
                        SWAP(dst_row[col], dst_row[ixj]);
                    }
                } else {
                    if (dst_row[ixj] >= ncols ||
                        (dst_row[col] < ncols && (order == SORT_ASC ?
                            x_row[dst_row[col]] < x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] > x_row[dst_row[ixj]]))
                    ) {
                        SWAP(dst_row[col], dst_row[ixj]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // copy the result to dst without the padding
    if (col < ncols) {
        dst[row * ncols + col] = dst_row[col];
    }
}

#define ARGSORT(T, RUST_T) \
kernel void asort_asc_##RUST_T( \
    device const T        * x, \
    device       uint32_t * dst, \
    constant     int64_t & ncols, \
    constant     int64_t & ncols_pad, \
    threadgroup uint32_t  * shared_values [[threadgroup(0)]], \
    uint3 tgpig[[threadgroup_position_in_grid]], \
    uint3 tpitg[[thread_position_in_threadgroup]] \
) {  \
    argsort<SORT_ASC, T>(x, dst, ncols, ncols_pad, shared_values, tgpig, tpitg); \
} \
kernel void asort_desc_##RUST_T( \
    device const T        * x, \
    device       uint32_t * dst, \
    constant     int64_t & ncols, \
    constant     int64_t & ncols_pad, \
    threadgroup uint32_t  * shared_values [[threadgroup(0)]], \
    uint3 tgpig[[threadgroup_position_in_grid]], \
    uint3 tpitg[[thread_position_in_threadgroup]] \
) {  \
    argsort<SORT_DESC, T>(x, dst, ncols, ncols_pad, shared_values, tgpig, tpitg); \
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
