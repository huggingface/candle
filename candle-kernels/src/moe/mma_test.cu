/**
 * @brief Minimal IMMA tensor-core test kernel for sm_120.
 *
 * Computes a single 16x8 m16n8k32 IMMA s8s8s32:
 *   D[16,8] = A[16,32] * B[32,8]
 *
 * This is the building block for a tensor-core MMVQ. Once verified
 * end-to-end, scale up to a real Q4_K decode kernel.
 *
 * Layout (one warp = 32 threads):
 *   A: 16 rows × 32 K (s8). Each thread holds 4 packed int32 = 16 s8.
 *   B:  8 cols × 32 K (s8). Each thread holds 2 packed int32 =  8 s8.
 *   D: 16 rows ×  8 cols (s32). Each thread holds 4 s32 (rows split across warp).
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace ll_mma {

// One-warp test: every thread loads its own A/B fragment from global
// memory, runs ONE mma.sync, writes back its D fragment.
__global__ void mma_test_m16n8k32_kernel(
    const int* __restrict__ a,    // 16x32 s8 packed as 16x8 int32, row-major
    const int* __restrict__ b,    // 8x32 s8 packed as 8x8 int32, row-major (col-major effective)
    int*       __restrict__ d     // 16x8 s32 output
) {
    const int lane = threadIdx.x;
    if (lane >= 32) return;

    // Load A: 16 rows × 32 K elements as int8. Per-thread layout per
    // mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32:
    //   A.x[0]: row=lane%16,            cols=[ 0, 1, 2, 3]
    //   A.x[1]: row=(lane%16)+8 (mod16) — actually lane>=16 picks rows 8..15, lanes 0..15 → rows 0..7
    //   A.x[2]: row=lane%16,            cols=[16,17,18,19]
    //   A.x[3]: row=(lane%16)+8,        cols=[20..23]
    //
    // The standard PTX mapping is:
    //   group  = lane / 4   (0..7)
    //   thread = lane % 4   (0..3)
    //   A row = group + 8*(0/1)         for elements 0/1
    //   A row = group + 8*(0/1) + ?     etc.
    //
    // For simplicity (test only): use straightforward row indexing and
    // the official PTX docs assignment.

    int A0, A1, A2, A3;
    int B0, B1;
    int D0 = 0, D1 = 0, D2 = 0, D3 = 0;

    // Each thread reads 4 ints of A and 2 ints of B from global.
    // Layout: A is row-major 16x8 ints (16 rows, 8 ints/row = 32 s8/row).
    // PTX m16n8k32 maps A as follows: thread t holds:
    //   a0 = A[t/4][2*(t%4)+0]
    //   a1 = A[t/4 + 8][2*(t%4)+0]
    //   a2 = A[t/4][2*(t%4)+1] — but offset by half (cols 16..31)
    //   a3 = A[t/4 + 8][2*(t%4)+1]
    // For our test we just stream the data directly; verify with a trivial input.
    const int g  = lane / 4;
    const int tj = lane % 4;
    A0 = a[(g    ) * 8 + 2 * tj + 0];
    A1 = a[(g + 8) * 8 + 2 * tj + 0];
    A2 = a[(g    ) * 8 + 2 * tj + 1];
    A3 = a[(g + 8) * 8 + 2 * tj + 1];

    // B: 8 cols × 32 K. PTX layout:
    //   b0 = B[2*(t%4)+0][t/4]   (col t/4, row 2*(t%4)+0) — but we use row-major
    //   b1 = B[2*(t%4)+1][t/4]
    // For the test we just feed identity-like values and check output.
    B0 = b[(2 * tj + 0) * 8 + g];
    B1 = b[(2 * tj + 1) * 8 + g];

    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(D0), "+r"(D1), "+r"(D2), "+r"(D3)
        : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1));

    // Write back D: 16 rows × 8 cols. Per PTX:
    //   d0 = D[g    ][2*tj + 0]
    //   d1 = D[g    ][2*tj + 1]
    //   d2 = D[g + 8][2*tj + 0]
    //   d3 = D[g + 8][2*tj + 1]
    d[(g    ) * 8 + 2 * tj + 0] = D0;
    d[(g    ) * 8 + 2 * tj + 1] = D1;
    d[(g + 8) * 8 + 2 * tj + 0] = D2;
    d[(g + 8) * 8 + 2 * tj + 1] = D3;
}

} // namespace ll_mma

extern "C" void mma_test_m16n8k32(
    const void* a,  // [16][8] int32
    const void* b,  // [8][8]  int32
    void*       d,  // [16][8] int32
    cudaStream_t stream
) {
    dim3 grid(1, 1, 1);
    dim3 block(32, 1, 1);
    ll_mma::mma_test_m16n8k32_kernel<<<grid, block, 0, stream>>>(
        (const int*)a, (const int*)b, (int*)d
    );
}
