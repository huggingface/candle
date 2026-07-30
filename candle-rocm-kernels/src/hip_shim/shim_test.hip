// Validates the two pieces of hand-written logic in hip_compat.h against real
// gfx1101 hardware: the 16-bit atomicAdd CAS loops and the *_sync shuffle
// macros. Everything else in the shim is a typedef or a header alias.
#include <cstdio>
#include <cmath>
#include "hip_compat.h"

// --- warp shuffle: butterfly reduction over an explicit width of 32 ---------
__global__ void shfl_sum(const float *in, float *out) {
    float v = in[threadIdx.x];
    for (int mask = 16; mask > 0; mask >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, mask, 32);
    }
    if (threadIdx.x % 32 == 0) out[threadIdx.x / 32] = v;
}

// --- __syncwarp: HIP's own definition must not be shadowed -----------------
// The shim deliberately does not define __syncwarp. A `#define __syncwarp()`
// would replace HIP's fenced version with a scheduling-only barrier *and* make
// the masked spelling a compile error, so using both spellings here is a
// compile-time regression guard.
__global__ void syncwarp_both_spellings(int *out) {
    __shared__ int scratch[32];
    scratch[threadIdx.x] = (int)threadIdx.x;
    __syncwarp();
    int v = scratch[threadIdx.x ^ 1];
    __syncwarp(0xffffffff);
    out[threadIdx.x] = v;
}

// --- 16-bit atomicAdd: every thread adds 1.0 to one accumulator ------------
__global__ void atomic_add_half(__half *acc) {
    atomicAdd(acc, __float2half(1.0f));
}

__global__ void atomic_add_bf16(__hip_bfloat16 *acc) {
    atomicAdd(acc, __float2bfloat16(1.0f));
}

// Unaligned case: accumulator at an odd 16-bit slot, to exercise the
// upper-half branch of the CAS loop.
__global__ void atomic_add_half_pair(__half *accs) {
    atomicAdd(&accs[1], __float2half(1.0f));
}

#define HIP_CHECK(expr)                                                        \
    do {                                                                       \
        hipError_t err_ = (expr);                                              \
        if (err_ != hipSuccess) {                                              \
            printf("%s:%d: %s -> %s\n", __FILE__, __LINE__, #expr,             \
                   hipGetErrorString(err_));                                   \
            return 1;                                                          \
        }                                                                      \
    } while (0)

#define CHECK(cond, name)                                                      \
    do {                                                                       \
        printf("%-28s %s\n", name, (cond) ? "PASS" : "*** FAIL ***");          \
        if (!(cond)) failures++;                                               \
    } while (0)

int main() {
    int failures = 0;
    const int N = 128; // 4 waves of 32

    float h_in[N], *d_in, *d_out;
    for (int i = 0; i < N; ++i) h_in[i] = (float)(i + 1);
    HIP_CHECK(hipMalloc(&d_in, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, 4 * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_in, h_in, N * sizeof(float), hipMemcpyHostToDevice));
    hipLaunchKernelGGL(shfl_sum, dim3(1), dim3(N), 0, 0, d_in, d_out);
    float h_out[4];
    HIP_CHECK(hipMemcpy(h_out, d_out, 4 * sizeof(float), hipMemcpyDeviceToHost));

    // Each group of 32 consecutive values must sum within its own group.
    for (int g = 0; g < 4; ++g) {
        float want = 0;
        for (int i = g * 32; i < (g + 1) * 32; ++i) want += h_in[i];
        char name[64];
        snprintf(name, sizeof(name), "shfl_xor_sync group %d", g);
        CHECK(fabsf(h_out[g] - want) < 1e-3f, name);
    }

    // One wave exchanging through shared memory across both __syncwarp forms.
    int *d_sw;
    HIP_CHECK(hipMalloc(&d_sw, 32 * sizeof(int)));
    hipLaunchKernelGGL(syncwarp_both_spellings, dim3(1), dim3(32), 0, 0, d_sw);
    int h_sw[32];
    HIP_CHECK(hipMemcpy(h_sw, d_sw, 32 * sizeof(int), hipMemcpyDeviceToHost));
    int sw_ok = 1;
    for (int i = 0; i < 32; ++i) sw_ok &= (h_sw[i] == (i ^ 1));
    CHECK(sw_ok, "__syncwarp both spellings");

    // 1024 threads each adding 1.0 -> 1024.0, exactly representable in both.
    __half *d_h;
    HIP_CHECK(hipMalloc(&d_h, 2 * sizeof(__half)));
    HIP_CHECK(hipMemset(d_h, 0, 2 * sizeof(__half)));
    hipLaunchKernelGGL(atomic_add_half, dim3(4), dim3(256), 0, 0, d_h);
    __half h_h[2];
    HIP_CHECK(hipMemcpy(h_h, d_h, 2 * sizeof(__half), hipMemcpyDeviceToHost));
    CHECK(fabsf(__half2float(h_h[0]) - 1024.0f) < 0.5f, "atomicAdd(__half*) aligned");

    HIP_CHECK(hipMemset(d_h, 0, 2 * sizeof(__half)));
    hipLaunchKernelGGL(atomic_add_half_pair, dim3(4), dim3(256), 0, 0, d_h);
    HIP_CHECK(hipMemcpy(h_h, d_h, 2 * sizeof(__half), hipMemcpyDeviceToHost));
    CHECK(fabsf(__half2float(h_h[1]) - 1024.0f) < 0.5f, "atomicAdd(__half*) unaligned");
    CHECK(fabsf(__half2float(h_h[0]) - 0.0f) < 0.5f, "atomicAdd(__half*) no neighbour clobber");

    // bf16 has 8 significand bits, so ulp at 256 is 2 and `256 + 1` rounds back
    // to 256 -- accumulating ones saturates there regardless of atomicity. Stay
    // under that so this tests the CAS loop rather than bf16 rounding.
    __hip_bfloat16 *d_b;
    HIP_CHECK(hipMalloc(&d_b, sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMemset(d_b, 0, sizeof(__hip_bfloat16)));
    hipLaunchKernelGGL(atomic_add_bf16, dim3(2), dim3(64), 0, 0, d_b);
    __hip_bfloat16 h_b;
    HIP_CHECK(hipMemcpy(&h_b, d_b, sizeof(__hip_bfloat16), hipMemcpyDeviceToHost));
    CHECK(fabsf(__bfloat162float(h_b) - 128.0f) < 0.5f, "atomicAdd(__hip_bfloat16*)");

    printf("\n%d failure(s)\n", failures);
    return failures != 0;
}
