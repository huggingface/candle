// Reproduces the ROCm >= 7.14 header layering on any ROCm: there,
// amd_hip_fp16.h / amd_hip_bf16.h define non-template atomicAdd overloads for
// the 16-bit float types *before* hip_compat.h declares its template fallbacks.
// The shim must (a) still compile — the original non-template shims were a
// redefinition, https://github.com/huggingface/candle/pull/3801 — and (b) lose
// overload resolution to the header's version, since that one maps to hardware
// atomics.
//
// The sentinel overloads below stand in for the 7.14 header definitions: they
// deliberately add twice the value, so which overload won is observable from
// the result. Non-atomic on purpose — launched single-thread.
//
// This file is only compiled where the real headers do NOT define the
// overloads themselves (the Makefile greps for that); on a >= 7.14 toolchain
// the sentinels would clash with the headers exactly as the original shim did,
// and there shim_test.hip already exercises the real coexistence.
#include <cstdio>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

__device__ __half atomicAdd(__half *address, __half val) {
    *address = __float2half(__half2float(*address) + 2.0f * __half2float(val));
    return *address;
}

__device__ __hip_bfloat16 atomicAdd(__hip_bfloat16 *address, __hip_bfloat16 val) {
    *address = __float2bfloat16(__bfloat162float(*address) + 2.0f * __bfloat162float(val));
    return *address;
}

#include "hip_compat.h"

__global__ void add_one_half(__half *acc) { atomicAdd(acc, __float2half(1.0f)); }
__global__ void add_one_bf16(__hip_bfloat16 *acc) { atomicAdd(acc, __float2bfloat16(1.0f)); }

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
        printf("%-36s %s\n", name, (cond) ? "PASS" : "*** FAIL ***");          \
        if (!(cond)) failures++;                                               \
    } while (0)

int main() {
    int failures = 0;

    __half *d_h;
    HIP_CHECK(hipMalloc(&d_h, sizeof(__half)));
    HIP_CHECK(hipMemset(d_h, 0, sizeof(__half)));
    hipLaunchKernelGGL(add_one_half, dim3(1), dim3(1), 0, 0, d_h);
    __half h_h;
    HIP_CHECK(hipMemcpy(&h_h, d_h, sizeof(__half), hipMemcpyDeviceToHost));
    // 2.0 -> the sentinel non-template won; 1.0 -> the shim template hijacked it.
    CHECK(__half2float(h_h) == 2.0f, "non-template atomicAdd(__half*) wins");

    __hip_bfloat16 *d_b;
    HIP_CHECK(hipMalloc(&d_b, sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMemset(d_b, 0, sizeof(__hip_bfloat16)));
    hipLaunchKernelGGL(add_one_bf16, dim3(1), dim3(1), 0, 0, d_b);
    __hip_bfloat16 h_b;
    HIP_CHECK(hipMemcpy(&h_b, d_b, sizeof(__hip_bfloat16), hipMemcpyDeviceToHost));
    CHECK(__bfloat162float(h_b) == 2.0f, "non-template atomicAdd(bf16*) wins");

    printf("\n%d failure(s)\n", failures);
    return failures != 0;
}
