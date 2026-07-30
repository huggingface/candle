// Force-included (via `-include`) ahead of every shared candle-kernels source
// when compiling for ROCm/HIP.
//
// The goal of this directory is that candle-kernels/src/*.cu stays a single
// source of truth shared with the CUDA backend: not one line of it is edited
// for ROCm. Anything CUDA-specific is bridged here, either by shadowing a CUDA
// header name (cuda_fp16.h, cuda_bf16.h, cuda_fp8.h, cuda.h, cuda/std/limits)
// or by supplying a missing intrinsic below.
#pragma once

// HIP's own headers must be parsed *before* the function-like macros at the
// bottom of this file are defined, otherwise those macros rewrite HIP's own
// declarations of the same names and the headers fail to compile.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

// ---------------------------------------------------------------------------
// Atomics
// ---------------------------------------------------------------------------
// HIP has atomicAdd for float/double/int, but not for the 16-bit float types.
// reduce.cu's SUM_OP instantiates sum_f16 and sum_bf16, both of which need one.
// Implemented with a 32-bit CAS over the containing aligned word, which is the
// same trick CUDA's own pre-sm_70 fallback uses.
#define CANDLE_HIP_ATOMIC_ADD_16(TYPE, TO_BITS, FROM_BITS)                     \
    __device__ __forceinline__ TYPE atomicAdd(TYPE *address, TYPE val) {       \
        unsigned int *base =                                                   \
            (unsigned int *)((char *)address - ((size_t)address & 2));         \
        bool upper = ((size_t)address & 2) != 0;                               \
        unsigned int old = *base, assumed;                                     \
        do {                                                                   \
            assumed = old;                                                     \
            unsigned short cur =                                               \
                (unsigned short)(upper ? (assumed >> 16) : (assumed & 0xffff));\
            unsigned short sum = TO_BITS(FROM_BITS(cur) + val);                \
            unsigned int next = upper ? ((assumed & 0x0000ffffu) |             \
                                         ((unsigned int)sum << 16))            \
                                      : ((assumed & 0xffff0000u) | sum);       \
            old = atomicCAS(base, assumed, next);                              \
        } while (assumed != old);                                              \
        unsigned short res =                                                   \
            (unsigned short)(upper ? (old >> 16) : (old & 0xffff));            \
        return FROM_BITS(res);                                                 \
    }

__device__ __forceinline__ unsigned short __candle_bf16_as_ushort(__hip_bfloat16 h) {
    return *reinterpret_cast<unsigned short *>(&h);
}

__device__ __forceinline__ __hip_bfloat16 __candle_ushort_as_bf16(unsigned short u) {
    return *reinterpret_cast<__hip_bfloat16 *>(&u);
}

CANDLE_HIP_ATOMIC_ADD_16(__half, __half_as_ushort, __ushort_as_half)
CANDLE_HIP_ATOMIC_ADD_16(__hip_bfloat16, __candle_bf16_as_ushort, __candle_ushort_as_bf16)

#undef CANDLE_HIP_ATOMIC_ADD_16

// ---------------------------------------------------------------------------
// Integer SIMD intrinsics (quantized.cu)
// ---------------------------------------------------------------------------
// Portable expansions. gfx11+ can do the dot product in one instruction via
// __builtin_amdgcn_sudot4, but correctness comes first; revisit once the
// quantized kernels are validated.
__device__ __forceinline__ int __dp4a(int a, int b, int c) {
    int res = c;
    for (int i = 0; i < 4; ++i) {
        const int shift = i * 8;
        res += (int)(signed char)((a >> shift) & 0xff) *
               (int)(signed char)((b >> shift) & 0xff);
    }
    return res;
}

// Per-byte saturating signed subtract.
__device__ __forceinline__ unsigned int __vsubss4(unsigned int a, unsigned int b) {
    unsigned int res = 0;
    for (int i = 0; i < 4; ++i) {
        const int shift = i * 8;
        int d = (int)(signed char)((a >> shift) & 0xff) -
                (int)(signed char)((b >> shift) & 0xff);
        d = d > 127 ? 127 : (d < -128 ? -128 : d);
        res |= ((unsigned int)(d & 0xff)) << shift;
    }
    return res;
}

// ---------------------------------------------------------------------------
// Warp shuffles
// ---------------------------------------------------------------------------
// HIP does declare __shfl_xor_sync, but its mask parameter is a 64-bit lane
// mask and it static_asserts when handed CUDA's 32-bit 0xffffffff. The kernels
// only ever pass a full mask, and HIP's unsuffixed __shfl_xor already has
// whole-wavefront semantics, so drop the mask.
//
// Every candle call site passes an explicit width of 32 (WARP_SIZE), which HIP
// honours as a sub-wavefront shuffle. That keeps the block-reduction indexing
// correct on both wave32 (RDNA) and wave64 (CDNA) hardware.
#define __shfl_xor_sync(mask, var, lane_mask, width) __shfl_xor(var, lane_mask, width)
#define __shfl_sync(mask, var, src_lane, width) __shfl(var, src_lane, width)
#define __shfl_up_sync(mask, var, delta, width) __shfl_up(var, delta, width)
#define __shfl_down_sync(mask, var, delta, width) __shfl_down(var, delta, width)
#define __syncwarp() __builtin_amdgcn_wave_barrier()
