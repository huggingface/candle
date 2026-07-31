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

CANDLE_HIP_ATOMIC_ADD_16(__half, __half_as_ushort, __ushort_as_half)
CANDLE_HIP_ATOMIC_ADD_16(__hip_bfloat16, __bfloat16_as_ushort, __ushort_as_bfloat16)

#undef CANDLE_HIP_ATOMIC_ADD_16

// ---------------------------------------------------------------------------
// Integer SIMD intrinsics (quantized.cu)
// ---------------------------------------------------------------------------
// gfx11/gfx12 (RDNA3/RDNA4) have v_dot4_i32_iu8, which __builtin_amdgcn_sudot4
// lowers to in a single instruction; the leading/third `true` select the signed
// interpretation of each operand and the trailing `false` disables clamping, so
// it computes exactly what the portable expansion below does. The MMVQ kernels
// are dp4a-bound, so this is the difference between four multiply-adds and one
// instruction per inner step.
//
// The gate is the architecture macro rather than __has_builtin: clang declares
// the builtin for every AMDGPU target and rejects it only during semantic
// analysis ("needs target feature dot8-insts"), so __has_builtin(...) is 1 on
// gfx1030 and gfx90a as well and would break their builds. CDNA and gfx10 carry
// the related __builtin_amdgcn_sdot4 (dot1-insts); it is deliberately left
// unwired because no such part was available to validate it on, and a wrong
// dot product here is a silent wrong answer in every quantized matmul.
//
// `make rocm-shim-test` compares this against the reference loop on the GPU.
#if defined(__GFX11__) || defined(__GFX12__)
__device__ __forceinline__ int __dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}
#else
__device__ __forceinline__ int __dp4a(int a, int b, int c) {
    int res = c;
    for (int i = 0; i < 4; ++i) {
        const int shift = i * 8;
        res += (int)(signed char)((a >> shift) & 0xff) *
               (int)(signed char)((b >> shift) & 0xff);
    }
    return res;
}
#endif

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

// __syncwarp gets the same treatment, for the same reason. HIP does define it
// (amd_warp_sync_functions.h) and its unmasked form is the one to use: it
// brackets the wave barrier with release/acquire wavefront fences, which a bare
// __builtin_amdgcn_wave_barrier() -- a scheduling barrier with no memory
// ordering -- does not. The forwarding helper is defined before the macro so it
// binds to that HIP function; the macro must not expand to anything weaker.
//
// The macro is variadic so both CUDA spellings work: HIP's masked overload
// static_asserts on a 32-bit mask exactly as __shfl_xor_sync does, so
// __syncwarp(0xffffffff) fails to compile without it.
__device__ __forceinline__ void __candle_syncwarp() { __syncwarp(); }
#define __syncwarp(...) __candle_syncwarp()
