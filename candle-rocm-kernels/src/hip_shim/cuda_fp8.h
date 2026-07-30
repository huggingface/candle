// Shadows CUDA's <cuda_fp8.h> for the shared candle-kernels sources.
//
// The fp8 kernels sit behind `#if __CUDA_ARCH__ >= 800`, the same guard as the
// bfloat16 helpers, so they cannot be excluded without also losing bf16. They
// are therefore compiled, but the ROCm backend never launches them: every
// F8E4M3 path in candle-core/src/rocm_backend bails.
//
// HIP's __hip_fp8_e4m3 is the OCP E4M3 encoding, which is the same one NVIDIA
// uses for __nv_fp8_e4m3 — unlike __hip_fp8_e4m3_fnuz, which is the AMD-only
// variant and must not be used here. gfx1101 has no fp8 hardware, so the
// conversions are emulated in software.
#pragma once

#include <hip/hip_fp8.h>

typedef __hip_fp8_e4m3 __nv_fp8_e4m3;
typedef __hip_fp8_e5m2 __nv_fp8_e5m2;

#define __NV_E4M3 __HIP_E4M3
#define __NV_E5M2 __HIP_E5M2

#define __nv_cvt_fp8_to_halfraw __hip_cvt_fp8_to_halfraw
#define __nv_cvt_float_to_fp8 __hip_cvt_float_to_fp8
