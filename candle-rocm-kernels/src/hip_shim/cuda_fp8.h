// Shadows CUDA's <cuda_fp8.h> for the shared candle-kernels sources.
//
// The fp8 kernels sit behind `#if __CUDA_ARCH__ >= 800`, the same guard as the
// bfloat16 helpers, so they cannot be excluded without also losing bf16. They
// are therefore all compiled, but only some are reachable:
//
//   * Launched today: `const_set_f8_e4m3` (fill.cu), from `const_set` in
//     candle-core/src/rocm_backend/mod.rs. `fill_f8_e4m3` and `copy2d_f8_e4m3`
//     are emitted too but currently unused — the backend's `copy2d` moves fp8
//     with hipMemcpy2D and `copy_strided_src` reuses `ucopy_u8`. All of these
//     are pure bit-moves: they store the 8-bit payload unchanged and perform no
//     conversion, so the encoding never enters into their result.
//   * Compiled but unreachable: binary.cu's fp8 arithmetic and comparison ops
//     (badd/bdiv/bmul/bsub/bmaximum/bminimum, eq/ne/lt/le/gt/ge). Every
//     elementwise F8E4M3 arm in the ROCm backend bails before reaching them.
//     Those *do* convert, through `__nv_cvt_fp8_to_halfraw` and the
//     `__nv_fp8_e4m3(float)` constructor, so anyone enabling them must first
//     validate the conversion on real hardware.
//
// The typedefs below are nevertheless encoding-exact, which is what makes the
// bit-moves safe to keep: HIP's __hip_fp8_e4m3 is the OCP E4M3 encoding, the
// same one NVIDIA uses for __nv_fp8_e4m3. __hip_fp8_e4m3_fnuz is the AMD-only
// variant with a different exponent bias and no infinities — it must never be
// substituted here. gfx1101 has no fp8 hardware, so the conversions above are
// emulated in software.
#pragma once

#include <hip/hip_fp8.h>

typedef __hip_fp8_e4m3 __nv_fp8_e4m3;
typedef __hip_fp8_e5m2 __nv_fp8_e5m2;

#define __NV_E4M3 __HIP_E4M3
#define __NV_E5M2 __HIP_E5M2

#define __nv_cvt_fp8_to_halfraw __hip_cvt_fp8_to_halfraw
#define __nv_cvt_float_to_fp8 __hip_cvt_float_to_fp8
