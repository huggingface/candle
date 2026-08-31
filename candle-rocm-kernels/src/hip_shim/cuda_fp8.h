// Shadows CUDA's <cuda_fp8.h> for the shared candle-kernels sources.
//
// The typedefs are encoding-exact, which is what makes every fp8 kernel usable
// on ROCm: HIP's __hip_fp8_e4m3 is the OCP E4M3 encoding, the same one NVIDIA
// uses for __nv_fp8_e4m3. __hip_fp8_e4m3_fnuz is the AMD-only variant with a
// different exponent bias and no infinities — it must never be substituted
// here.
//
// The arithmetic kernels convert through `__nv_cvt_fp8_to_halfraw` and the
// `__nv_fp8_e4m3(float)` constructor. No AMD part before gfx942/gfx12xx has fp8
// hardware, so on gfx1101 both are software: round-to-nearest-even on the way
// in, saturating to +/-448 rather than overflowing to infinity. Validated
// against the CPU backend by the `f8e4m3_*` tests in
// candle-core/src/rocm_backend/tests_f8e4m3.rs.
//
// One divergence from CUDA is known and unavoidable here: HIP encodes a float
// infinity as NaN (0x7f), while NVIDIA's SATFINITE conversion clamps it to the
// maximum finite value. Only reachable by feeding an already-infinite f32 to a
// conversion, which fp8's own saturation otherwise prevents.
#pragma once

#include <hip/hip_fp8.h>

typedef __hip_fp8_e4m3 __nv_fp8_e4m3;
typedef __hip_fp8_e5m2 __nv_fp8_e5m2;

#define __NV_E4M3 __HIP_E4M3
#define __NV_E5M2 __HIP_E5M2

#define __nv_cvt_fp8_to_halfraw __hip_cvt_fp8_to_halfraw
#define __nv_cvt_float_to_fp8 __hip_cvt_float_to_fp8
