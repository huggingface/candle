// Shadows CUDA's <cuda_fp16.h> for the shared candle-kernels sources.
//
// This directory is placed ahead of candle-kernels/src on the include path, so
// `#include "cuda_fp16.h"` from compatibility.cuh resolves here instead of to a
// CUDA toolkit that is not installed. HIP's half type is layout- and
// name-compatible with CUDA's `__half`, so nothing else is needed.
#pragma once

#include <hip/hip_fp16.h>
