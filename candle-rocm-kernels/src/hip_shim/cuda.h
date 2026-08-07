// Shadows CUDA's <cuda.h> for the shared candle-kernels sources.
//
// The driver-API surface is not used by the device code we compile; the HIP
// runtime header is force-included ahead of everything by hip_compat.h.
#pragma once
