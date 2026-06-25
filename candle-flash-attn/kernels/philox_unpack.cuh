// candle: stub replacement for the original PyTorch-dependent header.
//
// Upstream Tri Dao's `philox_unpack.cuh` includes
// <ATen/cuda/detail/UnpackRaw.cuh> to provide
// `at::cuda::philox::unpack(at::PhiloxCudaState)` for the dropout
// path in `flash_fwd_kernel.h`. candle-flash-attn doesn't link
// against PyTorch, and inference-only callers don't exercise the
// dropout codepath, so we stub `unpack()` to return a dummy
// (seed, offset) pair. The result is computed-but-unused by the
// kernel when `Is_dropout=false` (the caller path through
// `LOCAL_SWITCH(... Is_dropout && !Is_softcap, ...)`); the
// ostensible Dropout object built from these dummy values is
// dead code under that compile-time branch.
//
// `at::PhiloxCudaState` is also stubbed as an empty struct so
// `flash.h`'s `Flash_fwd_params::philox_args` field can stay in
// the layout — matching the field count Tri Dao expects without
// dragging PyTorch in.

#pragma once

#include <cstdint>
#include <tuple>

namespace at {
    struct PhiloxCudaState {};
    namespace cuda { namespace philox {
        inline __host__ __device__ std::tuple<uint64_t, uint64_t> unpack(PhiloxCudaState const&) {
            return {0ull, 0ull};
        }
    }}
}
