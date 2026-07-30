# candle-rocm-kernels

ROCm/HIP kernel support for the Candle deep learning framework.

## Overview

This crate does not ship kernels of its own. It embeds the **shared**
`candle-kernels/src/*.cu` sources — the very same ones the CUDA backend uses —
and compiles them with `hipcc`.

Everything CUDA-specific is bridged by a small shim in `src/hip_shim/`, which
works by *shadowing* the CUDA header names the sources `#include`. Because the
shim sits on its own include path ahead of `candle-kernels/src`, not a single
line of the shared sources is edited, and nothing here can affect the CUDA
build.

Keeping one set of sources matters. An earlier version of this crate
hand-wrote HIP kernels, and they silently drifted from their CUDA
counterparts: `utan` where candle calls `utanh` (breaking `tanh` on every
dtype), a `bminimum` that dropped NaN handling, a `UNARY_OP` missing its
in-place path, and a `BINARY_OP` missing the out-type parameter that
comparison operators need.

## What the shim provides

| Shim header | Purpose |
|---|---|
| `hip_compat.h` | Force-included. 16-bit `atomicAdd`, `__dp4a`, `__vsubss4`, `*_sync` shuffle wrappers |
| `cuda_fp16.h` | `#include <hip/hip_fp16.h>` |
| `cuda_bf16.h` | HIP bf16, plus the `__nv_bfloat16` alias and `__hmax_nan`/`__hmin_nan` |
| `cuda_fp8.h` | Maps `__nv_fp8_e4m3` onto HIP's OCP `__hip_fp8_e4m3` |
| `cuda.h` | Empty; the driver API is unused in device code |
| `cuda/std/limits` | Aliases `cuda::std` onto `std` for `reduce.cu` |

Sources are compiled with `-D__CUDA_ARCH__=800`, which enables the f16
(`>= 530`) and bf16 (`>= 800`) kernels. fp8 shares the `>= 800` guard, so those
kernels are built too, but the backend never launches them — every F8E4M3 path
in `candle-core/src/rocm_backend` returns an error.

## Compilation and caching

Compilation happens at runtime, on first use of a module, so one binary runs on
any GPU architecture without a rebuild. Results are cached on disk:

```
~/.cache/candle-rocm/{arch}-{rocm_version}/
    {module}_{source_hash}.hsaco
    src/                     # staged sources and shim headers
```

`{source_hash}` is the first 16 hex characters of the source's SHA-256, so
editing one kernel invalidates only its module. Cache writes go through a
uniquely named temporary followed by a rename, so concurrent processes cannot
observe a half-written file.

The pipeline is `hipcc --genco` followed by `clang-offload-bundler --unbundle`,
which yields the single-architecture code object that `hipModuleLoadData`
expects. The bundler is taken from `$ROCM_PATH` (default `/opt/rocm`) rather
than whichever `clang` happens to come first on `PATH`, since a mismatched LLVM
version cannot read the bundle.

Expect roughly 3–5 s per module on first use. `quantized.cu` is the outlier at
about 70 s, being 4,845 lines.

## Environment variables

| Variable | Effect |
|---|---|
| `CANDLE_ROCM_ARCH` | Target architecture, e.g. `gfx1101`. Otherwise detected with `rocm_agent_enumerator` |
| `CANDLE_ROCM_VERSION` | Overrides the version parsed from `hipcc --version` |
| `ROCM_PATH` | ROCm install root, default `/opt/rocm` |

Architecture detection fails loudly rather than guessing. A code object built
for the wrong architecture would otherwise surface much later as an opaque
"invalid device function".

## Requirements

ROCm 6.2 or newer, with `hipcc` and `clang-offload-bundler` available at
runtime.

`rocm-rs` supplies the HIP, rocBLAS and MIOpen bindings and is used with
`default-features = false`. Its default `gpu-sort` feature builds an amdgcn
kernel through a proc macro that shells out to a nested nightly `-Zbuild-std`
cargo invocation; candle never calls that sort, and the feature prevents the
crate from building on a stable toolchain.

## Testing the shim

```
make rocm-shim-test
```

Compiles every shared module for the local GPU and runs
`src/hip_shim/shim_test.hip`, which exercises the only hand-written device code
in the project — the 16-bit `atomicAdd` CAS loops (aligned, unaligned, and
neighbour preservation) and the `*_sync` shuffle wrappers — on real hardware.

## Layout

```
src/
  lib.rs           Id / Module / the eleven module constants
  compile.rs       runtime hipcc invocation, disk and in-memory caches
  error.rs         KernelError
  wrappers.rs      Send + Sync wrapper around rocm-rs' Module
  hip_shim/        the CUDA-to-HIP bridge; the only HIP-specific code here
```
