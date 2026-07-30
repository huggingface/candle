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
| `hip_compat.h` | Force-included. 16-bit `atomicAdd`, `__dp4a`, `__vsubss4`, `*_sync` shuffle wrappers, `__syncwarp` |
| `cuda_fp16.h` | `#include <hip/hip_fp16.h>` |
| `cuda_bf16.h` | HIP bf16, plus the `__nv_bfloat16` alias and `__hmax_nan`/`__hmin_nan` |
| `cuda_fp8.h` | Maps `__nv_fp8_e4m3` onto HIP's OCP `__hip_fp8_e4m3` |
| `cuda.h` | Empty; the driver API is unused in device code |
| `cuda/std/limits` | Aliases `cuda::std` onto `std` for `reduce.cu` |

Sources are compiled with `-D__CUDA_ARCH__=800`, which enables the f16
(`>= 530`) and bf16 (`>= 800`) kernels. fp8 shares the `>= 800` guard, so those
kernels are built too. Of them only `const_set_f8_e4m3` is ever launched, and
it is a pure bit-move; every *arithmetic* F8E4M3 path in
`candle-core/src/rocm_backend` returns an error, so none of `binary.cu`'s fp8
ops — the only fp8 code that actually converts — is reachable. The header
comment in `cuda_fp8.h` spells this out.

The `*_sync` wrappers exist because HIP's own `__shfl_*_sync` and `__syncwarp`
take a **64-bit** lane mask and `static_assert` on CUDA's 32-bit `0xffffffff`.
The macros drop the mask and forward to the whole-wavefront form. `__syncwarp`
forwards to HIP's function rather than to `__builtin_amdgcn_wave_barrier()`:
HIP brackets the barrier with release/acquire wavefront fences, and the bare
builtin is a scheduling barrier with no memory ordering.

## Compilation and caching

Compilation happens at runtime, on first use of a module, so one binary runs on
any GPU architecture without a rebuild. Results are cached on disk:

```
~/.cache/candle-rocm/{arch}-{rocm_version}/
    {module}_{key}.hsaco
    {module}_{key}.lock      # advisory lock over one cache entry
    src-{headers_key}/       # staged sources and shim headers
```

`{key}` is the first 16 hex characters of a SHA-256 over **everything that can
change the emitted code object**: the source, every staged header (name and
contents), the hipcc flags, the target architecture, the full `hipcc --version`
string and this crate's version. Editing one kernel still invalidates only its
own module, but editing a shim header or bumping the toolchain now invalidates
what it should. Keying on the source alone — as an earlier version did — meant a
shim fix was silently ignored: the staged headers were rewritten, the cache
filename was not, and the stale code object was loaded instead.

The staging directory is named after the digest of the header set, so two builds
with different headers never overwrite each other's staged copies.

Writes go through a uniquely named temporary followed by a rename, and the two
compile intermediates (the hipcc bundle and the unbundler's output) are named
per process. On a cache miss the whole miss → compile → write sequence runs
under an advisory `fslock` file lock, with the cache re-checked after the lock
is acquired: concurrent compilers of the same module cannot persist a corrupt
object, and only one of them pays the compile.

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
| `CANDLE_ROCM_ARCH` | Target architecture, e.g. `gfx1101`. Otherwise read from the device |
| `CANDLE_ROCM_VERSION` | Overrides the version parsed from `hipcc --version` |
| `CANDLE_ROCM_CACHE_DIR` | Cache root, overriding `~/.cache/candle-rocm` |
| `CANDLE_ROCM_FORCE_RECOMPILE` | `1` skips the cache read (the write still happens) |
| `ROCM_PATH` | ROCm install root, default `/opt/rocm` |

Architecture detection fails loudly rather than guessing. A code object built
for the wrong architecture would otherwise surface much later as an opaque
"invalid device function". It comes from `hipGetDeviceProperties(ordinal)
.gcnArchName` for the ordinal the `RocmDevice` was opened with, so a
heterogeneous box — an iGPU beside a dGPU, or two different dGPUs — builds each
device's objects for its own target. `gcnArchName` also carries the target
features (`gfx942:sramecc+:xnack-`) that `--offload-arch` expects; the cache
directory uses a path-safe spelling of it, the compiler and the cache key get
the raw string.

When `CANDLE_ROCM_CACHE_DIR` is unset and `~/.cache` is missing or read-only —
containers, CI runners, service accounts — the cache falls back to a per-uid
directory under the system temp dir with a warning on stderr, rather than
failing to run at all.

## Requirements

ROCm 6.2 or newer, with `hipcc` and `clang-offload-bundler` available at
runtime.

`rocm-rs` supplies the HIP and rocBLAS bindings and is used with
`default-features = false`. Its default `gpu-sort` feature builds an amdgcn
kernel through a proc macro that shells out to a nested nightly `-Zbuild-std`
cargo invocation; candle never calls that sort, and the feature prevents the
crate from building on a stable toolchain.

Its `miopen` feature is off by default too. `candle-core`'s own `miopen`
feature turns it back on and swaps `conv1d`/`conv2d` onto MIOpen, the way
`cudnn` layers over `cuda`; the plain `rocm` build convolves with `im2col` plus
a rocBLAS GEMM and never links libMIOpen.

## Op coverage

The backend lives in `candle-core/src/rocm_backend`. Everything below is the
state as of this crate's first release; "shared kernel" means the op launches
the same `candle-kernels` code the CUDA backend launches.

| Area | Status | Notes |
|---|---|---|
| Unary / binary / affine / powf / elu | shared kernel | every dtype except `F8E4M3` |
| `cmp`, `where_cond` | shared kernel | comparison writes `u8`, as on CUDA |
| `to_dtype` | shared kernel | see the dtype table below |
| `reduce_op` (sum/min/max/argmin/argmax) | shared kernel | `fast_*`; argmin/argmax return `u32` |
| `gather`, `scatter`, `scatter_add`, `index_select`, `index_add` | shared kernel | ids must be `u8`/`u32`/`i64` |
| `arg_sort` | shared kernel | `asort_asc_*` / `asort_desc_*` |
| `matmul` | rocBLAS | `f32`/`f64`/`f16`/`bf16`, strided-batched |
| `conv1d`, `conv2d` | im2col + rocBLAS GEMM | MIOpen instead under `--features miopen` |
| `conv_transpose1d` | col2im + GEMM | falls back to the shared kernel when dilation/padding/output-padding is non-trivial |
| `conv_transpose2d` | shared kernel | always direct, as on CUDA |
| `avg_pool2d`, `max_pool2d` | shared kernel | |
| `upsample_nearest2d`, `upsample_bilinear2d` | shared kernel | |
| `upsample_nearest1d` | **errors** | no kernel exists; CUDA errors here too |
| `copy_strided_src`, `copy2d`, `const_set` | shared kernel + `hipMemcpy` | contiguous copies go through `hipMemcpy`/`hipMemcpy2D` |
| `rand_uniform`, `rand_normal` | rocRAND | generated in `f32`/`f64`, then cast |
| Quantized load / `dequantize` / `dequantize_f16` / `embedding` (`get_rows`) | shared kernel | all GGML block types |
| Quantized `fwd` (matmul) | **dequantize + GEMM** | correct but slow; see below |
| `candle-nn`: `softmax_last_dim`, `rms_norm`, `layer_norm`, `sigmoid` | shared kernel | `rocm_fwd` on the custom op |
| `candle-nn`: `rope`, `rope_i`, `rope_thd` | shared kernel | |

### dtypes

| dtype | Elementwise | `to_dtype` | reduce / indexing | matmul |
|---|---|---|---|---|
| `u8`, `u32`, `i64` | yes | yes | yes | no |
| `f16`, `bf16`, `f32`, `f64` | yes | yes | yes | yes |
| `i16`, `i32` | yes | **no** | **no** | no |
| `f8e4m3` | **no** | **no** | no | no |
| `f4`, `f6e2m3`, `f6e3m2`, `f8e8m0` | no | **no** | no | no |

`i16`/`i32` are a shared-source limitation, not a ROCm one: `cast.cu`,
`reduce.cu` and `indexing.cu` never instantiate their templates for those
types, so there is no kernel to launch. `f8e4m3` kernels *do* compile (they sit
behind the same `__CUDA_ARCH__ >= 800` guard as bf16); the arithmetic ones are
never launched — every elementwise `F8E4M3` arm returns an error, and the
`cast_*_f8_e4m3` entry points are named differently from what the cast macro
derives. What *does* work is moving fp8 bytes around: `const_set` launches
`const_set_f8_e4m3`, `copy2d` uses `hipMemcpy2D` and `copy_strided_src` reuses
`ucopy_u8`, none of which converts.

### Deliberately not implemented

- **Quantized fast paths.** `QRocmStorage::fwd` dequantizes to `f32` and runs a
  rocBLAS GEMM. The DMMV, MMVQ and MMQ kernels in `quantized.cu` are compiled
  but never launched, and the `fast_mmq` / `fast_mmvq` modules are CUDA-only.
  This is the single largest performance gap.
- **MoE.** `indexed_moe_forward` errors; there are no fused MoE kernels.
- **Flash attention.** `candle-flash-attn` is CUDA-only. `candle-nn`'s `sdpa`
  fast path is Metal-only on every backend.
- **`quantize_imatrix` / `quantize_imatrix_onto`.** Error out; plain `quantize`
  round-trips through the CPU quantizer.
- **`device_ptr`** on quantized storage. Errors; only the CUDA path exposes it.
- **fp8 arithmetic** as described above. Enabling it means validating the
  OCP-vs-NVIDIA E4M3 conversion on hardware first; only the bit-moves are safe
  on the strength of the encodings matching.

## Divergences from `cuda_backend`

Three places where this backend deliberately does *not* mirror the CUDA one,
because the CUDA behaviour looks like an upstream bug. Recorded here so they can
be reported rather than re-derived.

1. **`index_add` applies `start_offset` twice.** `cuda_backend/mod.rs`'s
   `index_add` allocates a fresh accumulator, copies the source into it, and
   then passes the *source's* layout as the accumulator's layout. The freshly
   allocated buffer is contiguous with offset 0, so a non-zero
   `src_l.start_offset()` is applied a second time. `metal_backend` does not do
   this, and `Tensor::scatter` builds a `Layout::contiguous` for exactly this
   case. The ROCm backend uses `Layout::contiguous(shape)`.
2. **Dequantize fallback reads a misaligned slice.** `quantized/cuda.rs` builds
   a `&[BlockQ*]` over a `Vec<u8>` with `std::slice::from_raw_parts`. `Vec<u8>`
   guarantees only byte alignment, so this is UB for every block type with a
   stricter alignment requirement. The ROCm fallback (`quantized/rocm.rs::deq`)
   copies each block out with `read_unaligned`, and
   `deq_reads_unaligned_buffers` in `quantized/rocm/tests.rs` covers it.
3. **CPU `conv_transpose1d` is wrong for batch > 1.** `cpu_backend/mod.rs:1415`,
   `MatMul::f`, collapses the batch loop when `b_skip == 0 && a_skip == m * k`
   without checking that the lhs rows are contiguous. The col2im path of
   `conv_transpose1d` produces exactly that shape with a strided lhs, so every
   batch after the first is computed from the wrong rows. The neighbouring
   collapse branch also derives `dst_rs` from the pre-collapse `n`, which makes
   it write overlapping rows. This one is a CPU bug, not a GPU one — it is
   recorded here because the ROCm conv tests are what surfaced it.

## Testing the shim

```
make rocm-shim-test
```

Compiles every shared module for the local GPU and runs
`src/hip_shim/shim_test.hip`, which exercises the only hand-written device code
in the project — the 16-bit `atomicAdd` CAS loops (aligned, unaligned, and
neighbour preservation), the `*_sync` shuffle wrappers, and both `__syncwarp`
spellings — on real hardware.

The cache keying itself is covered by unit tests:

```
cargo test --manifest-path candle-rocm-kernels/Cargo.toml
```

## Layout

```
src/
  lib.rs           Id / Module / the eleven module constants
  compile/
    mod.rs         KernelCache: hipcc invocation, disk and in-memory caches
    cache.rs       cache keys, cache locations, locking, atomic writes
    detect.rs      GPU architecture and ROCm toolchain detection
  error.rs         KernelError
  wrappers.rs      Send + Sync wrapper around rocm-rs' Module
  hip_shim/        the CUDA-to-HIP bridge; the only HIP-specific code here
```
