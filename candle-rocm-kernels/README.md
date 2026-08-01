# candle-rocm-kernels

ROCm/HIP kernel support for the Candle deep learning framework.

## Origin

This backend began as [@airpods69](https://github.com/airpods69)'s
[#3424, "Initial implementation for ROCm support"](https://github.com/huggingface/candle/pull/3424),
which established everything the work below rests on: the decision to bind
ROCm through [`rocm-rs`](https://crates.io/crates/rocm-rs), the `rocm` feature
and its wiring through `Device`, `Storage` and the backend traits, the shape of
`rocm_backend` as a mirror of `cuda_backend`, and a working device, allocator,
RNG and matmul. That PR also picked up the thread from
[#3186](https://github.com/huggingface/candle/pull/3186) and the ROCm request
issue it referenced.

Its open checklist — convolutions, training a model, and "update the API to
make more sense" — is what this work continued. The one place it diverges is
the kernels: #3424 hand-wrote `.hip` sources, and this crate compiles the
shared `.cu` ones instead, for the reasons in the Overview.

Development and testing happened on a Radeon RX 7800 XT (gfx1101, RDNA3) under
ROCm 7.2.4 — the same architecture as #3424's RX 7700 XT, one ROCm major
version newer.

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

Sources are compiled with `-D__CUDA_ARCH__=890`, the lowest value at which
every dtype candle supports is instantiated. f16 needs `>= 530` and bf16
`>= 800`; `binary.cu`, `cast.cu` and `fill.cu` gate their fp8 at 800, but the
F8E4M3 kernels in `unary.cu`, `affine.cu`, `ternary.cu` and `indexing.cu` sit
behind `>= 890` — Ada, where NVIDIA first shipped fp8 hardware. Nothing else in
the shared sources is gated between 801 and 890, so 890 adds fp8 entry points
and changes no other kernel. They convert through `f32` in software, which is
what the CUDA sources ask for regardless; RDNA3 has no fp8 hardware. The value
is defined once, in `compile/cache.rs::COMPILE_FLAGS`, and `make
rocm-shim-test` greps it back out rather than repeating it.

One flag depends on the target: `-DRDNA2` for gfx103x and `-DRDNA3` for
gfx11xx/gfx12xx, which selects the MMQ tile geometry `quantized.cu` compiles
(`compile/cache.rs::arch_flag` over `compile/detect.rs::rdna_define`). CDNA,
Vega and RDNA1 get neither define and keep the kernel's default Ampere set —
RDNA1's own tiles are a third geometry no machine here can test. The host has
to launch the geometry it compiled, since the grid comes from `mmq_x`/`mmq_y`
and the block from `nwarps`, so the choice is readable back through
`KernelCache::mmq_tiles` and `RocmDevice::mmq_tiles`.

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
    {module}_{key}.lock             # advisory lock over one cache entry
    custom-{module}_{key}.hsaco     # a downstream crate's module; see below
    src-{headers_key}/              # staged sources and shim headers
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

## Kernels of your own

A crate that depends on `candle-core` with `features = ["rocm"]` can put its own
`.cu` source through the same pipeline, without vendoring the shim or patching
candle:

```rust
use candle_core::rocm_backend::{launch_config, RocmDevice};

const GDN: &str = include_str!("gdn.cu");   // or any &str, generated or not

let func = dev.get_or_load_custom_func("gdn_fwd_f32", "crane_gdn", GDN)?;
let (grid, block) = launch_config(&dev, n);
unsafe { func.launch(grid, block, 0, Some(dev.stream()), &mut args) }?;
```

This is the counterpart of `CudaDevice::get_or_load_custom_func`, and takes a
source rather than a compiled object for the obvious reason: the CUDA one hands
cudarc a PTX string, while here the source is what `hipcc` consumes. Compilation
and caching are exactly as above — the source text is part of the key, so an
edited kernel recompiles and an unchanged one is free from the second process
onwards.

The source is compiled the way candle's own modules are: the shim is
force-included and `cuda_utils.cuh`, `compatibility.cuh` and
`binary_op_macros.cuh` are on the include path, so a downstream kernel can be
written in the same CUDA-syntax dialect as `candle-kernels/src/*.cu`. What it
does *not* get is a say in the compile flags or its own staged headers; a module
has to be one translation unit.

The module name is not the whole key: the source is part of it in memory as well
as on disk, so reusing a name for a revised source loads the revision instead of
the module it replaces. The price is a SHA-256 pass over the source on every
call, which is worth avoiding in a launch loop — hold the returned function
rather than resolving it again. It stays valid because modules are never
unloaded (the handle borrows nothing, so unloading one would dangle it), which
is also the reason every distinct source stays resident for the life of the
process. A caller that generates a kernel per shape should bound the set of
sources it asks for.

Custom modules are namespaced apart from the built-ins, so naming one `unary`
resolves to yours rather than to candle's. `KernelCache::custom_function` and
`get_or_load_custom` are the same thing one layer down, for a caller holding the
cache directly. Only `RocmDevice`, `SendSyncDeviceMemory`, `launch_config*` and
`ParamBuffer` — all already public — are needed to get from a candle tensor to a
launch; `rocm_backend/tests_custom_kernel.rs` is a worked example that uses
nothing else.

## `ug` kernels

`UgIOp1` and `RocmDevice::compile` work under `--features "rocm ug"`, riding the
same custom-module pipeline: the generated HIP text goes to
`get_or_load_custom_func` under the module name `candle_ug_{func_name}`, so it is
cached and keyed exactly like any other source.

The code generator is the part with no upstream equivalent. `cuda` and `metal`
re-export `ug-cuda` and `ug-metal`; there is no `ug-rocm` crate, so
`candle-ug/src/rocm/code_gen.rs` emits HIP from the same SSA kernel directly, and
`candle-ug`'s `rocm` feature pulls no extra dependency. Block reductions come
from `candle-ug/src/rocm/reduce.hip`, appended only when the kernel contains a
`ReduceLocal`. It uses HIP's unsuffixed `__shfl_xor(x, mask, 32)` rather than the
`_sync` form for the reason in the shim section — the `_sync` spelling wants a
64-bit lane mask — with the width given explicitly so the reduction is also
correct on wave64 CDNA.

Two deliberate differences from `cuda_fwd`. The launch geometry is the one `ug`
lowered for (`kernel.launch_config()`), not a geometry re-derived from the
element count: for the `exp` sample `ug` asks for one block of one thread and
loops serially, where the CUDA rule launches 12 blocks that each redo the whole
loop and race on one buffer. Shared memory is honoured for the same reason; CUDA
hardcodes it to zero. Everything else is at parity, including the f32-and-
contiguous-only restriction the op carries on all three backends.

The one asymmetry that cannot be closed is the toolchain: CUDA compiles through
nvrtc in-process, while this path shells out to `hipcc`, so it must be on `PATH`
at *run* time and the first use of a kernel pays a compile.

## Environment variables

| Variable | Effect |
|---|---|
| `CANDLE_ROCM_ARCH` | Target architecture, e.g. `gfx1101`. Otherwise read from the device |
| `CANDLE_ROCM_VERSION` | Overrides the version parsed from `hipcc --version` |
| `CANDLE_ROCM_CACHE_DIR` | Cache root, overriding `~/.cache/candle-rocm` |
| `CANDLE_ROCM_FORCE_RECOMPILE` | `1` skips the cache read (the write still happens) |
| `CANDLE_ROCM_FORCE_DMMV` | Disables both `q8_1` quantized paths (MMVQ *and* MMQ); read in `quantized/rocm/q8_1.rs` |
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
| Unary / binary / affine / powf / elu | shared kernel | every dtype in the table below, `F8E4M3` included |
| `cmp`, `where_cond` | shared kernel | comparison writes `u8`, as on CUDA |
| `to_dtype` | shared kernel | see the dtype table below |
| `reduce_op` (sum/min/max/argmin/argmax) | shared kernel | `fast_*`; argmin/argmax return `u32`; no `F8E4M3` |
| `gather`, `scatter_add`, `index_select`, `index_add` | shared kernel | ids must be `u8`/`u32`/`i64` |
| `scatter` (set) | shared kernel | every dtype but `F8E4M3`, which has no `S_OP` |
| `arg_sort` | shared kernel | `asort_asc_*` / `asort_desc_*`; no `F8E4M3` |
| `matmul` | rocBLAS | `f32`/`f64`/`f16`/`bf16`, strided-batched; see the precision knobs below |
| `conv1d`, `conv2d` | im2col + rocBLAS GEMM | MIOpen instead under `--features miopen`; no `F8E4M3` |
| `conv_transpose1d` | col2im + GEMM | falls back to the shared kernel when dilation/padding/output-padding is non-trivial |
| `conv_transpose2d` | shared kernel | always direct, as on CUDA |
| `avg_pool2d`, `max_pool2d` | shared kernel | no `F8E4M3` |
| `upsample_nearest2d`, `upsample_bilinear2d` | shared kernel | |
| `upsample_nearest1d` | **errors** | no kernel exists; CUDA errors here too |
| `copy_strided_src`, `copy2d`, `const_set` | shared kernel + `hipMemcpy` | contiguous copies go through `hipMemcpy`/`hipMemcpy2D` |
| `rand_uniform`, `rand_normal` | rocRAND | generated in `f32`/`f64`, then cast |
| Quantized load / `dequantize` / `dequantize_f16` / `embedding` (`get_rows`) | shared kernel | all GGML block types |
| Quantized `fwd` (matmul) | MMVQ / DMMV / MMQ / dequantize + GEMM | batch 1-8 fused, larger batches tiled where MMQ pays; see below |
| `candle-nn`: `softmax_last_dim`, `rms_norm`, `layer_norm`, `sigmoid` | shared kernel | `rocm_fwd` on the custom op; f16/bf16/f32/f64 only |
| `candle-nn`: `rope`, `rope_i`, `rope_thd` | shared kernel | f16/bf16/f32/f64 only |
| `candle-nn`: `moe_gemm_gguf` | shared kernel | via `indexed_moe_forward`; see below |
| `ug` kernels (`UgIOp1`, `RocmDevice::compile`) | generated HIP | `--features "rocm ug"`; see below |

### dtypes

| dtype | Elementwise | `to_dtype` | reduce | indexing | matmul |
|---|---|---|---|---|---|
| `u8`, `u32`, `i64` | yes | yes | yes | yes | no |
| `f16`, `bf16`, `f32`, `f64` | yes | yes | yes | yes | yes |
| `i16`, `i32` | yes | **no** | **no** | **no** | no |
| `f8e4m3` | yes | partial | **no** | all but `scatter` | no |
| `f4`, `f6e2m3`, `f6e3m2`, `f8e8m0` | no | **no** | no | no | no |

Every gap in that table is **shared with CUDA** rather than specific to ROCm.
For all but the last, the reason is that the template is never instantiated in
`candle-kernels/src`, so there is no kernel for either backend to launch:

- `i16`/`i32`: `cast.cu` instantiates neither (its only `i32` pair is with fp8),
  and `reduce.cu`, `indexing.cu` and `conv.cu` instantiate neither.
- `f8e4m3` reduce, `arg_sort`, conv, pool and `candle-nn`'s softmax / rms_norm /
  layer_norm / rope: commented out at `reduce.cu:657-664`, `sort.cu:78-79` and
  `conv.cu:817-826`, each under a `NOTE: No … ops for f8`.
- `f8e4m3` `scatter` (set): `indexing.cu` has `IA_OP_F8` and `SA_OP_F8` for fp8
  but no `S_OP`, so only scatter-*add* has a kernel.
- `f8e4m3` `to_dtype` is "partial" because `cast.cu` pairs fp8 only with `f32`,
  `f64`, `f16`, `bf16`, `u8` and `i32`; `u32`, `i64` and `i16` have no `CAST_OP`.
- `f8e4m3` matmul: neither rocBLAS nor cuBLAS fp8 GEMM is wired up in candle.

Where a kernel *does* exist the ROCm backend launches it, so fp8 arithmetic is
live: unary, binary, `cmp`, `where_cond`, affine, powf, elu, both cast
directions, `gather`, `index_select`, `index_add`, `scatter_add`, `const_set`
and strided copy (`ucopy_f8_e4m3`, no longer the `ucopy_u8` bit-move). The one
gap that is genuinely ROCm-side is `candle-nn`'s `sigmoid` custom op, which
matches four slice variants and refuses fp8 even though `usigmoid_fp8_e4m3`
exists — `Tensor::sigmoid` through the plain unary path works.

Kernel *names* are the subtlety: the shared sources spell fp8 entry points
`_f8_e4m3` in some files and `_fp8_e4m3` in others. `rocm_backend/launch.rs`
carries the split as `FP8_SPELLED_ROOTS`, and a test scrapes the shipped `.cu`
text to keep the list honest.

### GEMM precision knobs

`candle::rocm` mirrors `candle::cuda`'s three getter/setter pairs, so portable
code can express the intent on either backend:

| knob | effect on ROCm |
|---|---|
| `set_gemm_reduced_precision_f16` | **live** — selects the `rocblas_datatype_f16_r` compute type (the analogue of `CUBLAS_COMPUTE_16F`) and passes alpha/beta as `f16` |
| `set_gemm_reduced_precision_bf16` | stored, no effect — rocBLAS has no bf16 compute type, so bf16 always accumulates in `f32` |
| `set_gemm_reduced_precision_f32` | stored, no effect — the xf32 analogue is a handle-wide *math mode* available only on CDNA (gfx94x), not a per-call compute type |

All three default to `false`, matching CUDA: f16 and bf16 go through
`rocblas_gemm_strided_batched_ex` with an `f32` compute type, so there is no
silent precision loss out of the box. `tests_gemm.rs` pins that the f16 knob
really changes the result rather than being accepted and ignored.

### Quantized matmul paths

`QRocmStorage::fwd` picks between four, all against kernels compiled from the
shared `quantized.cu`:

| batch (`b * m`) | activation | path |
|---|---|---|
| 1 | `f32`, K-quant, or under 8 Mi weight elements | DMMV (`dequantize_mul_mat_vec_*`) |
| 1 | `f32` legacy quant at 8 Mi elements or more | MMVQ (`mul_mat_vec_*_q8_1_cuda1`) |
| 1 | `f16` / `bf16` | MMVQ, activation cast to `f32` first |
| 2-8 | any float | MMVQ (`…_cuda<N>`) |
| any | any float, weights above the size threshold below | MMQ (`mul_mat_q*`) |
| any | anything left | dequantize the weights, then rocBLAS GEMM |

Both `q8_1` paths requantize the activations in a separate launch, which costs
roughly 14 µs; the size and dtype conditions are where that stops paying for
itself. `CANDLE_ROCM_FORCE_DMMV=1` disables both of them, which is an escape
hatch and how they are benchmarked against the alternatives
(`quantized::rocm::bench`, `--ignored`).

MMVQ's grid assumes the kernel's `rows_per_cuda_block` is 2 for a batch of two
or more (`mmvq.rs::launch_shape`). That constant has a 1-row arm, but it is
gated on `GGML_USE_HIPBLAS && __HIP_PLATFORM_AMD__ && (RDNA2 || RDNA3)`
(`quantized.cu:2972`) — and while this crate now *does* supply `RDNA3`, it has
never defined `GGML_USE_HIPBLAS`, so the arm stays off. Defining it would change
the tile height under a host that still divides by two; the geometry would have
to be read back the way `mmq_tiles` is.

The batch-of-one conditions are in `quantized/rocm/mmvq.rs::dmmv_is_faster`.
MMQ's are in `mmq.rs::min_work`, and are a size threshold per dtype rather than
a batch one — what MMQ saves is the dequantized weight matrix, so the weights
are what decide it:

| dtype | MMQ from | measured against the dense path |
|---|---|---|
| Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4K, Q5K | 0.25 Mi weight elements | 1.05-6.2x faster, at every size swept |
| Q6K | 1 Mi weight elements | 1.15-3.3x faster above it |
| Q3K | 2 Mi weight elements | 1.04-1.7x faster above it |
| Q2K | 4 Mi weight elements | 0.99-2.8x above it; a wash at batch 128, a win elsewhere |

These were measured on **gfx1101 (RDNA3) with the RDNA tile geometry** — the
`nwarps = 8` kernels that no longer spill — and are specific to it. The same
sweep under the Ampere tile set had seven of the ten dtypes losing at every
size, which is where the older, much higher thresholds came from. The first
seven above have no measured crossover at all: they win at 0.25 Mi, the
smallest weight matrix the sweep covers, and are floored there rather than at
zero only because nothing smaller was measured.

`bench_prefill_paths` prints the full sweep those come from, five timing
repeats per cell. Individual repeats spread up to ~47% at the smallest sizes,
where a call is 35 µs of mostly launch overhead, but the medians reproduce
across whole runs to under 2%. The thresholds sit where the ratio stops
favouring dense rather than where MMQ becomes decisive, because a wall-clock
ratio does not show what the dense path *allocates* — 524 MB of transient for a
4096x32000 `lm_head` at f32, which MMQ never materialises.

### Quantizing, and the raw weight pointer

`quantize`, `quantize_onto`, `quantize_imatrix` and `quantize_imatrix_onto` all
run the *CPU* quantizer and upload the result, exactly as `quantized/cuda.rs`
does — there is no device-side quantizer on either backend, so a ROCm-quantized
tensor is byte-identical to a CPU-quantized one
(`quantized::rocm::tests_imatrix`). The imatrix pair rejects a dtype outside
`Q2K`-`Q6K` up front, because `k_quants`' `from_float_imatrix` default body is a
`panic!` and the dtype comes from a config file.

`QRocmStorage::device_ptr` hands out the payload's device address for
downstream HIP kernels. `device_ptr_with_guard` is the ordered form: it takes
the *consumer's* stream, drains the device's own stream if that is a different
one, and drains the consumer's stream when the guard drops. Both are needed
because a freed block goes back on the allocator's free list rather than to the
driver, so a pointer that escapes its storage aliases the next tensor. The
guard borrows the storage, which makes that a compile error. `QStorage` exposes
it as `rocm_device_ptr_with_guard`, separate from the CUDA
`device_ptr_with_guard`, whose guard is tied to a `CudaStream`.

### Mixture-of-experts

`QRocmStorage::indexed_moe_forward` launches
`indexed_moe_forward_{q2k,q3k,q4k,q5k,q6k,q8_0}_q8_1` from the shared
`quantized.cu` — the MMVQ inner loop with the expert index folded into the
weight pointer, so it inherits the same `q8_1` activation requantization. It
takes a `(num_experts, n, k)` expert stack, `(batch, topk or 1, k)` f32
activations and `(batch, topk)` u32 routing, and returns `(batch, topk, n)`.

`candle_nn::moe::moe_gemm_gguf` is built on top of it: the CUDA entry point
takes the routing already sorted by expert (`sorted_token_ids` is the
permutation that sorted it), which `moe/rocm.rs` inverts with an arg-sort before
handing the unsorted routing to the kernel. `is_prefill` and `dtype` are ignored
— CUDA uses them to select a WMMA tile kernel, and there is only the one path
here. That makes `candle_transformers::fused_moe::FusedMoeGGUF`, and so
`quantized_qwen3_moe`, run on this backend.

Fixing this up exposed a bug in the shared kernel: `indexed_moe_forward` strided
between experts with a hard-coded `QK_K` rather than its own `qk` template
parameter. The two agree for every K-quant instantiation, but `q8_0` passes
`QK8_0` = 32, so its expert stride was an eighth of the expert matrix and every
routed pair past expert 0 read the wrong weights.

### Not implemented

These are the things CUDA does and this backend does not. Kept separate from the
*shared* limitations further down, because conflating the two overstates the gap.

- **The `fast_mmq` / `fast_mmvq` modules.** CUDA-only: they bind to
  `mmvq_gguf.cu` / `mmq_gguf/`, which use inline PTX (`ldmatrix.sync.aligned…`
  in `mmq_gguf/mmq_mma.cuh`), `nvcuda::wmma` and `__reduce_add_sync`. The plain
  `mul_mat_q*` kernels this backend does launch are the older, portable ones.
- **The dense fused MoE GEMM** (`candle_nn::moe::moe_gemm`, and the prefill half
  of `moe_gemm_gguf`). Both bind to `candle-kernels/src/moe/moe_wmma.cu` and
  `moe_wmma_gguf.cu`, which are `nvcuda::wmma` plus `<mma.h>` and are compiled
  into a host-side static library (`libmoe.a`) this backend has no build step
  for. The quantized MoE *is* implemented — see above — so `FusedMoeGGUF` runs
  at every batch size and `FusedMoe` does not; prefill goes through the same
  mat-vec kernel as decode, which is correct but not tile-optimised.
- **Flash attention.** `candle-flash-attn` and `candle-flash-attn-v3` depend on
  `candle-core` with `features = ["cuda"]` unconditionally — not behind a crate
  feature — so there is nothing to gate toward ROCm. Their build scripts drive
  `nvcc` over `*_sm80.cu` / `*_sm90.cu`.
- **NCCL/RCCL, and so multi-GPU collectives.** `candle-core`'s `nccl` feature is
  `["cuda", "cudarc/nccl"]`; the only consumer is the `llama_multiprocess`
  example, which also requires `flash-attn`. `rccl` appears nowhere in the tree.
- **CUDA graphs.** `CudaDevice::enable_cuda_graph_htod_cache` and the
  capture-aware upload path around it have no ROCm counterpart, and no
  `hipGraph` call exists here. `rocm_backend/params.rs` looks similar but is
  not the same thing: it caches launch-parameter buffers *unconditionally*, with
  an eviction bound, as a launch-overhead optimisation rather than to make
  replay legal.
- **A device built on an externally owned stream.** `RocmDevice::new_with_stream`
  and `Device::new_rocm_with_stream` exist for signature parity, but
  `RocmDevice::new` already calls `hipStreamCreate`, so they are aliases for it —
  unlike the CUDA pair, which chooses between a private and the per-thread
  default stream. Admitting a caller's stream was declined rather than deferred:
  roughly 120 sites assume one stream per device, and the free-list allocator
  recycles a freed block with no event between uses, so it would have to be
  reworked first. `is_event_tracking()` therefore returns `false` and
  `disable_event_tracking()` is a no-op.
- **`QStorage::device_ptr_with_guard` on the `CudaStream` signature.** It bails
  for a ROCm storage: a HIP allocation cannot be ordered against a CUDA stream.
  `rocm_device_ptr_with_guard` is the ROCm-side equivalent — see above.
- **`candle-nn`'s `sigmoid` on `F8E4M3`**, the one fp8 op with a kernel
  (`usigmoid_fp8_e4m3`) that this backend does not reach.
- **Two of the three GEMM precision knobs.** `set_gemm_reduced_precision_bf16`
  and `_f32` are accepted and stored but change nothing, for the reasons in the
  table above; both are live on CUDA. The f16 knob is live here too.

### Shared limitations

Neither backend does these, so they are not ROCm gaps. They are listed because
the document has previously counted them as such.

- `upsample_nearest1d` — no kernel exists; `cuda_backend/mod.rs` bails too.
- `i16`/`i32` in `to_dtype`, reduce, indexing and conv, and every `F8E4M3` entry
  in the dtype table above: reduce/argmin/argmax, `arg_sort`, conv, pool,
  softmax, rms_norm, layer_norm, rope, `scatter` (set) and matmul. See the notes
  under that table for which source omits which.
- `candle_nn::ops::sdpa` — the custom op is literally named `metal-sdpa` and
  implements only `metal_fwd`, so it errors on CUDA as well. CUDA's fused
  attention comes from the separate `candle-flash-attn` crates above.

### Errors

Everything the backend raises arrives as `candle::Error::Rocm`, wrapping a
`candle::rocm_backend::RocmError` — the same shape as `Error::Cuda`/`CudaError`,
so a ROCm failure carries a backtrace and can be matched on rather than
string-scraped. Both wrappers are `#[error(transparent)]`, so the rendered
message is the one written at the failing call site. `RocmError::Kernel` wraps
this crate's `KernelError`.

## Divergences from `cuda_backend`

Five places where this backend does *not* mirror the CUDA one, because the CUDA
behaviour looks like an upstream bug. Recorded here so they can be reported
rather than re-derived. One has since been fixed in the shared source; it is
kept below with the fix noted, since the point is the report.

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
3. **MMVQ wrote one row past the output for an odd row count.** *Fixed in the
   shared source, so this one is no longer a divergence — CUDA gets the fix
   too.* `mul_mat_vec_q` guarded its store on `threadIdx.x <
   rows_per_cuda_block` alone, where upstream llama.cpp also requires `row0 +
   threadIdx.x < nrows_dst`. At a batch of two or more the grid is
   `nrows.div_ceil(2)` blocks of two rows, so an odd `nrows` had the last block
   write `dst[j*nrows + nrows]` — the *first output of the next batch row*, not
   past the end of the buffer, so it corrupted silently. CUDA was exposed at
   every odd `nrows` with a batch of 2-8; ROCm dodged it by declining the shape.
   `quantized.cu` now carries the full guard plus a clamp on the phantom row's
   load, `mmvq.rs::supports` no longer needs its parity fallback, and
   `odd_output_rows_stay_correct` covers the shapes it now serves.
4. **CPU `conv_transpose1d` is wrong for batch > 1.** `cpu_backend/mod.rs:1415`,
   `MatMul::f`, collapses the batch loop when `b_skip == 0 && a_skip == m * k`
   without checking that the lhs rows are contiguous. The col2im path of
   `conv_transpose1d` produces exactly that shape with a strided lhs, so every
   batch after the first is computed from the wrong rows. The neighbouring
   collapse branch also derives `dst_rs` from the pre-collapse `n`, which makes
   it write overlapping rows. This one is a CPU bug, not a GPU one — it is
   recorded here because the ROCm conv tests are what surfaced it.
5. **Every CUDA `F8E4M3` launch names a kernel that does not exist.**
   `cuda_backend`'s `kernel_name` builds `{root}_{DType::as_str()}`, and
   `DType::F8E4M3.as_str()` is `"f8e4m3"` — while the shared sources spell their
   entry points `_f8_e4m3` or `_fp8_e4m3`. No symbol in `candle-kernels/src`
   ends in `_f8e4m3`. Since `cuda_backend/utils.rs` *does* dispatch the `F8E4M3`
   arm of every `Map*` trait, each of those ops resolves a missing function at
   run time instead of failing to compile. The hardcoded `"ucopy_f8e4m3"` for
   strided copy and the derived `cast_f8e4m3_*` names have the same defect. This
   backend keeps the two spellings in `rocm_backend/launch.rs::FP8_SPELLED_ROOTS`
   and pins them against the shipped `.cu` text with a test, which is why fp8
   works here and not there.

## Building and testing

Everything is driven from the repo-root `Makefile`:

| target | what it runs |
|---|---|
| `make check-rocm` | `cargo check` over the ROCm crates plus this one |
| `make clippy-rocm` | clippy on `candle-core`/`candle-nn` with tests and benches |
| `make test-rocm` | `test-rocm-core` then `test-rocm-nn` |
| `make test-rocm-core` / `-nn` | the GPU suites, filtered by `ROCM_FILTER` |
| `make test-rocm-suite SUITE=…` | one `candle-core` integration suite |
| `make test-rocm-ug` | the `ug` micro-kernel path |
| `make rocm-shim-test` | the shim, on real hardware; see below |
| `make rocm-cache-clean` | removes the compiled-kernel cache |

`ROCM_FILTER` defaults to `rocm`, not `_rocm`. The narrower spelling matched the
integration suites' `..._rocm` suffix but none of the backend's own module
tests, which are named for what they check and carry `rocm` only in their module
path (`quantized::rocm::tests_mmvq::…`) — so `test-rocm-core` was running 5 lib
tests and skipping 92. `ROCM_TEST_THREADS` defaults to 1 so that GPU memory use
and the attribution of a failure stay predictable; the disk cache is locked per
entry, so concurrency is safe, just noisier.

`test-rocm-ug` is separate because `ug` is not part of the `rocm` feature set —
`candle-core`'s `rocm` feature carries `candle-ug?/rocm`, which only fires if
`ug` is *also* enabled — so the suites above never build `UgIOp1`. It filters on
`ug` rather than `ROCM_FILTER`, the test being named for the op.

## Testing the shim

```
make rocm-shim-test
```

Compiles every shared module for the local GPU and runs
`src/hip_shim/shim_test.hip`, which exercises the only hand-written device code
in the project — the 16-bit `atomicAdd` CAS loops (aligned, unaligned, and
neighbour preservation), the `*_sync` shuffle wrappers, both `__syncwarp`
spellings, and `__dp4a` (which on gfx11/gfx12 lowers to `v_dot4_i32_iu8` via
`__builtin_amdgcn_sudot4` rather than to the portable loop) — on real
hardware.

The cache keying itself is covered by unit tests:

```
cargo test --manifest-path candle-rocm-kernels/Cargo.toml
```

## Layout

```
src/
  lib.rs           Id / Module / the eleven module constants
  compile/
    mod.rs         KernelCache: disk and in-memory caches, built-in and
                   custom module namespaces
    cache.rs       compile flags, cache keys and locations, locking,
                   atomic writes
    detect.rs      GPU architecture, MMQ tile geometry, toolchain detection
    hipcc.rs       driving hipcc and the bundler; staging the sources
    tests.rs       unit tests for the two module namespaces
  error.rs         KernelError
  wrappers.rs      Send + Sync wrapper around rocm-rs' Module
  hip_shim/        the CUDA-to-HIP bridge; the only HIP-specific code here
```
