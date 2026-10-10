# A candle GPU backend on Rust `std::offload`

Status: **llama runs on the GPU.** TinyLlama-1.1B, f32, gfx1201 (RX 9070 XT),
**38.8 tok/s against candle's own CPU backend at 10.2 tok/s (3.8x)**, with
byte-identical output.

> **2026-10-06:** two more models measured on this backend, and the direct
> (framework-free) ports for comparison, in
> [../../ports/offload-models/REPORT.md](../../ports/offload-models/REPORT.md) §3.5: SmolLM2-135M
> runs 99.3% on the GPU at 54 tok/s (1.9x candle CPU) but issues 577 launches
> per token; candle's `llama2_c` model (stories15M) falls back to the CPU for
> 39% of its ops (broadcast `mul/sub/div`, strided copies) and runs at 45 tok/s,
> 4x *slower* than candle's CPU path. `build-offload.sh` now takes `TC=` and
> `FEATURES=` (e.g. `EXAMPLE=llama2-c FEATURES=candle-datasets`).

```sh
./build-offload.sh
B=~/.cache/rust-offload/candle-target/release/examples/llama
HF_HUB_OFFLINE=1 $B --which tiny-llama-1.1b-chat --dtype f32 -n 64 \
    --prompt "The capital of France is" --temperature 0
$B --cpu ...      # same binary, CPU baseline
```

---

## 1. What changed

Only 158 lines of existing candle were touched. Everything else is new and
self-contained.

| Where | Lines | What |
|---|---:|---|
| `candle-offload-kernels/` (new) | 930 | 14 amdgcn kernels + their `offload!` launchers, exported over the C ABI |
| `candle-core/src/offload_backend/` (new) | 1447 | `BackendDevice` + `BackendStorage`, libomptarget allocation, transfer accounting |
| `build-offload.sh` (new) | 69 | the four-pass build |
| `candle-core/src/lib.rs` | 7 | select the backend |
| `candle-core/src/utils.rs` | 4 | `cuda_is_available()` must see it |
| `candle-nn/src/ops.rs`, `rotary_emb.rs` | 133 | GPU arms for 7 `CustomOp`s |
| 5 × `Cargo.toml` | 13 | an `offload` feature |
| `candle-examples/.../llama/main.rs` | 3 | print the transfer report |

### It occupies the CUDA slot rather than adding a variant

`Device` and `Storage` are closed enums and every match arm across candle-core,
candle-nn and candle-transformers enumerates `Cpu | Cuda | Metal`. A fourth
variant would touch hundreds of sites. But candle already compiles a *dummy*
CUDA backend when the `cuda` feature is off, so `lib.rs:117` is a ready-made
seam:

```rust
#[cfg(all(not(feature = "cuda"), feature = "offload"))]
pub use offload_backend as cuda;
```

`Device::new_cuda(0)` now returns an offload device, and every existing match
arm works unchanged. This is why the diff is 158 lines instead of thousands.

### The kernels are a cdylib, not a dependency

This is the part that took the longest to get right, and it is a real constraint
rather than a style choice.

rustc performs the offload lowering **during fat LTO**, and fat LTO never runs
for an rlib. So the crate whose codegen ends up holding an `offload!` expansion
must be a *final linked artifact*. Anything else fails with:

```
error: using the offload feature requires -Z offload=<Device or Host=...>
error: using the offload feature requires -C lto=fat
```

and passing `-C lto=fat` to an rlib **silently produces no artifact at all** --
cargo reports success and there is no `.rlib` on disk.

That rules out the obvious layering, kernels <- candle-core <- candle-nn <-
example: whichever rlib in that chain holds the launcher codegen cannot satisfy
the requirement, and *which one that is depends on inlining decisions*. Marking
the launchers `#[inline(never)]` only moved the failure from candle-core to the
kernels crate.

Building the kernels as a **cdylib** cuts the graph. The offload intrinsic exists
only there; candle-core reaches the launchers through `extern "C"`. No candle
crate needs an offload flag, fat LTO, or a nightly feature gate.

Measured, not assumed: a three-crate probe confirmed a plain `cargo rustc --`
flag reaches only the top crate (the manifest came back empty, 14 bytes), that
RUSTFLAGS reaches all of them but then `-Zbuild-std`'s own crates each emit a
competing `device.bin`, and finally that a cdylib consumer links and runs with
zero offload flags.

### Data stays on the device

`std::offload`'s entire public API is four items and none of them allocate.
`offload!` infers transfers from argument *types*, so a buffer passed as `&[T]`
is copied host->device before the launch and back after, **every launch**. That
is fatal for a framework backend, where `alloc_uninit` must return storage that
survives hundreds of ops.

So allocation goes straight to libomptarget, and device addresses reach kernels
as `usize` scalars -- `offload!` passes scalars by value and therefore does not
map them. Every kernel signature in this backend is `(usize, usize, ..., usize)`
for that reason.

One correction worth recording: the OpenMP device API is **split across two
libraries**. `omp_target_alloc` / `_free` / `_memcpy` are in **libomptarget**;
`omp_get_default_device` / `_initial_device` / `_num_devices` are in **libomp**.
A single `#[link(name = "omptarget")]` block appears to work only because
`-Zoffload` implicitly adds `-lomp`. Take the offload flag away, as candle-core
does, and the `omp_get_*` symbols go undefined at link time.

---

## 2. Data transfer

64 tokens, TinyLlama-1.1B f32:

```
  host -> device          552 calls       2109.60 MiB
  device -> host          481 calls        272.52 MiB
  device allocs         33287 calls       7504.55 MiB   (33039 frees)
  kernel launches       27147
  ops on GPU            27147 / 27345 (99.3%)
```

The 2.1 GiB host->device is the model: safetensors ship f16, and it is uploaded
**as f16** and widened on the device. The 272 MiB back is almost entirely one
prompt-time matmul (see below).

The clearest evidence that residency works is what happens between run lengths:

| | 12 tokens | 64 tokens | per extra token |
|---|---:|---:|---:|
| host -> device | 2103.25 MiB | 2109.60 MiB | 0.12 MiB |
| device -> host | 259.83 MiB | 272.52 MiB | 0.24 MiB |

52 extra tokens cost 12.7 MiB. Tensors are living on the device across ops, which
is exactly what `std::offload`'s own API cannot express.

### How it got there

Transfer was the whole story at first, and three rounds of measurement moved it:

| | tok/s | D2H | ops on GPU |
|---|---:|---:|---:|
| GPU kernels off, storage still device-resident | 1.65 | 52851 MiB | 0% |
| GEMM + elementwise only | 4.30 | 5358 MiB | 68.7% |
| \+ RmsNorm, softmax, rotary | 4.22 | 5358 MiB | 93.4% |
| \+ `to_dtype`, `index_select` | 5.51 | 260 MiB | 97.3% |
| \+ GEMV | **38.84** | 260 MiB | 97.3% |

Per-op byte attribution was what made this tractable. Call counts are
misleading: adding the three transformer CustomOps removed 1332 of 1691
fallbacks and barely moved throughput, because the **bytes** were elsewhere.
Sampling the transfer counters around each fallback showed it immediately:

```
      to_dtype                          202 calls     6294.54 MiB
      index_select                       12 calls     3000.13 MiB
```

`to_dtype` was the model load (f16 up, down, convert, back up); `index_select`
was the 256 MiB embedding table downloaded **once per token**, because the CPU
fallback path pulls the whole buffer. Two small kernels removed 9.3 GiB.

### What still crosses the bus

```
      matmul                              1 call      250.17 MiB
      copy_strided_src                  110 calls      10.12 MiB
      reduce_op                          64 calls       7.81 MiB
      where_cond                         22 calls       0.19 MiB
```

The 250 MiB matmul is one-off, at prompt time. Decode narrows to the last
position with `x.i((.., seq_len - 1, ..))`; for `seq_len == 1` that leaves the
left operand row-major and the kernel takes it, but for the prompt pass the row
stride is `seq_len * hidden` and the kernel declines, so the 256 MiB lm_head
weight is downloaded once. Passing explicit left-hand strides would fix it.

---

## 3. Kernel launches

27147 launches for 64 tokens, **~424 per token**. Against the 18-22 us launch
floor measured earlier in this project (`../../probes/offload-candle/FINDINGS.md` §3), that
is roughly 8 ms/token of pure launch overhead against a 25.8 ms/token budget --
on the order of 30%, and the largest single structural cost left.

It is high because `offload!` launches one kernel per candle op and there is no
way to do otherwise: no streams, no graphs, no fusion, and `synchronize()` is a
no-op because every launch is already blocking (G2).

### The kernels

14 in the device image. The ones that mattered:

- **`k_gemv_f32_bt`** -- the single biggest win in the whole exercise, **5.5 ->
  38.8 tok/s**. Autoregressive decode is all matrix-*vector*, and the general
  GEMM parallelises over output elements: a 2048x2048 decode matmul gets 512
  threads and leaves the GPU idle. Here one *wave* owns one output element and
  its 32 lanes split the k-loop -- 32x the parallelism, coalesced reads of B.
- **`k_gemm_f32`** -- batched, with explicit strides on B so `x.matmul(&w.t())`
  never materialises the transpose.
- **`k_gemm_f32_bt4`** -- 4 output columns per thread for the prompt pass.
- **`k_rms_norm_f32`, `k_softmax_last_f32`** -- one wave per row, two wave
  reductions each.
- **`k_rope_f32`, `k_index_select_f32`, `k_convert`** -- embarrassingly parallel.
- **`k_binary_f32`, `k_unary_f32`** -- opcode dispatch inside one kernel. The
  opcode is a scalar, so it lands in a scalar register and the branch is uniform
  across the wave: a scalar compare, not divergence. Twenty separate kernels
  would mean twenty `offload!` sites and twenty device-image entries.

Anything not on that list falls back through the CPU, which keeps the backend
total -- llama runs end to end -- and makes the GPU coverage explicit and
measurable rather than implicit. `CANDLE_OFFLOAD_CPU_ONLY=1` disables every GPU
path while keeping tensors resident, which is how the first row of the table
above was measured; `CANDLE_OFFLOAD_DISABLE=matmul,unary,...` disables them
individually.

---

## 4. Gaps, scoped to this task

Numbering continues from `../../probes/offload-candle/FINDINGS.md` (G1-G13). These are the
ones that actually cost work here; the earlier list still stands.
Re-verified 2026-10-06 in `../../probes/offload-candle/VERIFICATION.md`: G14 confirmed
(known upstream as rust-lang/rust#157563), G15 is reachable via
`link_llvm_intrinsics`, G17 disappears with `-Cpanic=immediate-abort`, and G19
is mostly cheap small allocations. The backend builds and runs unchanged on
`nightly-2026-10-06`.

**G14 -- the offload lowering only works in a final linked artifact.**
Fat LTO never runs for an rlib, so any library that holds an `offload!`
expansion cannot be compiled. Two things make this worse than a documentation
problem. First, *which* crate holds the expansion depends on inlining, so the
error lands on an innocent crate -- here candle-core, which contains no offload
code at all. Second, the suggested remedy fails silently: passing `-C lto=fat`
to an rlib reports success and emits nothing. The cdylib workaround costs a C
ABI boundary, hand-synchronised opcode constants on both sides, and the loss of
every Rust type across the seam.
*Fix:* allow the lowering for rlibs, or reject `-C lto=fat` on an rlib loudly.

**G15 -- wave reductions are integer-only.**
`core::arch::amdgpu` has `wave_reduce_add` (`u32`), `wave_reduce_max` / `_min`
(`i32`), and no float form at all. Every reduction in a transformer is float:
RmsNorm, softmax, and the GEMV that gave the 7x. Combined with G6 (LDS has no
Rust accessor), the only route left is a hand-rolled XOR butterfly over
`ds_bpermute`, bitcasting f32 through `u32` by hand. It works -- verified with
`../../probes/offload-candle/gapscratch/wave_reduce_probe.rs`, wavefront 32, every lane
holding the correct sum -- but three of the most important kernels here depend
on the author knowing that trick.
*Fix:* `wave_reduce_add`/`_max`/`_min` for `f32`, or an LDS accessor.

**G16 -- candle-nn hard-gates every GPU path on `feature = "cuda"`.**
Seven `CustomOp`s -- softmax, RmsNorm, LayerNorm, Sigmoid, three rotary variants
-- implement `cuda_fwd` under `#[cfg(feature = "cuda")]`, and the trait default
is an error. A backend in the `Cuda` slot therefore has *every primitive op
working* and still dies the moment a model calls `rms_norm`, which is every
transformer. This is a framework-design gap rather than a language one, but it
is the single thing most likely to block anyone repeating this on another
framework.
*Fix (upstream candle):* let `cuda_fwd` default to the CPU kernel instead of an
error, or key the cfg on "some GPU backend" rather than on cudarc specifically.

**G17 -- guarding a division's operands is not enough.**
G4 (panic paths do not link on device) recurs on every new kernel, but with a
sharper edge than previously recorded: `if m == 0 || n == 0 { return }` does
**not** let LLVM elide the check on `idx / (m * n)`, because the multiply can
wrap. The divisor *expression* has to be tested. This cost a device-link failure
in `k_gemm_f32` and `k_gemm_f32_bt4` with an error naming
`core::panicking::panic_const_div_by_zero` and no indication of which division
was responsible.

**G18 -- no `bf16`, but the bit layout is reachable anyway.**
`f16` is a real primitive under `#![feature(f16)]` and `as` casts to and from it
lower to hardware instructions on gfx1201. `bf16` has no primitive (G10), so
`k_convert` handles it as `u16` with the conversion written by hand -- widening
is a 16-bit shift, narrowing is round-to-nearest-even. That works and is how the
model loads, but nothing in the type system knows those `u16`s are floats, and
the rounding had to be reverse-engineered from `half::bf16::from_f32` to match.

**G19 -- allocation churn has no answer in the API.**
33287 allocations and frees for 64 tokens, one per intermediate tensor. candle's
CUDA backend keeps a `ParamCache` precisely because allocation is too expensive
to repeat; there is no equivalent here, and `omp_target_alloc` is the only
primitive available. Not yet measured in isolation -- it is the obvious next
thing to look at after the launch floor.

---

## 5. Honest limits

- **f32 only on the GPU.** f16 and bf16 tensors are stored and converted on the
  device but arithmetic on them falls back. `--dtype f16` will run and will be
  slow.
- **Not tested beyond TinyLlama-1.1B, batch 1, greedy decoding.** Correctness is
  established by byte-identical output against the CPU backend at 64 tokens, not
  by a test suite.
- **Broadcast layouts fall back.** Binary ops require both operands contiguous;
  a stride-0 broadcast is not contiguous and goes to the CPU.
- **The 250 MiB prompt-time matmul download is a real cost** for long prompts and
  grows with prompt length; it has a known fix that is not yet implemented.
- **The `usize`-laundered device pointer relies on scalars not being mapped**,
  which may be an implementation detail rather than a guarantee. The whole
  backend rests on it. Still worth confirming upstream.
