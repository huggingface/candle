# Writing cuTile kernels

Candle's `cutile` feature is an opt-in bridge for authoring JIT-compiled CUDA kernels in Rust. It
enables the normal `cuda` feature and re-exports the cuTile version pinned by Candle.

## Requirements

- Rust 1.89 or newer.
- CUDA 13.2 or newer.
- NVIDIA driver r580 or newer, subject to the toolkit's driver compatibility requirements.
- An NVIDIA GPU with compute capability `sm_80` or newer.
- clang and libclang when building, because cuTile generates CUDA bindings.
- `tileiras` from the CUDA toolkit when a kernel is first compiled.

CUDA 13.2 supports cuTile on `sm_8x` and `sm_100+`. Hopper `sm_90` requires CUDA 13.3. CUDA 13.3
is the recommended toolkit and supports all architectures from `sm_80` onward. Upstream currently
tests cuTile on Linux.

Set `CUDA_TOOLKIT_PATH` to the toolkit root for a reproducible build and runtime setup:

```bash
export CUDA_TOOLKIT_PATH=/usr/local/cuda-13.3
cargo run --package candle-core --example cutile --features cutile
```

At runtime, cuTile searches for `tileiras` in this order: `CUTILE_TILEIRAS_PATH`,
`$CUDA_TOOLKIT_PATH/bin/tileiras`, standard CUDA installation directories, and `PATH`.

`CutileContext::new` asks that `tileiras` which architectures it accepts and fails with the
device architecture and the supported list when the device is not among them, so an unsupported
GPU or toolkit is reported before any kernel is compiled.

## Kernel interop

Import Candle's re-export under the name `cutile`. The cuTile procedural macro generates paths with
that name, and using the re-export keeps its compiler and runtime types on the version Candle expects.

```rust,ignore
use candle_core::cutile;
use cutile::cuda_async::device_operation::DeviceOp;
use cutile::tile_kernel::TileKernel;

#[cutile::module]
mod kernels {
    use candle_core::cutile;
    use cutile::core::*;
    use cutile::cutile_compiler;

    // Define #[cutile::entry] kernels here.
}
```

Keep `use candle_core::cutile` inside each annotated module. cuTile 0.3 generates unqualified paths
in child modules, and this import makes them resolve to Candle's re-export without a second
dependency. Adding a separate cuTile version would produce distinct stream, compiler, and launcher
types.

Existing [`CustomOp1`](https://docs.rs/candle-core/latest/candle_core/trait.CustomOp1.html),
`CustomOp2`, and `CustomOp3` implementations can launch cuTile kernels from `cuda_fwd`. For kernels
with more inputs, an ordinary function can use `Tensor::storage_and_layout`, allocate through
`CudaDevice`, and wrap its output with `CudaStorage::wrap_cuda_slice`.

`CutileContext::new` borrows the current Candle CUDA context and stream without taking ownership.
Use its guarded pointer helpers to preserve cudarc's event tracking:

```rust,ignore
let context = cutile::CutileContext::new(device)?;
let input = context.read_storage::<f32>(storage, layout)?;
let mut output = unsafe { device.alloc::<f32>(layout.shape().elem_count())? };
let output_ptr = context.write(&mut output, 0)?;

let launcher = unsafe {
    kernels::my_kernel(output_ptr.device_pointer(), input.device_pointer())
};
cutile::kernel("my kernel launch", || unsafe {
    launcher.async_on(context.stream())
})?;
drop((output_ptr, input));
```

Keep every `CutileRead` and `CutileWrite` guard alive until after `async_on` enqueues the launch.
Dropping the guards then records the reads and writes on Candle's stream, so later Candle operations
observe the correct ordering and allocations stay alive while the kernel runs.

## Warmup

Enable cuTile's persistent cubin cache once during process startup. The in-memory kernel cache is
always process-wide; the persistent cache lets later process starts reuse compiled cubins:

```rust,ignore
candle_core::cutile::jit_cache::enable_default()?;
```

The default cache is opt-in because it stores executable device code. It uses cuTile's private,
per-user cache directory and returns an error if a suitable directory cannot be established.

The first invocation of a new specialization runs the cuTile JIT compiler. Build the same launcher
that inference will use and call `compile_on` during model warmup:

```rust,ignore
// Precompile this specialization without executing the kernel.
cutile::kernel("my kernel warmup", || {
    launcher.compile_on(context.stream())
})?;
```

Warm every specialization selected by const generics, tensor stride metadata, scalar and pointer
divisibility hints, constant grid dimensions, and compile options. Runtime tensor extents and a
dynamic `.grid(...)` are not cache-key dimensions. `compile_on` compiles and caches the kernel
without launching it. Use `compile_on` and `async_on` with `CutileContext::stream`; the default
cuTile scheduler may choose a different stream and would not preserve Candle's ordering.

Warm kernels before latency-sensitive serving or CUDA graph capture. The full custom-op example is
[`candle-core/examples/cutile.rs`](https://github.com/huggingface/candle/blob/main/candle-core/examples/cutile.rs).

## Routed MoE

The `candle-nn` `cutile` feature provides a BF16 routed MoE grouped matrix multiplication under
`candle_nn::moe::cutile`. `MoeRouting::new` builds reusable routing metadata from a tensor of top-k
expert IDs and the expert count. `MoeInputMode::TokenRows` and `MoeInputMode::RoutedRows` select
whether the input has one row per token or one row per route. `routed_grouped_matmul` accepts
optional per-route weights.

Top-k IDs must be a contiguous `[tokens, top_k]` U32 or I32 CUDA tensor, with 1 to 1023 experts.
Inputs and `[experts, out_features, in_features]` weights must be contiguous BF16 CUDA tensors;
optional `[tokens, top_k]` route weights must be contiguous F32. Matrix dimensions and flattened
element offsets are limited to the kernel's i32 index range.

Enable `cutile` on `candle-nn`; this also enables the Candle core integration and CUDA:

```bash
cargo add --git https://github.com/huggingface/candle.git candle-nn --features "cutile"
```

```rust,ignore
use candle_nn::moe::cutile::{
    routed_grouped_matmul, warmup_routed_grouped_matmul, MoeInputMode, MoeRouting,
};

let routing = MoeRouting::new(&topk_ids, num_experts)?;
warmup_routed_grouped_matmul(
    &input,
    &expert_weights,
    &routing,
    MoeInputMode::TokenRows,
    route_weights.as_ref(),
)?;
let output = routed_grouped_matmul(
    &input,
    &expert_weights,
    &routing,
    MoeInputMode::TokenRows,
    route_weights.as_ref(),
)?;
```

Call `warmup_routed_grouped_matmul` with the same input, expert weights, routing, input mode, and
optional route weights before serving. It compiles the model-specific launch specializations used
by `routed_grouped_matmul` without executing the grouped GEMM. The implementation uses the same
`CutileContext` stream and guard model as custom ops; it does not add a paged-attention or serving
runtime dependency.
