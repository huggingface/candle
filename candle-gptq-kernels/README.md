# candle-gptq-kernels

Fused dequantize+GEMM CUDA/Metal kernels for GPTQ-quantized linear layers, used from
[`candle_transformers::quantized_gptq`](../candle-transformers/src/quantized_gptq.rs) when the
`gptq-cuda`/`gptq-metal` feature is enabled on `candle-transformers`.

## Why a separate crate instead of living in `candle-kernels`

[`candle-kernels`](../candle-kernels) holds the original CUDA kernels candle ships with (elementwise
ops, reductions, the GGUF quantized matmul types, etc.), built via `nvcc` through `build.rs` and
consumed unconditionally by `candle-core`'s `cuda` feature.

GPTQ/AWQ/FP8 support is split into its own crate per format (this one, plus
[`candle-awq-kernels`](../candle-awq-kernels) and [`candle-fp8-kernels`](../candle-fp8-kernels))
instead of being added as more `.cu` files inside `candle-kernels`, for a few reasons:

- **Optionality**: these kernels are only needed by checkpoints in that specific quantization
  format. Folding them into `candle-kernels` would mean every `candle-core` consumer with the
  `cuda` feature compiles GPTQ/AWQ/FP8 kernels (and links against `cudaforge`'s codegen for them)
  whether or not they ever load such a checkpoint. As separate crates, they're only built when
  `candle-transformers` is built with the corresponding `{gptq,awq,fp8}-cuda`/`-metal` feature.
- **No `candle-core` dependency edge**: `candle-kernels` is a dependency of `candle-core` itself
  (used to build the base CUDA backend). These crates instead depend *on* `candle-core` (for
  `Tensor`/`DType` glue, vendored Marlin repacking, etc.) and sit at the `candle-transformers`
  layer, alongside the format's CPU dequantization logic in `quantized_{gptq,awq,fp8}.rs`.
- **Independent CUDA/Metal feature gating**: each crate has its own `cuda`/`metal` Cargo features
  and `build.rs`, so e.g. enabling `gptq-metal` doesn't require pulling in AWQ's or FP8's kernel
  sources at all.
- **Vendored third-party code**: this crate also vendors the upstream Marlin CUDA kernel
  (`kernels/marlin/marlin_cuda_kernel.cu`) for the fast 4-bit tensor-core GEMM path, which is kept
  isolated from candle's own kernel sources rather than mixed into `candle-kernels`.

See each crate's top-level module doc comment (`src/lib.rs`) for what each one implements.
