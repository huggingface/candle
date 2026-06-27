# candle-flashinfer-kernels

Optional, feature-gated decode-attention backend for candle, alongside
`candle-flash-attn`. See
[huggingface/candle#3651](https://github.com/huggingface/candle/issues/3651).

`candle-flash-attn` targets the prefill (multi-token query) case. This crate
targets the decode case: each sequence in the batch contributes exactly one
new query token, attending over an arbitrarily long key/value cache, which is
the workload FlashInfer's batch-decode kernels are built for. The kernel
shipped here is a reference, numerically-stable (streaming softmax)
implementation of that same shape contract — one CUDA block per
`(batch, query head)`, looping sequentially over the KV cache. It is not a
port of FlashInfer's tensor-core / split-KV kernels, so it should not be
expected to match FlashInfer's own throughput; it exists to give candle a
working, swappable decode-attention entry point with the right API shape.

```rust
let out = candle_flashinfer_kernels::flashinfer_decode_attention(
    &q, // (batch, num_heads_q, head_dim)
    &k, // (batch, num_heads_kv, seqlen_k, head_dim)
    &v, // (batch, num_heads_kv, seqlen_k, head_dim)
    softmax_scale,
)?;
```

## Features

- **`cuda`** (off by default): compiles and links the CUDA kernel and enables the
  GPU forward pass. Requires the CUDA toolchain (`nvcc`).
- **`metal`** (off by default): enables the Metal forward pass on Apple Silicon
  (`f32`/`f16`). The shader is compiled at runtime, so no extra toolchain is needed.

With no GPU feature the crate builds CPU-only and the same
`flashinfer_decode_attention` entry point runs a reference CPU implementation
(`cpu_fwd`, `f32`/`f16`/`bf16`, parallelized across `(batch, head)` with rayon).
This makes the backend usable as a CPU fallback and testable without a GPU;
enable `cuda` or `metal` to run the corresponding GPU kernel.
