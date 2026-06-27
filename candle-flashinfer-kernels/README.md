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
