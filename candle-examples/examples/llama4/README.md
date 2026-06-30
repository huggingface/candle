# Llama 4

Llama 4 is Meta's MoE-based LLM family (Scout / Maverick). This implementation covers
the text-only decoder: iRoPE (alternating RoPE/NoPE attention layers), QK-norm,
attention temperature tuning on NoPE layers, and a top-k sigmoid-gated MoE block with
an always-on shared expert.

- Scout: 17B active / 109B total params, 16 routed experts.
- Maverick: 17B active / 400B total params, 128 routed experts.

Both are very large checkpoints; running them requires significant RAM/VRAM.

## Running the example

```bash
$ cargo run --example llama4 --release -- --prompt "Recursive fibonacci code in Rust:" --sample-len 150
```

By default this targets `meta-llama/Llama-4-Scout-17B-16E-Instruct` (gated on the Hub,
requires `huggingface-cli login`); pass `--model-id` to point at a different checkpoint.
