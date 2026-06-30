# DeepSeek V3

DeepSeek V3 is an MoE model that builds on DeepSeek V2's Multi-head Latent
Attention (MLA), and switches the MoE router to an auxiliary-loss-free
("noaux_tc") sigmoid gate: expert and group selection is biased by a learned
per-expert correction term, while the weights used to combine expert outputs
come from the un-biased sigmoid scores.

Multi-token prediction and FP8 weights, which the original DeepSeek-V3
checkpoint also introduces, are not implemented here; this example expects
dequantized (e.g. bf16) weights and only runs the main next-token-prediction
path.

- Context length of **128k tokens**
- 256 routed experts + 1 shared expert per MoE layer, 8 experts active per
  token

## Running the example

```bash
$ cargo run --example deepseekv3 --release --features metal -- --prompt "Recursive fibonacci code in Rust:" --sample-len 150
```

Note that DeepSeek-V3 is a ~671B parameter model, so running it requires a
machine with a large amount of RAM/VRAM (or a quantized/dequantized subset of
the weights).
