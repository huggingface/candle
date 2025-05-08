# candle-quantized-qwen3

[Qwen3]((https://qwenlm.github.io/blog/qwen3/)) is an upgraded version of Qwen2.5, released by Alibaba Cloud.

## Running the example

```bash
cargo run --example quantized-qwen3 --release -- --prompt "Write a function to count prime numbers up to N."
```

0.6b is used by default, 1.7b, 4b, 8b, 14b, and 32b models are available via `--model` argument.
