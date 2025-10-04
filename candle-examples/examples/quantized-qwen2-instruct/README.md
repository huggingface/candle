# candle-quantized-qwen2-instruct

[Qwen2]((https://qwenlm.github.io/blog/qwen2/)) is an upgraded version of Qwen1.5, released by Alibaba Cloud.

## Running the example

```bash
cargo run --example quantized-qwen2-instruct --release -- --prompt "Write a function to count prime numbers up to N."
```

0.5b, 1.5b, 7b and 72b models are available via `--which` argument.

```bash
 cargo run --release --example quantized-qwen2-instruct --   --which 0.5b   --prompt "Write a function to count prime numbers up to N."
```
