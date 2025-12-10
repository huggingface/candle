# candle-quantized-qwen3

[Qwen3](<(https://qwenlm.github.io/blog/qwen3/)>) is an upgraded version of Qwen2.5, released by Alibaba Cloud.
Here is the MoE version of Qwen3, but quantized.

## Running the example

```bash
cargo run --example quantized-qwen3-moe --release -- --prompt "Write a function to count prime numbers up to N."
```

30b is used by default, 235b model is available via `--which` argument.

```bash
cargo run --example quantized-qwen3-moe --release -- --which 235b --prompt "A train is travelling at 120mph, how far does it travel in 3 minutes 30 seconds?"
```

To run on cuda(gpu).

```bash
cargo run --example quantized-qwen3-moe --release --features cuda -- --prompt "Write a function to count prime numbers up to N."
```
