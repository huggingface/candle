# candle-quantized-qwen25-instruct

[Qwen2.5](https://qwenlm.github.io/blog/qwen2.5/) is an upgraded version of Qwen2, released by Alibaba Cloud.

## Running the example

```bash
cargo run --example quantized-qwen25-instruct --release -- --prompt "Write a function to count prime numbers up to N."
```

0.5b, 1.5b, 7b, 14b, 32b and DeepSeek-R1-Distill-Qwen-7B models are available via `--which` argument.

```bash
 cargo run --release --example quantized-qwen25-instruct --   --which 0.5b   --prompt "Write a function to count prime numbers up to N."
```