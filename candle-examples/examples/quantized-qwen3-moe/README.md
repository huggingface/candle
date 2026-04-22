# candle-quantized-qwen3-moe

[Qwen3 MoE GGUF]((https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF)) contains the GGUF format of Qwen3 32B MoE models, developed by Alibaba Cloud.

## Running the example

```bash
# Local GGUF file
cargo run --features cuda --example quantized-qwen3-moe --release -- --model /path/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --prompt "Write a function to count prime numbers up to N."
```

Models available via `--which` argument: 16b_q2k, 16b_q4k, 16b_q6k, 16b_q80; 32b_q2k, 32b_q4k, 32b_q6k, 32b_q80;

```bash
# Obtained from Huggingface
cargo run --features cuda --example quantized-qwen3-moe --release -- --which 32b_q4k --prompt "A train is travelling at 120mph, how far does it travel in 3 minutes 30 seconds?"
```

