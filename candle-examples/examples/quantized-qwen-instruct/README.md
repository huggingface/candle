# candle-quantized-qwen-instruct

- [Qwen2]((https://qwenlm.github.io/blog/qwen2/)) is an upgraded version of Qwen1.5, released by Alibaba Cloud.
- [Qwen2.5](https://qwenlm.github.io/blog/qwen2.5/) is an upgraded version of Qwen2, released by Alibaba Cloud.

## Running the example

```bash
cargo run --example quantized-qwen-instruct --release -- --prompt "Write a function to count prime numbers up to N."
```

- Available quantized models via `--which` argument:
    - "2-0.5b": Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_0.gguf
    - "2-1.5b": Qwen/Qwen2-1.5B-Instruct-GGUF/qwen2-1_5b-instruct-q4_0.gguf  
    - "2-7b": Qwen/Qwen2-7B-Instruct-GGUF/qwen2-7b-instruct-q4_0.gguf
    - "2-72b": Qwen/Qwen2-72B-Instruct-GGUF/qwen2-72b-instruct-q4_0.gguf
    - "2.5-0.5b": Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf
    - "2.5-1.5b": Qwen/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_0.gguf
    - "2.5-7b": Qwen/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-q4_0.gguf
    - "2.5-14b": Qwen/Qwen2.5-14B-Instruct-GGUF/qwen2.5-14b-instruct-q4_0.gguf
    - "2.5-32b": Qwen/Qwen2.5-32B-Instruct-GGUF/qwen2.5-32b-instruct-q4_0.gguf