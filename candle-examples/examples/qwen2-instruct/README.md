# candle-qwen2-instruct: large language model series from Alibaba Cloud

- [Qwen2]((https://qwenlm.github.io/blog/qwen2/)) is an upgraded version of Qwen1.5, released by Alibaba Cloud.

## Running the example

```bash
cargo run --example qwen2-instruct --release  -- --prompt "Write a function to count prime numbers up to N."
```

- Various model sizes are available via the `--model` argument:
    - "0.5b": Qwen/Qwen2-0.5B-Instruct
    - "1.5b": Qwen/Qwen2-1.5B-Instruct  
    - "7b": Qwen/Qwen2-7B-Instruct
    - "72b": Qwen/Qwen2-72B-Instruct