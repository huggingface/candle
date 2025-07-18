# candle-qwen2.5-instruct: large language model series from Alibaba Cloud

- [Qwen2.5](https://qwenlm.github.io/blog/qwen2.5/) is an upgraded version of Qwen2, released by Alibaba Cloud.

## Running the example

```bash
cargo run --example qwen25-instruct --release  -- --prompt "Write a function to count prime numbers up to N."
```

- Various model sizes are available via the `--model` argument:
    - "0.5b": Qwen/Qwen2.5-0.5B-Instruct
    - "1.5b": Qwen/Qwen2.5-1.5B-Instruct
    - "7b": Qwen/Qwen2.5-7B-Instruct
    - "14b": Qwen/Qwen2.5-14B-Instruct
    - "32b": Qwen/Qwen2.5-32B-Instruct