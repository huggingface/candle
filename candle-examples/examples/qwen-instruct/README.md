# candle-qwen2-instruct: large language model series from Alibaba Cloud

- [Qwen2]((https://qwenlm.github.io/blog/qwen2/)) is an upgraded version of Qwen1.5, released by Alibaba Cloud.
- [Qwen2.5](https://qwenlm.github.io/blog/qwen2.5/) is an upgraded version of Qwen2, released by Alibaba Cloud.

## Running the example

```bash
$ cargo run --example qwen-instruct --release  -- --prompt "Hello there "
```

- Various model sizes are available via the `--model` argument:
    - "2-0.5b": Qwen/Qwen2-0.5B-Instruct
    - "2-1.5b": Qwen/Qwen2-1.5B-Instruct  
    - "2-7b": Qwen/Qwen2-7B-Instruct
    - "2-72b": Qwen/Qwen2-72B-Instruct
    - "2.5-0.5b": Qwen/Qwen2.5-0.5B-Instruct
    - "2.5-1.5b": Qwen/Qwen2.5-1.5B-Instruct
    - "2.5-7b": Qwen/Qwen2.5-7B-Instruct
    - "2.5-14b": Qwen/Qwen2.5-14B-Instruct
    - "2.5-32b": Qwen/Qwen2.5-32B-Instruct