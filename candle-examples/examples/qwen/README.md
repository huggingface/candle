# candle-qwen: large language model series from Alibaba Cloud

Qwen 1.5 is a series of large language models that provide strong performances
on English and Chinese.

- [Blog post](https://qwenlm.github.io/blog/qwen1.5/) introducing Qwen1.5.
- [Model card](https://huggingface.co/Qwen/Qwen1.5-0.5B) on the HuggingFace Hub.
- [Blog post](https://qwenlm.github.io/blog/qwen-moe/) for the
  mixture-of-experts (MoE) variant.

## Running the example

```bash
$ cargo run --example qwen --release  -- --prompt "Hello there "
```

Various model sizes are available via the `--model` argument, including the MoE
variant.

```bash
$ cargo run --example qwen --release  -- --prompt "Hello there " --model moe-a2.7b
```

