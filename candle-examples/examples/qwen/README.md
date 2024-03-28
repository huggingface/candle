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
$ cargo run --example qwen --release  -- --model moe-a2.7b --prompt 'def print_prime(n: int): '
def print_prime(n: int):  # n is the number of primes to be printed
    for i in range(2, n + 1):
        if all(i % j != 0 for j in range(2, i)):
            print(i)
```

