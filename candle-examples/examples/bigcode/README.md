# candle-starcoder: code generation model

[StarCoder/BigCode](https://huggingface.co/bigcode/starcoderbase-1b) is a LLM
model specialized to code generation. The initial model was trained on 80
programming languages.

## Running some example

```bash
cargo run --example bigcode --release -- --prompt "fn fact(n: u64) -> u64 "

> fn fact(n: u64) -> u64  {
>     if n == 0 {
>         1
>     } else {
>         n * fact(n - 1)
>     }
> }
```
