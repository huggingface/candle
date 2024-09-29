# candle-phi3_5_moe: High performing 16x3.8B model, 6.6B active parameters

Model: [Phi-3.5 MoE](https://huggingface.co/microsoft/Phi-3.5-MoE-instruct)

The candle implementation provides the standard version.

## Running some examples

For the v2 version.
```bash
$ cargo run --example phi3_5_moe --release -- --model 2 \
  --prompt "A skier slides down a frictionless slope of height 40m and length 80m. What's the skier speed at the bottom?"
```
