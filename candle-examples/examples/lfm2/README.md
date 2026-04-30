# candle-lfm2: LFM2.5 (Liquid Foundation Model 2.5)

[LFM2.5](https://www.liquid.ai/) is a hybrid architecture from LiquidAI that combines
attention and short convolution layers for efficient sequence processing.

## Running the example

```bash
cargo run --example lfm2 --release -- --prompt "The capital of France is"
```

For the "thinking" model variant with chat template:

```bash
cargo run --example lfm2 --release -- \
    --which lfm2.5-1.2b-thinking \
    --prompt "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
```

On a CUDA-enabled machine with flash attention:

```bash
cargo run --example lfm2 --features cuda,flash-attn --release -- \
    --use-flash-attn --prompt "The capital of France is"
```

## Supported Models

1. [LFM2.5-1.2B](https://huggingface.co/LiquidAI/LFM2.5-1.2B)
2. [LFM2.5-1.2B-Thinking](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking)
