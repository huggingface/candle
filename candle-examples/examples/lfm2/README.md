# candle-lfm2: LFM2.5 (Liquid Foundation Model 2.5)

[LFM2.5](https://www.liquid.ai/) is a hybrid architecture from LiquidAI that combines
attention and short convolution layers for efficient sequence processing.
The 8B-A1B variants also use a mixture of experts in the feed-forward layers.

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

| `--which` | Model |
|---|---|
| `lfm2.5-230m` | [LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M) |
| `lfm2.5-230m-base` | [LFM2.5-230M-Base](https://huggingface.co/LiquidAI/LFM2.5-230M-Base) |
| `lfm2.5-350m` | [LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M) |
| `lfm2.5-350m-base` | [LFM2.5-350M-Base](https://huggingface.co/LiquidAI/LFM2.5-350M-Base) |
| `lfm2.5-1.2b` | [LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) |
| `lfm2.5-1.2b-base` | [LFM2.5-1.2B-Base](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Base) |
| `lfm2.5-1.2b-thinking` | [LFM2.5-1.2B-Thinking](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking) |
| `lfm2.5-1.2b-jp` | [LFM2.5-1.2B-JP](https://huggingface.co/LiquidAI/LFM2.5-1.2B-JP) |
| `lfm2.5-1.2b-jp-202606` | [LFM2.5-1.2B-JP-202606](https://huggingface.co/LiquidAI/LFM2.5-1.2B-JP-202606) |
| `lfm2.5-2.6b` | [LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) |
| `lfm2.5-2.6b-base` | [LFM2.5-2.6B-Base](https://huggingface.co/LiquidAI/LFM2.5-2.6B-Base) |
| `lfm2.5-8b-a1b` | [LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B) |
| `lfm2.5-8b-a1b-base` | [LFM2.5-8B-A1B-Base](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-Base) |

Other LFM2 checkpoints can be loaded with `--model-id`.

The weights use bf16 on CUDA and Metal and f32 on CPU, change this with `--dtype`.
On CPU the 2.6B and 8B-A1B models need about 10 GB and 32 GB of memory in f32,
the [quantized-lfm2](../quantized-lfm2) example is a lighter option there.
