# xLSTM Text Generation

This example demonstrates text generation using the [xLSTM](https://arxiv.org/abs/2405.04517) (Extended Long Short-Term Memory) model from NX-AI.

## Model Overview

xLSTM is a modernized LSTM architecture that achieves competitive performance with Transformers while maintaining linear complexity for inference. Key innovations include:

- **Exponential gating**: Stabilized gates using soft-capping and log-space computations
- **Matrix memory (mLSTM)**: Replaces the vector cell state with a matrix for increased capacity
- **Covariance update rule**: Memory update based on outer product of key-value pairs

## Quick Start

```bash
# Generate text with default prompt (requires GPU with ~14GB VRAM for bf16)
cargo run --example xlstm --release --features cuda -- --prompt "Once upon a time" -n 50

# Use f32 precision (requires ~28GB, more accurate)
cargo run --example xlstm --release --features metal,accelerate -- --dtype f32 --prompt "The meaning of life is"
```

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompt` | "Once upon a time" | The text prompt to generate from |
| `--cpu` | false | Run on CPU instead of GPU |
| `--dtype` | "bf16" | Data type: `f32`, `bf16`, or `f16` |
| `-n, --sample-len` | 100 | Number of tokens to generate |
| `--temperature` | None (greedy) | Sampling temperature (higher = more random) |
| `--top-p` | None | Nucleus sampling probability cutoff |
| `--seed` | 42 | Random seed for sampling |
| `--repeat-penalty` | 1.1 | Penalty for repeating tokens |
| `--repeat-last-n` | 64 | Context window for repeat penalty |
| `--model-id` | "NX-AI/xLSTM-7b" | HuggingFace model ID |
| `--revision` | "main" | Model revision/branch |
| `--tokenizer-file` | None | Path to local tokenizer.json |
| `--weight-files` | None | Comma-separated paths to weight files |
| `--config-file` | None | Path to local config.json |
| `--tracing` | false | Enable Chrome tracing output |