# SmolLM3 Unified Inference

A unified Rust implementation for running SmolLM3 models using the Candle ML framework. Supports both quantized (GGUF) and full precision (safetensors) models with a single codebase.

## Features

- **Dual Model Support**: Run either quantized or full precision models
- **Multiple Quantization Levels**: Q4_K_M (1.9GB), Q8_0 (3.3GB), F16 (6.2GB)
- **Chat Template Support**: Automatic formatting for instruction-tuned models
- **Thinking Mode**: Enable reasoning traces with `/think` mode
- **NoPE Architecture**: Supports SmolLM3's mixed RoPE/NoPE layer configuration
- **Auto-download**: Automatically fetches models from HuggingFace Hub

## Quick Start

### Quantized Model (Recommended)
```bash
cargo run --release --example smollm3 -- \
  --model-type quantized \
  --quantization q8_0 \
  --prompt "Explain Rust's ownership system"
```

### Full Precision Model
```bash
cargo run --release --example smollm3 -- \
  --model-type full \
  --dtype f16 \
  --prompt "Write a sorting algorithm in Rust"
```

## Command Line Options

### Model Selection
- `--model-type <TYPE>`: Choose `quantized` or `full` (default: quantized)
- `--model <VARIANT>`: Choose `3b` (instruct) or `3b-base` (default: 3b)
- `--quantization <LEVEL>`: For quantized models - `q4_k_m`, `q8_0`, or `f16` (default: q8_0)
- `--dtype <TYPE>`: For full models - `f32`, `f16`, `bf16`, or `auto` (default: auto)

### Generation Parameters
- `--prompt <TEXT>`: The prompt to generate from
- `-n, --sample-len <NUM>`: Number of tokens to generate (default: 1000)
- `--temperature <FLOAT>`: Sampling temperature, 0 for greedy (default: 0.8)
- `--top-p <FLOAT>`: Nucleus sampling probability cutoff
- `--top-k <NUM>`: Only sample among top K tokens
- `--repeat-penalty <FLOAT>`: Penalty for repeating tokens (default: 1.1)
- `--repeat-last-n <NUM>`: Context size for repeat penalty (default: 64)

### Advanced Options
- `--no-chat-template`: Disable chat template formatting (use for base models)
- `--thinking`: Enable thinking/reasoning mode with `/think` tags
- `--split-prompt`: Process prompt tokens individually (for debugging)
- `--tracing`: Enable performance tracing (generates trace JSON)
- `--model-path <PATH>`: Use local model file instead of auto-download
- `--tokenizer <PATH>`: Use local tokenizer instead of auto-download

## Quantization Comparison

| Level  | Size  | Quality | Use Case |
|--------|-------|---------|----------|
| Q4_K_M | 1.9GB | Good    | Fast inference, constrained environments |
| Q8_0   | 3.3GB | Better  | Balanced quality and speed |
| F16    | 6.2GB | Best    | Maximum quality in GGUF format |

## Examples

### Creative Writing with Thinking Mode
```bash
cargo run --release --example smollm3 -- \
  --thinking \
  --temperature 0.9 \
  --prompt "Write a short sci-fi story about AI"
```

### Code Generation (Base Model)
```bash
cargo run --release --example smollm3 -- \
  --model 3b-base \
  --no-chat-template \
  --temperature 0.2 \
  --prompt "def fibonacci(n):"
```

### High Quality Output
```bash
cargo run --release --example smollm3 -- \
  --model-type full \
  --dtype f16 \
  --temperature 0.7 \
  --prompt "Explain quantum entanglement"
```

## Model Architecture

SmolLM3 uses a hybrid RoPE/NoPE architecture:
- **RoPE layers**: Standard rotary position embeddings (75% of layers)
- **NoPE layers**: No position embeddings (25% of layers - every 4th layer)

This configuration is automatically detected and handled by the implementation.

## Hardware Requirements

- **Quantized Q4_K_M**: ~2.5GB RAM
- **Quantized Q8_0**: ~4GB RAM  
- **Full F16**: ~7GB RAM
- **Full F32**: ~13GB RAM

GPU acceleration supported via CUDA (with `cuda` feature) or Metal (macOS).

## Troubleshooting

**Model download fails**: Check internet connection and HuggingFace Hub access

**Out of memory**: Try a smaller quantization level or use `--sample-len` to reduce generation length

**Compilation errors**: Ensure you're using the latest version of the Candle crate

## License

This implementation follows the Candle framework license. SmolLM3 models are available under Apache 2.0.