# HunyuanOCR

HunyuanOCR is a Vision-Language Model optimized for document OCR tasks, developed by Tencent.

## Features

- High-quality text recognition from document images
- Dynamic resolution support for variable-sized images
- xDRoPE (Extended Dynamic Rotary Position Embedding) for position encoding
- Multi-image support for multi-page documents
- Flash Attention support on CUDA for faster inference
- SDPA (Scaled Dot Product Attention) on Metal for efficient decode

## Running the Example

```bash
# Basic OCR with default prompt
cargo run --example hunyuan-ocr --release --features cuda -- \
    --image document.png

# Enable Flash Attention for faster inference (CUDA only)
cargo run --example hunyuan-ocr --release --features cuda,flash-attn -- \
    --image document.png \
    --flash-attn

# Custom prompt
cargo run --example hunyuan-ocr --release --features cuda -- \
    --image document.png \
    --prompt "Extract all text from this image"

# Multi-page document OCR
cargo run --example hunyuan-ocr --release --features cuda -- \
    --image page1.png --image page2.png

# Batch mode - process multiple images sequentially
cargo run --example hunyuan-ocr --release --features cuda -- \
    --batch doc1.png doc2.png doc3.png

# Run on CPU
cargo run --example hunyuan-ocr --release -- \
    --cpu \
    --image document.png

# Run on Metal (Apple Silicon) - SDPA is automatically used for decode
cargo run --example hunyuan-ocr --release --features metal -- \
    --image document.png

# Use local model path
cargo run --example hunyuan-ocr --release --features cuda -- \
    --model-id /path/to/HunyuanOCR \
    --image document.png

# Adjust generation parameters
cargo run --example hunyuan-ocr --release --features cuda -- \
    --image document.png \
    --max-length 2048
```

## Model

The default model is loaded from HuggingFace: `tencent/HunyuanOCR`

## Supported Platforms

- CUDA (recommended for best performance, supports Flash Attention with `--features flash-attn`)
- Metal (Apple Silicon, uses SDPA for efficient decode)
- CPU (slower but universal)
