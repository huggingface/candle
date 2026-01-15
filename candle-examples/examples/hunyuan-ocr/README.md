# HunyuanOCR

HunyuanOCR is a Vision-Language Model optimized for document OCR tasks, developed by Tencent.

## Features

- High-quality text recognition from document images
- Dynamic resolution support for variable-sized images
- xDRoPE (Extended Dynamic Rotary Position Embedding) for position encoding
- Multi-image support for multi-page documents

## Running the Example

```bash
# Basic OCR with default prompt
cargo run --example hunyuan-ocr --release --features cuda -- \
    --image document.png

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

# Use local model path
cargo run --example hunyuan-ocr --release --features cuda -- \
    --model-id /path/to/HunyuanOCR \
    --image document.png

# Adjust generation parameters
cargo run --example hunyuan-ocr --release --features cuda -- \
    --image document.png \
    --max-length 2048 \
    --temperature 0.0
```

## Model

The default model is loaded from HuggingFace: `tencent/HunyuanOCR`

## Supported Platforms

- CUDA (recommended for best performance)
- Metal (Apple Silicon)
- CPU (slower but universal)
