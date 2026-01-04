# PaddleOCR-VL

[PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) is a state-of-the-art
vision-language model for document parsing, developed by PaddlePaddle. With only 0.9B
parameters, it achieves competitive performance against much larger models (72B+) while
maintaining fast inference speeds.

## Features

- **Multilingual**: Supports 109 languages including Chinese, English, Japanese, Korean, Arabic, and more
- **Multi-element Recognition**: Handles text, tables, formulas, and charts
- **Dynamic Resolution**: NaViT-style encoder processes images at variable resolutions without distortion
- **Multi-Image Processing**: Process multiple images (e.g., multi-page documents) in a single prompt
- **Video Support**: Extract and process video frames with temporal position encoding
- **Efficient**: Compact 0.9B parameters with grouped query attention (GQA)
- **Position Embedding Caching**: LFU cache for interpolated position embeddings improves performance

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--image` | Path to document image (can be specified multiple times) | (required\*) |
| `--video` | Path to video file | (required\*) |
| `--fps` | Frames per second to extract from video | `1.0` |
| `--max-frames` | Maximum frames to extract from video | `16` |
| `--task` | Task type: `ocr`, `table`, `formula`, `chart` | `ocr` |
| `--model-id` | HuggingFace model ID | `PaddlePaddle/PaddleOCR-VL` |
| `--revision` | Model revision | `main` |
| `--max-length` | Maximum generation length | `1024` |
| `--cpu` | Run on CPU | `false` |
| `--bf16` | Use bfloat16 precision | `false` |
| `--seed` | Random seed | `299792458` |

\* Either `--image` or `--video` is required (mutually exclusive).

## Examples

### Basic Recognition

```bash
cargo run --example paddleocr-vl --release -- \
    --image candle-examples/examples/paddleocr-vl/test_ocr.jpg \
    --task ocr
```

### Table Recognition

```bash
cargo run --example paddleocr-vl --release -- \
    --image candle-examples/examples/paddleocr-vl/test_table.png \
    --task table
```

### Formula Recognition

```bash
cargo run --example paddleocr-vl --release -- \
    --image candle-examples/examples/paddleocr-vl/test_formula.png \
    --task formula
```

### Chart Recognition

```bash
cargo run --example paddleocr-vl --release -- \
    --image candle-examples/examples/paddleocr-vl/test_chart.png \
    --task chart
```

### Multi-Image (combined output)

Multi-Image OCR works with any task and uses `--task ocr` by default.

```bash
# Process multiple images with combined output
cargo run --example paddleocr-vl --release -- \
    --image candle-examples/examples/paddleocr-vl/test_ocr.png \
    --image candle-examples/examples/paddleocr-vl/test_ocr_page2.png
```

### Mutli-Image (batch)

```bash
# Process chosen images sequentially with distinct output
cargo run --example paddleocr-vl --release -- \
    --batch candle-examples/examples/paddleocr-vl/test_ocr.png candle-examples/examples/paddleocr-vl/test_ocr_page2.png

# With shell glob expansion
cargo run --example paddleocr-vl --release -- \
    --batch candle-examples/examples/paddleocr-vl/test_ocr*.png
```

### Video OCR

```bash
cargo run --example paddleocr-vl --release -- \
    --video candle-examples/examples/paddleocr-vl/test_video.mp4 \
    --task video \
    --fps 0.6 \
    --max-frames 64 \
    --max-length 2048
```
