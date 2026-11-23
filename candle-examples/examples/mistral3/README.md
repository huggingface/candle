# Mistral3 Vision-Language Model

This example demonstrates inference with Mistral3, a multimodal model combining Pixtral vision encoder with Mistral language model.

## Supported Models

- `mistralai/Mistral-Small-3.1-24B-Instruct-2503`
- `mistralai/Mistral-Small-3.2-24B-Instruct-2506`

## Requirements

- ~48GB memory for 24B model with BF16
- For GPU: CUDA or Metal support
- Optional: `tekken` feature for Mistral's custom tokenizer format

## Usage

### Basic Usage (HuggingFace Hub)

```bash
# Text-only mode
cargo run --release --example mistral3 -- \
    --prompt "What is the capital of France?"

# With image
cargo run --release --example mistral3 -- \
    --image path/to/image.jpg \
    --prompt "Describe this image in detail."
```

### With Local Model Directory

```bash
cargo run --release --example mistral3 -- \
    --model-dir /path/to/Mistral-Small-3.2-24B-Instruct-2506 \
    --image path/to/image.jpg \
    --prompt "What do you see in this image?"
```

### Platform-Specific

```bash
# CUDA
cargo run --release --features cuda --example mistral3 -- \
    --model-dir /path/to/model \
    --image image.jpg

# Metal (macOS)
cargo run --release --features metal --example mistral3 -- \
    --model-dir /path/to/model \
    --image image.jpg

# CPU (slow, requires more memory)
cargo run --release --example mistral3 -- \
    --cpu \
    --dtype f32 \
    --model-dir /path/to/model
```

### With Tekken Tokenizer

Mistral3 uses a custom tokenizer format (tekken.json). Enable the `tekken` feature for native support:

```bash
cargo run --release --features tekken --example mistral3 -- \
    --model-dir /path/to/model \
    --image image.jpg
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--cpu` | Run on CPU | GPU if available |
| `--dtype` | Data type: f32, f16, bf16, auto | auto |
| `--which` | Model variant: small-3.1-24b, small-3.2-24b | small-3.2-24b |
| `--model-id` | HuggingFace model ID | Based on --which |
| `--model-dir` | Local model directory | None |
| `--image` | Path to input image | None |
| `--prompt` | Text prompt | "Describe this image in detail." |
| `--temperature` | Sampling temperature | 0 (greedy) |
| `--top-p` | Nucleus sampling probability | None |
| `--top-k` | Top-K sampling | None |
| `--sample-len` | Max tokens to generate | 1000 |
| `--repeat-penalty` | Repeat penalty | 1.1 |
| `--repeat-last-n` | Context for repeat penalty | 64 |
| `--seed` | Random seed | 299792458 |
| `--vision-only` | Only run vision encoder | false |
| `--tracing` | Enable tracing | false |

## Architecture

Mistral3 consists of:
- **Vision Tower**: Pixtral vision encoder (24 layers)
- **Multi-Modal Projector**: RMSNorm + PatchMerger + MLP
- **Language Model**: Mistral (reused from `candle_transformers::models::mistral`)

The model uses `spatial_merge_size=2` to reduce image tokens by 4x through patch merging.

## Example Output

```
$ cargo run --release --features metal --example mistral3 -- \
    --model-dir ./weights/Mistral-Small-3.2-24B-Instruct-2506 \
    --image cat.jpg \
    --prompt "What animal is in this image?"

avx: true, neon: false, simd128: false, f16c: true
temp: 0.00 repeat-penalty: 1.10 repeat-last-n: 64
Device: Metal(MetalDevice), dtype: BF16
Loading from local directory: "./weights/Mistral-Small-3.2-24B-Instruct-2506"
Retrieved files in 1.23ms
Model config:
  Vision: hidden_size=1024, layers=24
  Text: hidden_size=5120, vocab_size=131072, layers=40
  Image token index: 10
  Spatial merge size: 2

Loading image: cat.jpg
  Image shape: [1, 3, 448, 448]
  Image sizes: [(448, 448)]

Loading model weights...
Loaded model in 12.34s

Generating...

The image shows a cat. It appears to be a domestic cat with orange and white fur...

156 tokens generated (8.45 token/s)
```
