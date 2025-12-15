# Swin Transformer

[Swin Transformer](https://arxiv.org/abs/2103.14030) is a hierarchical vision transformer that uses shifted windows for efficient self-attention computation. It achieves strong performance on image classification while maintaining linear computational complexity with respect to image size.

## Running the example

```bash
cargo run --example swin --release -- --image strawberry.jpg
```

### Command-line options

- `--image`: Path to the input image (required)
- `--which`: Model variant (default: `tiny`)
  - `tiny`: Swin-Tiny (28M params, 224x224)
  - `small`: Swin-Small (50M params, 224x224)
  - `base`: Swin-Base (88M params, 224x224)
  - `base-large`: Swin-Base (88M params, 384x384)
  - `large`: Swin-Large (197M params, 224x224)
  - `large-large`: Swin-Large (197M params, 384x384)
- `--model`: Optional path to local safetensors weights
- `--cpu`: Run on CPU instead of GPU
- `--version`: `v1` (default) or `v2` (not yet implemented)

### Examples

```bash
# Run with Swin-Tiny (default)
cargo run --example swin --release -- --image strawberry.jpg

# Run with Swin-Base
cargo run --example swin --release -- --image strawberry.jpg --which base

# Run with Swin-Large at 384x384 resolution
cargo run --example swin --release -- --image strawberry.jpg --which large-large

# Run on CPU
cargo run --example swin --release -- --image strawberry.jpg --cpu
```

## Model weights

Weights are automatically downloaded from HuggingFace Hub:

| Variant | HuggingFace Model ID | Top-1 Acc |
|---------|---------------------|-----------|
| tiny | `microsoft/swin-tiny-patch4-window7-224` | 81.2% |
| small | `microsoft/swin-small-patch4-window7-224` | 83.2% |
| base | `microsoft/swin-base-patch4-window7-224` | 83.5% |
| base-large | `microsoft/swin-base-patch4-window12-384` | 84.5% |
| large | `microsoft/swin-large-patch4-window7-224` | 86.3% |
| large-large | `microsoft/swin-large-patch4-window12-384` | 87.3% |