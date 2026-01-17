<!--
 * @Author: SpenserCai
 * @Date: 2026-01-12 19:55:16
 * @version: 
 * @LastEditors: SpenserCai
 * @LastEditTime: 2026-01-12 20:58:49
 * @Description: file content
-->
# BiRefNet

BiRefNet (Bilateral Reference Network) is a state-of-the-art model for high-resolution dichotomous image segmentation, commonly used for background removal tasks.

- [Paper](https://arxiv.org/abs/2401.03407)
- [GitHub](https://github.com/ZhengPeng7/BiRefNet)
- [HuggingFace](https://huggingface.co/ZhengPeng7/BiRefNet)

## Model Variants

| Model | HuggingFace Repo | Description |
|-------|------------------|-------------|
| BiRefNet | [ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) | Original BiRefNet model |
| BiRefNet-General | [ZhengPeng7/BiRefNet_T](https://huggingface.co/ZhengPeng7/BiRefNet_T) | Lightweight version |
| RMBG-2.0 | [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) | Fine-tuned for background removal |

All variants share the same architecture and can be used with this example.

## Usage

### Auto-download from HuggingFace

The example will automatically download the model from HuggingFace if no local model is specified:

```bash
cargo run --example birefnet --release --features cuda -- \
    --image input.jpg \
    --output output.png
```

### Use Local Model

Download the model manually:

```bash
hf download ZhengPeng7/BiRefNet model.safetensors --local-dir ./BiRefNet
```

Then run with the local model:

```bash
cargo run --example birefnet --release --features cuda -- \
    --model ./BiRefNet/model.safetensors \
    --image input.jpg \
    --output output.png
```

### Output Mask Only

To output a grayscale mask instead of an RGBA image with transparent background:

```bash
cargo run --example birefnet --release --features cuda -- \
    --image input.jpg \
    --output mask.png \
    --mask-only
```

### Benchmark Mode

Run multiple iterations to measure inference performance:

```bash
cargo run --example birefnet --release --features cuda -- \
    --image input.jpg \
    --output output.png \
    --bench 10
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Path to local model file | Auto-download |
| `--model-id` | HuggingFace model repository | `ZhengPeng7/BiRefNet` |
| `--image` | Input image path | Required |
| `--output` | Output image path | `birefnet_output.png` |
| `--size` | Input size for the model | `1024` |
| `--mask-only` | Output grayscale mask only | `false` |
| `--cpu` | Run on CPU instead of GPU | `false` |
| `--bench` | Run N iterations for benchmarking | `0` (disabled) |

## Platform Support

- CUDA: `--features cuda`
- Metal (macOS): `--features metal`
- CPU: `--cpu` flag or no GPU features

## Output Format

By default, the output is an RGBA PNG image where:
- RGB channels contain the original image pixels
- Alpha channel contains the segmentation mask

With `--mask-only`, the output is a grayscale image where:
- White (255) = foreground
- Black (0) = background
