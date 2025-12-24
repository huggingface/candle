# candle-z-image: Text-to-Image Generation with Flow Matching

Z-Image is a ~24B parameter text-to-image generation model developed by Alibaba,
using flow matching for high-quality image synthesis.
[ModelScope](https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo),
[HuggingFace](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo).

## Model Architecture

- **Transformer**: 24B parameter DiT with 30 main layers + 2 noise refiner + 2 context refiner
- **Text Encoder**: Qwen3-based encoder (outputs second-to-last hidden states)
- **VAE**: AutoEncoderKL with diffusers format weights
- **Scheduler**: FlowMatchEulerDiscreteScheduler with dynamic shifting

## Getting the Weights

Download the model weights from HuggingFace:

```bash
# Using huggingface-cli
huggingface-cli download Tongyi-Wanx/Z-Image-Turbo --local-dir weights/Z-Image-Turbo

# Or using git-lfs
git lfs install
git clone https://huggingface.co/Tongyi-Wanx/Z-Image-Turbo weights/Z-Image-Turbo
```

## Running the Model

### Basic Usage

```bash
cargo run --features cuda --example z_image --release -- \
    --model-path weights/Z-Image-Turbo \
    --prompt "A beautiful landscape with mountains and a lake" \
    --width 1024 --height 768 \
    --num-steps 8
```

### Using Metal (macOS)

```bash
cargo run --features metal --example z_image --release -- \
    --model-path weights/Z-Image-Turbo \
    --prompt "A futuristic city at night with neon lights" \
    --width 1024 --height 1024 \
    --num-steps 9
```

### Command-line Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--model-path` | Path to the model weights directory | `weights/Z-Image-Turbo` |
| `--prompt` | The text prompt for image generation | Required |
| `--negative-prompt` | Negative prompt for CFG guidance | `""` |
| `--width` | Width of the generated image (must be divisible by 16) | `1024` |
| `--height` | Height of the generated image (must be divisible by 16) | `1024` |
| `--num-steps` | Number of denoising steps | `9` |
| `--guidance-scale` | Classifier-free guidance scale | `5.0` |
| `--seed` | Random seed for reproducibility | Random |
| `--output` | Output image filename | `z_image_output.png` |
| `--cpu` | Use CPU instead of GPU | `false` |

## Image Size Requirements

Image dimensions **must be divisible by 16**. Valid sizes include:

- ✅ 1024×1024, 1024×768, 768×1024, 512×512, 1280×720, 1920×1088
- ❌ 1920×1080 (1080 is not divisible by 16)

If an invalid size is provided, the program will suggest valid alternatives.

## Performance Notes

- **Turbo Version**: Z-Image-Turbo is optimized for fast inference, requiring only 8-9 steps
- **Memory Usage**: The 24B model requires significant GPU memory. Reduce image dimensions if encountering OOM errors
- **Metal Backend**: Fully supported on macOS with Apple Silicon

## Example Outputs

```bash
# Landscape (16:9)
cargo run --features metal --example z_image -r -- \
    --model-path weights/Z-Image-Turbo \
    --prompt "A serene mountain lake at sunset, photorealistic, 4k" \
    --width 1280 --height 720 --num-steps 8

# Portrait (3:4)
cargo run --features metal --example z_image -r -- \
    --model-path weights/Z-Image-Turbo \
    --prompt "A portrait of a wise elderly scholar, oil painting style" \
    --width 768 --height 1024 --num-steps 9

# Square (1:1)
cargo run --features metal --example z_image -r -- \
    --model-path weights/Z-Image-Turbo \
    --prompt "A cute robot holding a candle, digital art" \
    --width 1024 --height 1024 --num-steps 8
```

## Technical Details

### Latent Space

The VAE operates with an 8× upsampling factor. Latent dimensions are calculated as:

```
latent_height = 2 × (image_height ÷ 16)
latent_width = 2 × (image_width ÷ 16)
```

### 3D RoPE Position Encoding

Z-Image uses 3D Rotary Position Embeddings with axes:
- Frame (temporal): 32 dims, max 1536 positions
- Height (spatial): 48 dims, max 512 positions
- Width (spatial): 48 dims, max 512 positions

### Dynamic Timestep Shifting

The scheduler uses dynamic shifting based on image sequence length:

```
mu = BASE_SHIFT + (image_seq_len - BASE_SEQ_LEN) / (MAX_SEQ_LEN - BASE_SEQ_LEN) × (MAX_SHIFT - BASE_SHIFT)
```

Where `BASE_SHIFT=0.5`, `MAX_SHIFT=1.15`, `BASE_SEQ_LEN=256`, `MAX_SEQ_LEN=4096`.
