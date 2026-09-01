# LightOnOCR-2-1B
https://huggingface.co/lightonai/LightOnOCR-2-1B

```bash
# GPU — auto-downloads model from HuggingFace
cargo run --example lightonocr_2_1B --features cuda -- --image-location ./doc.png

# CPU
cargo run --example lightonocr_2_1B -- --cpu --image-location ./doc.png

# Custom settings
cargo run --example lightonocr_2_1B --features cuda -- \
  --image-location ./doc.png \
  --max-new-tokens 512 --max-edge 1024

# Local files (no download)
cargo run --example lightonocr_2_1B --features cuda -- \
  --config ./config.json --tokenizer ./tokenizer.json --weights ./model.safetensors \
  --image-location ./doc.png
```
