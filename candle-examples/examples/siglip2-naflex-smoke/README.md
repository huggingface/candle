## SigLIP2 NaFlex (smoke test)

End-to-end smoke test for the NaFlex (variable-resolution) variant of SigLIP2,
[HuggingFace](https://huggingface.co/google/siglip2-base-patch16-naflex).
Loads the model, runs zero-shot classification on two images against three text prompts.

### Running

```
$ cargo run --release --features accelerate --example siglip2-naflex-smoke -- \
    --bear=path/to/bear.jpg --teddy=path/to/teddy.jpg

Device: Cpu
Model loaded.
Images shape: [2, 256, 768]
Input IDs shape: [3, 64]
logits_per_text shape: [3, 2]
Sigmoid scores (text x image):
  'a bear in the woods' -> bear=0.0052, teddy=0.0001
  'a robot holding a candle' -> bear=0.0000, teddy=0.0000
  'a group of teddy bears' -> bear=0.0000, teddy=0.4432

Per-image best match:
  bear.jpg: best='a bear in the woods' prob=0.9956
  teddy.jpg: best='a group of teddy bears' prob=0.9999
```

Image paths can also come from `SIGLIP2_BEAR_IMAGE` / `SIGLIP2_TEDDY_IMAGE`
environment variables. Build with `--features cuda` on Linux+CUDA, `--features
metal` on Apple Silicon, or pass `--cpu` to force CPU.
