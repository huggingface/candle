# Parakeet (ASR)

Run Parakeet models (TDT/RNNT/CTC) with Candle.

## Usage

```bash
cargo run -p candle-examples --example parakeet -- \
  --input /path/to/audio.wav \
  --model-id mlx-community/parakeet-tdt-0.6b-v3
```

Beam search (TDT only):

```bash
cargo run -p candle-examples --example parakeet -- \
  --input /path/to/audio.wav \
  --beam-size 5
```

Chunking for long audio:

```bash
cargo run -p candle-examples --example parakeet -- \
  --input /path/to/audio.wav \
  --chunk-duration 120 \
  --overlap-duration 15
```

Notes:
- Requires `ffmpeg` in PATH for audio decoding.
