# Kokoro-82M TTS

[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) is a compact (82M parameter) text-to-speech model that achieves quality rivaling much larger models. It ranked #1 on the TTS Arena.

## Architecture

```
Text → G2P → Phoneme IDs
                   │
            PLBERT encoder (12-layer ALBERT, weight-shared)
                   │
         ProsodyPredictor (duration + F0 + energy)
                   │
          TextEncoder (conv + BiLSTM)
                   │
            ISTFTNet Decoder
                   │
              24 kHz PCM audio
```

## Usage

```bash
# Basic synthesis
cargo run --example kokoro -- \
    --text "こんにちは、元気ですか？" \
    --voice jf_alpha \
    --language ja \
    --output hello.wav

# English with a different voice
cargo run --example kokoro -- \
    --text "Hello, this is Kokoro speaking." \
    --voice af_heart \
    --output english.wav

# With Metal acceleration (macOS)
cargo run --example kokoro --features metal -- \
    --text "Bonjour le monde" \
    --voice ff_siwis \
    --language fr
```

## Available Voices

| Code | Language | Gender |
|------|----------|--------|
| `jf_alpha`, `jf_nezumi`, `jf_tebukuro` | Japanese | Female |
| `jm_kumo` | Japanese | Male |
| `af_heart`, `af_bella`, `af_sarah` | English (American) | Female |
| `am_adam`, `am_michael` | English (American) | Male |
| `bf_emma`, `bf_alice` | English (British) | Female |
| `ff_siwis` | French | Female |
| `zf_xiaoxiao`, `zf_xiaoni` | Chinese | Female |

## Notes

- Model downloads ~330 MB of weights from HuggingFace on first run.
- Weights are cached in `~/.cache/huggingface/hub/`.
- For production G2P (grapheme-to-phoneme), integrate the `misaki` library or a Rust equivalent.
  The built-in G2P in this example is character-level only (works for languages where characters ≈ phonemes).
- On Apple Silicon, pass `--features metal` for GPU acceleration.
