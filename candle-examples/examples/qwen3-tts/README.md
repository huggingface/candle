# Qwen3-TTS example

Run text-to-speech synthesis with [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-0.6B).

## Model variants

| HuggingFace repo | Variant | Notes |
|---|---|---|
| `Qwen/Qwen3-TTS-0.6B` | Base (voice cloning) | Requires reference audio |
| `Qwen/Qwen3-TTS-0.6B-CustomVoice` | CustomVoice | 9 preset speakers |
| `Qwen/Qwen3-TTS-1.7B` | Base (voice cloning) | Larger model |
| `Qwen/Qwen3-TTS-1.7B-VoiceDesign` | VoiceDesign | Text-described voice |

## Usage

### CustomVoice — preset speaker

```bash
cargo run --example qwen3-tts --release -- \
  --model-id Qwen/Qwen3-TTS-0.6B-CustomVoice \
  --text "Hello, world!" \
  --speaker ryan \
  --language english \
  --output output.wav
```

### VoiceDesign — describe a voice in text

```bash
cargo run --example qwen3-tts --release -- \
  --model-id Qwen/Qwen3-TTS-1.7B-VoiceDesign \
  --text "Good morning, how are you today?" \
  --instruct "A calm, professional female newscaster" \
  --language english \
  --output output.wav
```

### Base — voice cloning from reference audio

```bash
cargo run --example qwen3-tts --release -- \
  --model-id Qwen/Qwen3-TTS-0.6B \
  --text "Hello, this is a cloned voice." \
  --ref-audio reference.wav \
  --language english \
  --output output.wav
```

## Notes

The example downloads model weights from the HuggingFace Hub on first run.
Weights are cached in `~/.cache/huggingface/hub/`.

The speech tokenizer (`speech_tokenizer/model.safetensors`) is only needed for
ICL voice cloning (when `--ref-text` is also provided together with `--ref-audio`).
