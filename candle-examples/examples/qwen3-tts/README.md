# candle-qwen3-tts

Minimal Qwen3‑TTS (12Hz) inference example with batching, voice design, and voice clone.

## Usage

```bash
# Custom voice
cargo run --example qwen3-tts -r -- \
  --model-dir /path/to/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --mode custom \
  --prompt "Hello world." \
  --speaker Vivian \
  --language English \
  --out-file out.wav

# Voice design (instruct)
cargo run --example qwen3-tts -r -- \
  --model-dir /path/to/Qwen3-TTS-12Hz-0.6B-VoiceDesign \
  --mode voice-design \
  --prompt "Hello world." \
  --instruct "Warm, friendly, mid‑tempo." \
  --out-file out.wav

# Voice clone (ICL)
cargo run --example qwen3-tts -r --features symphonia, rubato -- \
  --model-dir /path/to/Qwen3-TTS-12Hz-0.6B-Base \
  --mode voice-clone \
  --prompt "Hello world." \
  --ref-audio /path/to/ref.wav \
  --ref-text "Hello, I am the reference." \
  --out-file out.wav
```

The model directory must follow the Hugging Face layout and include:

- `config.json`, `tokenizer.json`, `model.safetensors` (or `model.safetensors.index.json`)
- `speech_tokenizer/config.json` and `speech_tokenizer/model.safetensors` (or index)

Batching: repeat `--prompt` (and matching `--speaker`/`--language`/`--instruct`/`--ref-*`) or pass a single value to broadcast.
