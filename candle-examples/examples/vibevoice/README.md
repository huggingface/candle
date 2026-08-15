# VibeVoice

This example implements inference for **[VibeVoice](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B)** (Microsoft Research), a unified speech model capable of both Text-to-Speech (TTS) and Automatic Speech Recognition (ASR). It uses an ultra-low frame rate acoustic tokenizer (7.5 Hz) built on top of a Qwen2.5 LLM backbone.

## Setup

Depending on your environment, you may want to build this example with specific features to leverage your hardware:

**For CPU (AVX/Neon/Simd128):**
```bash
cargo run --release --example vibevoice --features symphonia
```

**For GPU (CUDA):**
```bash
cargo run --release --example vibevoice --features symphonia,cuda -- <args>
```

**For Metal (Apple Silicon):**
```bash
cargo run --release --example vibevoice --features symphonia,metal -- <args>
```

## Usage

### Text-to-Speech (TTS)

In TTS mode, the model generates an audio waveform from text using a provided `.pt` or `.safetensors` voice prompt (which acts as the speaker identity).

```bash
cargo run --release --example vibevoice --features symphonia,cuda -- \
    --task tts \
    --voice-prompt path/to/voice_prompt.pt \
    --prompt "Hello! This is a test of the Microsoft VibeVoice text to speech system." \
    --output output.wav
```

#### The `--low-vram` Flag
Because VibeVoice's audio components (Acoustic Tokenizer, Diffusion Head) are extremely sensitive to floating-point precision, running the **entire model** in `F32` on a GPU requires a massive amount of VRAM (often >24GB) and will trigger an Out Of Memory (OOM) error on standard consumer GPUs.

Conversely, running the entire model in `BF16` or `F16` on the GPU causes the sensitive audio components to suffer from severe rounding errors, generating pure noise.

To fix this, you can pass the `--low-vram` flag:

```bash
cargo run --release --example vibevoice --features symphonia,cuda -- \
    --task tts \
    --voice-prompt path/to/voice_prompt.pt \
    --prompt "Hello, world!" \
    --low-vram
```

When `--low-vram` is enabled:
1. The massive **Qwen2 LLM backbone** is loaded onto your GPU in **`F16`** to save VRAM and maximize speed.
2. The sensitive **Audio Components** are seamlessly offloaded to the CPU and processed in **`F32`** to ensure perfect, high-fidelity audio generation.

*If you have a massive GPU (e.g. RTX 4090/A6000), you can omit `--low-vram` to run everything strictly on the GPU in `F32`.*

### Automatic Speech Recognition (ASR)

In ASR mode, the model takes an input audio file, encodes the speech into continuous latent features, injects them into the Qwen2 LLM, and autoregressively generates the transcription text.

```bash
cargo run --release --example vibevoice --features symphonia,cuda -- \
    --task asr \
    --input path/to/audio.wav
```

*(Note: Ensure your input audio file has a compatible sample rate, generally 24kHz. The example attempts to automatically resample if necessary).*

## Command-Line Arguments

* `--task <asr|tts>`: The task to perform (default: `asr`).
* `--input <path>`: Input audio file (required for ASR mode).
* `--prompt <string>`: Text prompt (required for TTS mode).
* `--voice-prompt <path>`: A `.pt` or `.safetensors` file containing the speaker's voice prompt (required for TTS mode).
* `--output <path>`: The path where the generated audio should be saved (default: `output.wav`).
* `--low-vram`: Offloads sensitive audio components to the CPU in `F32` while keeping the LLM on the GPU in `F16` (Highly recommended for consumer GPUs).
* `--cpu`: Force execution on the CPU rather than the GPU.
* `--model-id <string>`: The model repository on HuggingFace Hub.
* `--weight-path <path>`: Path to a local model directory (overrides `--model-id`).
* `--temperature <float>`: Temperature for generation.
* `--seed <int>`: Random seed for diffusion/sampling.
