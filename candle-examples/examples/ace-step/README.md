# candle-ace-step: music generation with ACE-Step 1.5

Candle implementation of [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5),
a music generation foundation model based on a Diffusion Transformer (DiT)
architecture with text conditioning.

- [HuggingFace Model Hub](https://huggingface.co/ACE-Step)
- [GitHub Repository](https://github.com/ace-step/ACE-Step-1.5)
- [Technical Report](https://arxiv.org/abs/2602.00744)

ACE-Step generates stereo 48kHz music audio from text prompts and optional
lyrics. Two inference modes are supported:

- **DiT-only** (`--infer-type dit`): text encoder → DiT → VAE. Best with
  base/sft models.
- **LM+DiT** (`--infer-type lm-dit`): a Qwen3 language model first generates
  chain-of-thought metadata and audio codes, then the DiT produces audio
  conditioned on these codes. Best with turbo models for higher quality.
  LM metadata (BPM, keyscale, caption, language) is fed back into the DiT
  text encoder for richer conditioning.

## Supported models

| Model | Params | Steps | HuggingFace repo |
|-------|--------|-------|------------------|
| Base | 2B | 50 | [`ACE-Step/acestep-v15-base`](https://huggingface.co/ACE-Step/acestep-v15-base) |
| SFT | 2B | 50 | [`ACE-Step/acestep-v15-sft`](https://huggingface.co/ACE-Step/acestep-v15-sft) |
| Turbo (shift3) | 2B | 8 | [`ACE-Step/acestep-v15-turbo-shift3`](https://huggingface.co/ACE-Step/acestep-v15-turbo-shift3) |
| XL Base | 4B | 50 | [`ACE-Step/acestep-v15-xl-base`](https://huggingface.co/ACE-Step/acestep-v15-xl-base) |
| XL SFT | 4B | 50 | [`ACE-Step/acestep-v15-xl-sft`](https://huggingface.co/ACE-Step/acestep-v15-xl-sft) |
| XL Turbo | 4B | 8 | [`ACE-Step/acestep-v15-xl-turbo`](https://huggingface.co/ACE-Step/acestep-v15-xl-turbo) |

The **SFT** (supervised fine-tuned) variant is recommended for best quality
in DiT-only mode. **Turbo** models are best used with the LM+DiT pipeline.

### Language models (for LM+DiT mode)

| Model | Params | HuggingFace repo |
|-------|--------|------------------|
| LM 0.6B | 0.6B | [`ACE-Step/acestep-5Hz-lm-0.6B`](https://huggingface.co/ACE-Step/acestep-5Hz-lm-0.6B) |
| LM 1.7B | 1.7B | [`ACE-Step/acestep-5Hz-lm-1.7B`](https://huggingface.co/ACE-Step/acestep-5Hz-lm-1.7B) |
| LM 4B | 4B | [`ACE-Step/acestep-5Hz-lm-4B`](https://huggingface.co/ACE-Step/acestep-5Hz-lm-4B) |

The LM is automatically unloaded after generating audio codes to free GPU
memory before the DiT denoising pass.

See the full [ACE-Step model collection](https://huggingface.co/ACE-Step) for
additional variants.

## Running the model

Generate a 10-second music clip on Metal (macOS):

```bash
cargo run --example ace-step --release --features metal -- \
    --prompt "pop dance electronic synthesizer. energetic, happy, party" \
    --duration 10 --model ACE-Step/acestep-v15-sft \
    --out-file output.wav
```

On CUDA:

```bash
cargo run --example ace-step --release --features cuda -- \
    --prompt "acoustic guitar melody with soft piano accompaniment" \
    --duration 15 --model ACE-Step/acestep-v15-sft \
    --out-file output.wav
```

On CPU (slower):

```bash
cargo run --example ace-step --release -- \
    --prompt "jazz piano solo" --duration 10 --cpu \
    --model ACE-Step/acestep-v15-sft --out-file output.wav
```

With the turbo model (8 steps, faster):

```bash
cargo run --example ace-step --release --features metal -- \
    --prompt "cinematic orchestral epic" --duration 10 \
    --model ACE-Step/acestep-v15-turbo-shift3 --out-file output.wav
```

### Lyrics

Pass lyrics as a string or as a path to a text file:

```bash
# Inline lyrics
cargo run --example ace-step --release --features metal -- \
    --prompt "pop ballad" --lyrics "[Verse]\nHello world\n[Chorus]\nLa la la" \
    --duration 30

# From file
cargo run --example ace-step --release --features metal -- \
    --prompt "pop ballad" --lyrics lyrics.txt \
    --duration 30
```

If the `--lyrics` value is a path to an existing file, its contents are loaded
automatically. Otherwise it is used as literal lyrics text.

### LM+DiT pipeline (recommended for turbo models)

The LM generates chain-of-thought metadata (BPM, key, style, caption,
language) and audio codes that condition the DiT for higher quality
generation. The metadata is fed back into the DiT text prompt:

```bash
cargo run --example ace-step --release --features metal -- \
    --infer-type lm-dit \
    --lm-model ACE-Step/acestep-5Hz-lm-0.6B \
    --model ACE-Step/acestep-v15-turbo-shift3 \
    --prompt "pop dance electronic synthesizer, energetic" \
    --duration 30 --out-file output.wav
```

With custom LM sampling parameters:

```bash
cargo run --example ace-step --release --features metal -- \
    --infer-type lm-dit \
    --lm-model ACE-Step/acestep-5Hz-lm-0.6B \
    --model ACE-Step/acestep-v15-turbo-shift3 \
    --prompt "jazz piano trio" --lyrics lyrics.txt \
    --temperature 0.9 --top-p 0.95 \
    --duration 15 --seed 42
```

### Cover mode (reference audio conditioning)

Provide a reference WAV file to guide the generation style:

```bash
cargo run --example ace-step --release --features metal -- \
    --prompt "electronic remix" \
    --reference-audio original_song.wav \
    --model ACE-Step/acestep-v15-sft \
    --duration 30 --out-file cover.wav
```

Use `--timbre-audio` to supply a separate timbre source (matching Python's
`reference_audio` vs `src_audio` separation). Without it, the reference
audio is used for both structure and timbre:

```bash
cargo run --example ace-step --release --features metal -- \
    --prompt "electronic remix" \
    --reference-audio melody_source.wav \
    --timbre-audio singer_voice.wav \
    --infer-type lm-dit --lm-model ACE-Step/acestep-5Hz-lm-0.6B \
    --model ACE-Step/acestep-v15-turbo-shift3 \
    --duration 30 --out-file cover.wav
```

Use `--audio-cover-strength` (0.0-1.0) to control how much of the
denoising process uses the cover condition before switching to text-only:

```bash
cargo run --example ace-step --release --features metal -- \
    --reference-audio original.wav --audio-cover-strength 0.7 \
    --prompt "jazz remix" --duration 30
```

For best cover quality, combine with LM mode
(`--infer-type lm-dit --lm-model ACE-Step/acestep-5Hz-lm-0.6B`).

### Repaint mode (selective re-generation)

Regenerate a specific time region while preserving the rest:

```bash
cargo run --example ace-step --release --features metal -- \
    --repaint-audio source.wav --repaint-start 3 --repaint-end 7 \
    --prompt "energetic drum solo" --duration 10 --out-file repainted.wav
```

The original audio outside `[3s, 7s]` is preserved; only the specified
region is regenerated according to the text prompt.

## Options

```
--prompt               Text describing the desired music style
--lyrics               Lyrics text or path to a lyrics file (default: instrumental)
--duration             Audio duration in seconds (default: 10)
--model                HuggingFace DiT model repo (default: ACE-Step/acestep-v15-sft)
--num-steps            Denoising steps (default: 50, auto 8 for turbo)
--guidance-scale       CFG/APG guidance strength (default: 5.0)
--shift                Timestep schedule shift factor (default: 1.0)
--seed                 Random seed for reproducibility
--out-file             Output WAV file path (default: ace_step_output.wav)
--cpu                  Force CPU execution
--tracing              Enable tracing profiler
--infer-type           "dit" (default) or "lm-dit" (LM+DiT pipeline)
--lm-model             HuggingFace LM repo (for lm-dit mode)
--temperature          LM sampling temperature (default: 0.85)
--top-p                LM top-p nucleus sampling threshold
--reference-audio      WAV file for cover mode — structural source (melody/rhythm)
--timbre-audio         Separate WAV for timbre; if omitted, uses reference-audio
--repaint-audio        WAV file for repaint mode (selective re-generation)
--repaint-start        Repaint region start in seconds (default: 0)
--repaint-end          Repaint region end in seconds
--audio-cover-strength Cover condition fraction (0.0-1.0, default: 1.0)
--cfg-interval-start   CFG guidance applied when timestep >= this (default: 0.0)
--cfg-interval-end     CFG guidance applied when timestep <= this (default: 1.0)
--infer-method         "ode" (deterministic, default) or "sde" (stochastic)
--vae-chunk-frames     Latent frames per VAE-decode chunk; 0 = built-in default (128 ≈ 5.1s of audio)
--vae-chunk-overlap    Latent-frame overlap on each side of a VAE-decode chunk; 0 = default (16)
```

## Long-duration generation: tiled VAE decoding

The VAE decoder is the biggest memory consumer after latent denoising — its
widest intermediate activation scales linearly with output length. At 30+
seconds of 48 kHz stereo this intermediate tensor alone can exceed 10–20 GB
and trigger OOM on consumer GPUs.

To work around this, the decoder runs by default in **tiled** mode: the
latent is split along the time axis into overlapping chunks, each chunk is
decoded independently, the overlap regions are trimmed, and the cores are
concatenated. Because the decoder is fully convolutional with local padding
and its receptive field is well under the default 16-frame overlap, the
stitched output is numerically equivalent to a single full-length decode.

Defaults (`--vae-chunk-frames 128 --vae-chunk-overlap 16`) cut peak decoder
memory by roughly an order of magnitude on 30 s+ tracks. If you still hit
OOM, shrink the chunk:

```bash
cargo run --example ace-step --release --features metal -- \
    --prompt "..." --duration 60 \
    --vae-chunk-frames 64 --vae-chunk-overlap 16
```

For short latents (`T_latent <= chunk`) tiling short-circuits to a plain
forward pass, so there is no overhead on small durations.

### DiT sliding-window self-attention

Half of the DiT's 24 layers are configured as `sliding_attention` with a
window of 128 positions, alternating with `full_attention` layers. On those
sliding layers the self-attention is computed in query chunks with local KV
slices of size `window + 2*window`, so the full `L×L` score matrix is never
materialized. Peak per-layer memory on the attention scores drops from
O(L²) to O(L·W) — roughly two orders of magnitude at 60 s and above.

This activates automatically when `L > sliding_window`. On shorter
sequences the window covers everything and the layer falls back to full
attention. The other half of the layers remain full-attention (they need
global context) and still scale O(L²); they are the next binding limit
past ~2-minute tracks.

Together with tiled VAE, this means the practical ceiling for generation
length is set by the full-attention DiT layers, not by the decoder or the
sliding layers.

## First run

On the first run, model weights are automatically downloaded from HuggingFace
Hub (~5GB for 2B DiT models, ~9GB for XL). Sharded models (XL DiT, LM 4B)
are handled transparently via `model.safetensors.index.json`. After download,
weights are cached locally.

When using `--infer-type lm-dit`, the LM model weights are also downloaded
(~1.3GB for 0.6B, ~3.5GB for 1.7B, ~8GB for 4B). The LM is unloaded after
generating audio codes to free memory for the DiT pass.

The model also requires a `silence_latent.pt` file which is downloaded from
the model repo and parsed automatically.

## Audio post-processing

Output audio is post-processed with:

1. **BS1770 loudness normalization** to -14 LUFS per channel with `tanh`
   compressor for consistent loudness across generations
2. **Peak clamp** to [-1, 1]
3. **50ms fade-out** at the end to avoid clicks

## Architecture overview

### DiT-only mode (default)

```
Text prompt --> Qwen3-Embedding-0.6B --> text_hidden_states
                                              |
Lyrics -----> embed_tokens --> LyricEncoder --+
                                              v
                                    ConditionEncoder
                                         |
                                         v
                              encoder_hidden_states
                                         |
Random noise                             |
     |                                   v
     +--------> DiT (24/32 layers) <-----+
     |          - Self-attention (sliding/full)
     |          - Cross-attention
     |          - AdaLN-Zero from timestep
     |          x 50 Euler ODE steps + APG
     v
Clean latent --> AutoencoderOobleck --> loudness norm --> 48kHz stereo WAV
```

### LM+DiT mode (`--infer-type lm-dit`)

```
                         Phase 1: CoT                  Phase 2: Codes
Text prompt ──> Qwen3 LM ──> <think>              ──> <|audio_code_N|>...
                              bpm: 120                       |
                              caption: ...                   v
                              duration: 30            parse code values
                              keyscale: C major              |
                              language: en                   v
                              timesignature: 4      ResidualFSQ.indices_to_codes
                              </think>                       |
                                                             v
                           [LM unloaded]            AudioTokenDetokenizer
                                                             |
                                                             v
                                                    lm_hints_25Hz (latents)
                                                             |
LM metadata ──> caption/bpm/key/language (fed back to DiT text input)
                       |
Text prompt + LM meta ──> Qwen3-Embedding --> ConditionEncoder  |
                                         |                      |
                                         v                      v
                              encoder_hidden_states + lm_hints_25Hz
                                         |
Random noise                             |
     |                                   v
     +--------> DiT (8 turbo steps) <----+
     v
Clean latent --> AutoencoderOobleck --> loudness norm --> 48kHz stereo WAV
```
