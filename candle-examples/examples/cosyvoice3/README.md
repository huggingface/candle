# CosyVoice3 Text-to-Speech

This example demonstrates text-to-speech synthesis using [CosyVoice3](https://github.com/FunAudioLLM/CosyVoice), a state-of-the-art multilingual zero-shot TTS model from FunAudioLLM.

## Model Weights

Pre-converted weights are available on Hugging Face:

**[spensercai/CosyVoice3-0.5B-Candle](https://huggingface.co/spensercai/CosyVoice3-0.5B-Candle)**

```bash
# Download using hf CLI
hf download spensercai/CosyVoice3-0.5B-Candle --local-dir weights/CosyVoice3-0.5B-Candle
```

Or manually convert from the original PyTorch weights:

```bash
python convert_weights.py \
    --input /path/to/Fun-CosyVoice3-0.5B-2512 \
    --output weights/CosyVoice3-0.5B-Candle
```

## Usage

### Basic Usage (with ONNX support)

For full voice cloning capabilities, compile with ONNX support:

```bash
cargo run --release --example cosyvoice3 --features="symphonia,onnx" -- \
    --text "你好，这是一个测试。" \
    --prompt-wav /path/to/prompt.wav \
    --model-dir weights/CosyVoice3-0.5B-Candle \
    --output output.wav
```

### With GPU Acceleration

```bash
# Metal (macOS)
cargo run --release --example cosyvoice3 --features="symphonia,onnx,metal" -- \
    --text "Hello, this is a test." \
    --prompt-wav prompt.wav \
    --model-dir weights/CosyVoice3-0.5B-Candle \
    --output output.wav

# CUDA (Linux/Windows)
cargo run --release --example cosyvoice3 --features="symphonia,onnx,cuda" -- \
    --text "Hello, this is a test." \
    --prompt-wav prompt.wav \
    --model-dir weights/CosyVoice3-0.5B-Candle \
    --output output.wav
```

### Without ONNX (using pre-extracted features)

If you don't have ONNX support, you can use pre-extracted prompt features:

```bash
cargo run --release --example cosyvoice3 --features="symphonia" -- \
    --text "Hello, this is a test." \
    --prompt-features /path/to/features.safetensors \
    --model-dir weights/CosyVoice3-0.5B-Candle \
    --output output.wav
```

### Save Extracted Features

You can save extracted features for later use or debugging:

```bash
cargo run --release --example cosyvoice3 --features="symphonia,onnx" -- \
    --text "Test" \
    --prompt-wav prompt.wav \
    --model-dir weights/CosyVoice3-0.5B-Candle \
    --save-features prompt_features.safetensors \
    --output output.wav
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--text` | Text to synthesize | (required) |
| `--prompt-wav` | Prompt audio file for voice cloning | - |
| `--prompt-features` | Pre-extracted features file | - |
| `--prompt-text` | Prompt text for zero-shot mode (see below) | "You are a helpful assistant..." |
| `--model-dir` | Model directory path | (required) |
| `--output` | Output audio file path | `output.wav` |
| `--mode` | Inference mode: `zero-shot`, `cross-lingual`, `instruct` | `zero-shot` |
| `--instruct` | Instruction text (for instruct mode) | - |
| `--speed` | Speech synthesis speed (0.5-2.0) | `1.0` |
| `--n-timesteps` | Number of CFM sampling steps | `10` |
| `--temperature` | Sampling temperature | `1.0` |
| `--top-k` | Top-k sampling | `25` |
| `--top-p` | Top-p sampling | `0.8` |
| `--seed` | Random seed | `42` |
| `--cpu` | Run on CPU | `false` |
| `--f16` | Use f16 precision (GPU only) | `false` |
| `--verbose` | Enable verbose output | `false` |
| `--text-normalize` | Enable text normalization (requires `wetext` feature) | `false` |
| `--wetext-dir` | Path to wetext FST files directory | - |
| `--remove-erhua` | Remove erhua (儿化音) during normalization | `false` |

### About `--prompt-text`

In zero-shot mode, `--prompt-text` should follow this format:

```
You are a helpful assistant.<|endofprompt|>[transcript of prompt audio]
```

- **Fixed prefix**: `You are a helpful assistant.<|endofprompt|>` - This is required by CosyVoice3's LLM
- **Transcript**: The actual text content spoken in the `--prompt-wav` audio file

**Example**: If your prompt audio says "希望你以后能够做的比我还好呦", use:

```bash
--prompt-text "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
```

**Note**: For best voice cloning quality, the transcript should accurately match the audio content. Mismatched text may result in degraded synthesis quality.

## Synthesis Modes

### Zero-Shot Voice Cloning

Clone a voice from a reference audio sample:

```bash
cargo run --release --example cosyvoice3 --features="symphonia,onnx" -- \
    --mode zero-shot \
    --text "This is synthesized speech." \
    --prompt-wav reference_voice.wav \
    --prompt-text "You are a helpful assistant.<|endofprompt|>Hello, this is my voice sample." \
    --model-dir weights/CosyVoice3-0.5B-Candle \
    --output output.wav
```

### Cross-Lingual Voice Cloning

Clone a voice across different languages:

```bash
cargo run --release --example cosyvoice3 --features="symphonia,onnx" -- \
    --mode cross-lingual \
    --text "这是跨语言合成的语音。" \
    --prompt-wav english_reference.wav \
    --model-dir weights/CosyVoice3-0.5B-Candle \
    --output output.wav
```

### Instruction-Guided Synthesis

Control speech style with instructions:

```bash
cargo run --release --example cosyvoice3 --features="symphonia,onnx" -- \
    --mode instruct \
    --text "Hello, how are you today?" \
    --instruct "Speak in a cheerful and energetic tone." \
    --prompt-wav reference.wav \
    --model-dir weights/CosyVoice3-0.5B-Candle \
    --output output.wav
```

## Helper Scripts

### Weight Conversion

Convert PyTorch weights to safetensors format:

```bash
python convert_weights.py \
    --input /path/to/Fun-CosyVoice3-0.5B-2512 \
    --output weights/CosyVoice3-0.5B-Candle
```

### Random Noise Extraction (Optional)

For exact numerical reproducibility with the Python implementation:

```bash
python extract_rand_noise.py \
    --output weights/CosyVoice3-0.5B-Candle/rand_noise.safetensors
```

Note: This file is optional. Without it, the model generates its own deterministic noise.

## Text Normalization

For better TTS quality, text normalization is recommended. It converts numbers, dates, currencies, and other non-standard text to spoken form.

### Enable Text Normalization

First, compile with the `wetext` feature:

```bash
cargo run --release --example cosyvoice3 --features="symphonia,wetext" -- \
    --text "2024年1月15日，价格是$100.50" \
    --text-normalize \
    --wetext-dir /path/to/wetext/fsts \
    --model-dir weights/CosyVoice3-0.5B-Candle \
    --output output.wav
```

### Download WeText FST Files

WeText FST files are available on Hugging Face:

```bash
# Download using hf CLI
hf download mio/wetext --local-dir weights/wetext
```

### Normalization Examples

#### Chinese (中文)

| Input | Output |
|-------|--------|
| `2024年` | `二零二四年` |
| `3.14` | `三点一四` |
| `1月15日` | `一月十五日` |
| `$100.50` | `一百美元五十美分` |
| `50%` | `百分之五十` |
| `3/4` | `四分之三` |

#### English

| Input | Output |
|-------|--------|
| `$100.50` | `one hundred dollars and fifty cents` |
| `3.14` | `three point one four` |
| `Jan. 15` | `january fifteen` |
| `don't` | `do not` |
| `I'm gonna go` | `I am going to go` |

#### Japanese (日本語)

| Input | Output |
|-------|--------|
| `2024年` | `二千二十四年` |
| `100円` | `百円` |
| `3月15日` | `三月十五日` |

### Options

- `--text-normalize`: Enable text normalization
- `--wetext-dir`: Path to wetext FST files directory (required when normalization is enabled)
- `--remove-erhua`: Remove erhua (儿化音, e.g., "哪儿" → "哪")

## Performance

| Device | RTF (Real-Time Factor) |
|--------|------------------------|
| Apple M1 Pro (Metal) | ~0.3-0.5x |
| CPU (x86_64) | ~2-4x |

*RTF < 1.0 means faster than real-time*

## Model Architecture

CosyVoice3 consists of four main components:

1. **Frontend** - Text tokenization, mel extraction, speaker embedding (ONNX)
2. **CosyVoice3LM** - Qwen2-based autoregressive LM for speech token generation
3. **Flow Decoder** - 22-layer DiT with Conditional Flow Matching
4. **HiFT Vocoder** - Neural Source Filter based vocoder

## References

- [CosyVoice Paper](https://arxiv.org/abs/2407.05407)
- [CosyVoice GitHub](https://github.com/FunAudioLLM/CosyVoice)
- [Original Weights](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
