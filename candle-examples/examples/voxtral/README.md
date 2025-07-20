# Voxtral Example

This example demonstrates how to use the Voxtral multimodal model for audio-to-text generation tasks.

## Overview

Voxtral is a multimodal model that combines:
- A Whisper-based audio encoder for processing audio features
- A multimodal projector to map audio embeddings to text space
- A LLaMA-based language model for text generation

The model can process audio inputs and generate contextually relevant text outputs, making it suitable for tasks like:
- Audio transcription with context
- Audio-based question answering
- Audio captioning and description
- Voice-based conversation

## Prerequisites

Before running this example, ensure you have:
1. Rust installed with cargo
2. (Optional) CUDA toolkit for GPU acceleration
3. Audio files in a supported format

## Usage

### Basic Usage

```bash
# Run with the included sample audio file
cargo run --example voxtral --features symphonia --no-default-features --release

# Or specify your own audio file
cargo run --example voxtral --features symphonia --no-default-features --release -- --audio-file your_audio.mp4
```

**Note**: Due to tokenizer compilation issues, this example currently runs in demonstration mode showing audio processing capabilities. For full model inference, you would need to set up the tokenizer dependencies properly.

### Command Line Options

- `--audio-file`: Path to the audio file to process (default: "hello.mp4")
- `--prompt`: Text prompt for generation (default: "Transcribe the following audio:")
- `--cpu`: Use CPU instead of GPU
- `--temperature`: Sampling temperature, 0 for greedy (default: 0.7)
- `--top-p`: Top-p sampling parameter
- `--max-new-tokens`: Maximum tokens to generate (default: 512)

### Examples

1. **Basic audio processing:**
   ```bash
   cargo run --example voxtral --features symphonia --no-default-features --release
   ```

2. **Custom audio file:**
   ```bash
   cargo run --example voxtral --features symphonia --no-default-features --release -- \
     --audio-file your_audio.wav
   ```

3. **CPU inference:**
   ```bash
   cargo run --example voxtral --features symphonia --no-default-features --release -- \
     --audio-file your_audio.wav \
     --cpu
   ```

4. **Custom prompt:**
   ```bash
   cargo run --example voxtral --features symphonia --no-default-features --release -- \
     --prompt "Describe the audio content:" \
     --temperature 0.8
   ```

## Model Details

### Architecture

1. **Audio Encoder**: 
   - Based on Whisper architecture
   - Processes mel-spectrogram features
   - 32 transformer layers with 1280 hidden dimensions
   - Convolutional preprocessing layers

2. **Multimodal Projector**:
   - Maps audio features to text embedding space
   - Two-layer MLP with GELU activation
   - Projects from audio intermediate size (5120) to text hidden size (3584)

3. **Language Model**:
   - LLaMA-based architecture
   - 28 layers with 3584 hidden dimensions
   - Supports long context (32k tokens)
   - Uses RoPE positional embeddings

### Audio Processing

The model expects audio features as mel-spectrograms:
- Sample rate: 16kHz
- Number of mel bins: 128
- Frame shift: 10ms (160 samples)
- Frame length: 25ms (400 samples)

For long audio files, the model supports chunked processing with overlap to maintain context across boundaries.

## Implementation Notes

### Audio Feature Extraction

Currently, the example includes a placeholder for audio loading. In production, you would:

1. Load audio using a library like `hound` or `symphonia`
2. Resample to 16kHz if needed
3. Extract mel-spectrogram features
4. Normalize according to model requirements

Example audio loading with `hound`:
```rust
use hound;

fn load_wav(path: &str) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    
    // Resample if needed
    let samples: Vec<f32> = if spec.sample_rate != 16000 {
        // Resample to 16kHz
        resample(reader.samples(), spec.sample_rate, 16000)?
    } else {
        reader.samples::<f32>()
            .collect::<Result<Vec<_>, _>>()?
    };
    
    Ok(samples)
}
```

### Memory Optimization

For processing long audio files or running on limited memory:

1. Use chunked processing for audio longer than 30 seconds
2. Enable half-precision (F16) inference with `--use-f16`
3. Adjust chunk size based on available memory
4. Use CPU inference if GPU memory is limited

### Custom Integration

To integrate Voxtral into your application:

```rust
use candle_transformers::models::voxtral::{
    VoxtralConfig, VoxtralForConditionalGeneration
};

// Load model
let model = VoxtralForConditionalGeneration::new(&config, vb)?;

// Process audio
let audio_embeds = model.get_audio_embeds(&audio_features)?;

// Generate text
let output = model.generate(
    &input_ids,
    Some(&audio_features),
    max_tokens,
    temperature,
    top_p,
    &device
)?;
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: 
   - Use smaller chunks with `--chunk-seconds`
   - Enable F16 with `--use-f16`
   - Use CPU inference with `--cpu`

2. **Slow Generation**:
   - Ensure CUDA is properly installed for GPU inference
   - Use smaller `--max-new-tokens`
   - Adjust chunk size for optimal performance

3. **Poor Quality Output**:
   - Experiment with temperature and top-p values
   - Ensure audio quality is sufficient (16kHz, clear speech)
   - Try different prompts to guide generation

## References

- [Voxtral Model Card](https://huggingface.co/fixie-ai/voxtral-16x3B)
- [Candle Framework](https://github.com/huggingface/candle)
- [Whisper Paper](https://arxiv.org/abs/2212.04356)
- [LLaMA Paper](https://arxiv.org/abs/2302.13971)