# Qwen3 WASM Text Generation

A high-performance WebAssembly implementation of the Qwen3-0.6B language model running entirely in the browser. This project demonstrates efficient on-device inference using Rust, WASM, and the Candle ML framework with SIMD optimizations.

## Features

- **Pure Browser Inference**: No server required - runs 100% client-side
- **SIMD Optimized**: Leverages WebAssembly SIMD for faster inference
- **Quantized Models**: Supports Q8_0 and Q4_K_M GGUF quantization
- **Performance Profiling**: Built-in profiler for optimization analysis
- **Flexible CLI**: Automatic model downloads with progress tracking
- **Smart Caching**: Uses HuggingFace cache to avoid re-downloads

## Performance

Running on a modern CPU with WASM SIMD support:

| Quantization | Speed | Model Size | Quality |
|--------------|-------|------------|---------|
| **Q8_0** (default) | **9.5 tok/s** | ~645MB | Best |
| Q4_K_M | 6.0 tok/s | ~380MB | Good |

*Q8_0 provides superior quality with better throughput despite larger size, making it the recommended choice.*

## Requirements
### Python Dependencies
```bash
pip install huggingface-hub tqdm
```

### Build Tools
- Rust (latest stable)
- wasm-pack: `cargo install wasm-pack`

### Browser
- Modern browser with WebAssembly SIMD support (Chrome 91+, Firefox 89+, Safari 16.4+)

## Quick Start

### 1. Build the WASM Module
```bash
wasm-pack build --target web --release
```

### 2. Run the Server (Auto-downloads model)
```bash
./serve.py
```

The server will:
- Check for the model in HuggingFace cache
- Download Q8_0 model (~645MB) if not present
- Download tokenizer and config files
- Start serving at http://localhost:8080

### 3. Open Browser
Navigate to http://localhost:8080 and start generating text!

## CLI Usage

### Basic Usage
```bash
# Use default Q8_0 model
./serve.py

# Use smaller Q4_K_M model (faster download, lower quality)
./serve.py --model 0.6b-q4

# Change port
./serve.py --port 3000

# Use custom GGUF model file
./serve.py --path /path/to/custom-model.gguf
```

### Available Options
```bash
./serve.py --help
```

**Options:**
- `--model, -m`: Choose model variant (`0.6b-q8` or `0.6b-q4`)
- `--path, -p`: Path to custom GGUF model file
- `--port`: Server port (default: 8080)
- `--list-models`: Show available models and exit

### List Models
```bash
./serve.py --list-models
```

Output:
```
Available models:

  0.6b-q8:
    Size: ~645MB
    Description: 8-bit quantization (best quality)
    File: Qwen3-0.6B-Q8_0.gguf

  0.6b-q4:
    Size: ~380MB
    Description: 4-bit quantization (smaller, faster)
    File: Qwen3-0.6B-Q4_K_M.gguf
```

## Project Structure
```
.
├── src/
│   ├── lib.rs           # WASM bindings
│   ├── m.rs             # Model implementation
│   └── profiler.rs      # Performance profiler
├── index.html           # Web interface
├── serve.py             # Development server with auto-download
├── Cargo.toml           # Rust dependencies
├── .cargo/
│   └── config.toml      # WASM build config (SIMD flags)
└── pkg/                 # Generated WASM (after build)
```

## Technical Details

### WASM SIMD
The project uses WebAssembly SIMD128 instructions for accelerated matrix operations. The SIMD feature is enabled in `config.toml`:
```toml
[target.wasm32-unknown-unknown]
rustflags = [
    '-C', 'target-feature=+simd128',
]
```

### Quantization
Models use GGUF format with different quantization schemes:
- **Q8_0**: 8-bit quantization, minimal quality loss
- **Q4_K_M**: 4-bit K-quants, good balance of size and quality

### Model Architecture
- **Base Model**: Qwen3-0.6B by Alibaba Cloud's Qwen Team
- **Framework**: Candle (Rust ML framework)
- **Format**: GGUF (quantized weights)
- **Context**: Supports variable context length with KV cache

## Development

### Debug Build
```bash
wasm-pack build --target web --dev
```

### Profile Performance
Open browser console after generation to see detailed timing breakdown:
```javascript
// In browser console
showProfile()  // Print performance stats
clearProfile() // Reset profiler
updateMemory() // Check memory usage
```

## Credits

- **Qwen3 Model**: Developed by the [Qwen Team at Alibaba Cloud](https://github.com/QwenLM/Qwen)
- **Candle Framework**: Rust ML framework by Hugging Face
- **GGUF Quantization**: Models from [unsloth/Qwen3-0.6B-GGUF](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF)

## License

This implementation is provided as-is. Please refer to the original Qwen3 license for model usage terms.

## Links

- **Qwen Project**: https://github.com/QwenLM/Qwen
- **Original Model**: https://huggingface.co/Qwen/Qwen3-0.6B
- **Quantized Models**: https://huggingface.co/unsloth/Qwen3-0.6B-GGUF
- **Your GitHub**: https://github.com/DrJesseGlass

---

Built using Rust, WebAssembly, and the Candle framework