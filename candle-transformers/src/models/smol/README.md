# SmolLM Model Family

This directory contains implementations for the SmolLM family of models
developed by HuggingFace.

## Models

### SmolLM2 (see `models/llama`)
SmolLM2 models (135M, 360M, 1.7B) use the standard Llama3 architecture 
and are implemented in `models/llama.rs`. No separate implementation 
is needed.

**Variants:**
- HuggingFaceTB/SmolLM2-135M
- HuggingFaceTB/SmolLM2-360M  
- HuggingFaceTB/SmolLM2-1.7B

### SmolLM3
SmolLM3-3B introduces NoPE (No Positional Encoding) which requires
a custom implementation in `smollm3.rs`.

**Key innovations:**
- Hybrid RoPE/NoPE (3:1 ratio - every 4th layer uses NoPE)
- GQA with 4 groups (32 attention heads, 8 KV heads)
- Very high rope_theta (5M vs typical 10k-500k)
- Long context support (64k-128k tokens)
- Thinking mode support with `<think>` tags

**Implementations:**
- `smollm3.rs` - Full precision model (safetensors)
- `quantized_smollm3.rs` - Quantized GGUF model with weight reconstruction

**Available Models:**
- HuggingFaceTB/SmolLM3-3B (Instruct-tuned)
- HuggingFaceTB/SmolLM3-3B-Base (Base model)
- unsloth/SmolLM3-3B-GGUF (Quantized: Q4_K_M, Q8_0, F16)

### SmolVLM (planned)
Vision-language model variant, to be implemented.

## Implementation Details

### NoPE Architecture
SmolLM3 uses a mixed approach to positional encoding:
```rust
pub fn should_skip_rope(&self, layer_idx: usize) -> bool {
    // Method 1: Explicit array from config
    if let Some(ref no_rope_layers) = self.no_rope_layers {
        if layer_idx < no_rope_layers.len() {
            return no_rope_layers[layer_idx] == 0;
        }
    }
    
    // Method 2: Interval pattern (SmolLM3-3B default)
    // Every 4th layer (indices 3, 7, 11, ...) skips RoPE
    if let Some(interval) = self.no_rope_layer_interval {
        return (layer_idx + 1) % interval == 0;
    }
    
    false // Default: use RoPE
}
```

### Quantized Weight Reconstruction
The quantized implementation includes special handling for Q/K weight
reconstruction to maintain compatibility with the GGUF format's
interleaved weight storage.

### Thinking Mode
SmolLM3 supports explicit reasoning with thinking tags:
- **Enabled**: `<|im_start|>assistant\n<think>\n` (model generates reasoning)
- **Disabled**: `<|im_start|>assistant\n<think>\n\n</think>\n` (skip to answer)

## Usage Example

See `examples/smollm3/main.rs` for a unified implementation that supports
both quantized and full precision models with a single codebase.

```bash
# Quantized model (recommended)
cargo run --release --example smollm3 -- \
  --model-type quantized \
  --quantization q8_0 \
  --prompt "Explain Rust's ownership system"

# Full precision model
cargo run --release --example smollm3 -- \
  --model-type full \
  --dtype f16 \
  --prompt "Write a sorting algorithm"

# Enable thinking mode
cargo run --release --example smollm3 -- \
  --thinking \
  --prompt "Solve this logic puzzle step by step"
```

## Performance Characteristics

| Model Type | Size  | Speed | Quality | Use Case |
|------------|-------|-------|---------|----------|
| Q4_K_M     | 1.9GB | Fast  | Good    | Resource-constrained |
| Q8_0       | 3.3GB | Fast  | Better  | Balanced |
| F16 (GGUF) | 6.2GB | Med   | Best    | High quality GGUF |
| F16 (Safe) | 6.2GB | Med   | Best    | Maximum quality |
| F32 (Safe) | 12GB  | Slow  | Best    | Research/debugging |

# Credits & Attribution

## SmolLM3 Model

### Developers
**HuggingFace Team (HuggingFaceTB)**

The SmolLM family of models represents cutting-edge work in efficient language models, demonstrating that small models can achieve impressive capabilities when trained on high-quality data.

### Resources
- **Model Card**: https://huggingface.co/HuggingFaceTB/SmolLM3-3B
- **Model Card (Base)**: https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base
- **Collection**: https://huggingface.co/collections/HuggingFaceTB/smollm3-6723884a9c35673e4f9b74a2
- **Blog Post**: https://huggingface.co/blog/smollm3
- **GitHub Repository**: https://github.com/huggingface/smollm
- **License**: Apache 2.0

### Key Contributors
The SmolLM project is developed by the HuggingFace team with contributions from researchers focused on efficient LLM architectures and training methods.

## NoPE Architecture

### Research Paper
**Title**: "Length Generalization of Causal Transformers without Position Encoding"

**Authors**: 
- Jie Wang (Fudan University)
- Tao Ji (Fudan University)
- Yuanbin Wu (Fudan University)
- Hang Yan (Fudan University)
- Tao Gui (Fudan University)
- Qi Zhang (Fudan University)
- Xuanjing Huang (Fudan University)
- Xiaoling Wang (Fudan University)

**Published**: NeurIPS 2024 (Thirty-Eighth Annual Conference on Neural Information Processing Systems)

**Abstract Summary**: The paper demonstrates that removing positional encoding from selected layers (NoPE - No Positional Encoding) can improve length generalization in causal transformers while maintaining or improving performance. SmolLM3 implements this with a 3:1 RoPE/NoPE ratio.

**Resources**:
- **arXiv**: https://arxiv.org/abs/2410.01926
- **Conference**: NeurIPS 2024

### Key Innovation
The hybrid approach uses:
- **RoPE layers** (75%): Standard rotary positional embeddings for local context
- **NoPE layers** (25%): No positional encoding for improved length generalization
- **Pattern**: Every 4th layer uses NoPE (layers 3, 7, 11, 15, etc.)

This architecture enables SmolLM3 to handle much longer contexts (64k-128k tokens) while maintaining efficiency.

## Quantized Models

### Unsloth
Quantized GGUF models are provided by **Unsloth**, a team focused on making LLM inference and fine-tuning more accessible.

**Resources**:
- **GGUF Repository**: https://huggingface.co/unsloth/SmolLM3-3B-GGUF
- **Available Quantizations**: Q4_K_M, Q8_0, F16
- **Website**: https://unsloth.ai/

The quantization work enables running SmolLM3 efficiently on consumer hardware with minimal quality loss.

## Implementation Credits

### This Candle Implementation
**Implemented for**: Candle ML Framework  
**Implementation Date**: Nov 2025  
**Features**:
- Full precision model (F32/F16/BF16)
- Quantized model (Q4_K_M/Q8_0/F16 GGUF)
- Unified example supporting both
- Verified against reference implementations

**Verification**:
- Full precision: Validated against HuggingFace Transformers Python implementation
- Quantized: Validated against llama.cpp implementation

### Related Tools & Frameworks

**Candle**: Minimalist ML framework in Rust by HuggingFace  
- GitHub: https://github.com/huggingface/candle

**llama.cpp**: Efficient LLM inference in C/C++  
- GitHub: https://github.com/ggerganov/llama.cpp
- Used for quantized model verification

**HuggingFace Transformers**: Reference Python implementation  
- GitHub: https://github.com/huggingface/transformers
- Used for full model verification

## Acknowledgments

Special thanks to:

1. **HuggingFace Team** - For developing SmolLM3 and making it openly available under Apache 2.0 license
2. **NoPE Researchers** - For advancing the field with novel positional encoding approaches
3. **Unsloth** - For providing optimized quantized versions
4. **Candle Contributors** - For building an excellent ML framework in Rust
5. **Open Source Community** - For tools like llama.cpp that enable verification and benchmarking

## Citation

If you use SmolLM3 in your research or applications, please cite:

### SmolLM3 Model
```bibtex
@misc{smollm3,
  title={SmolLM3},
  author={HuggingFace Team},
  year={2024},
  publisher={HuggingFace},
  howpublished={\url{https://huggingface.co/HuggingFaceTB/SmolLM3-3B}}
}
```

### NoPE Paper
```bibtex
@inproceedings{wang2024length,
  title={Length Generalization of Causal Transformers without Position Encoding},
  author={Wang, Jie and Ji, Tao and Wu, Yuanbin and Yan, Hang and Gui, Tao and Zhang, Qi and Huang, Xuanjing and Wang, Xiaoling},
  booktitle={Thirty-Eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```

### Candle Framework
```bibtex
@software{candle,
  title={Candle: Minimalist ML Framework},
  author={HuggingFace},
  year={2024},
  url={https://github.com/huggingface/candle}
}
```

## License

- **SmolLM3 Model**: Apache 2.0
- **This Implementation**: Follows Candle framework license
- **Candle Framework**: Apache 2.0 and MIT dual-licensed

## Further Reading

- **SmolLM Blog Series**: https://huggingface.co/blog/smollm and https://huggingface.co/blog/smollm3
- **Model Card Details**: https://huggingface.co/HuggingFaceTB/SmolLM3-3B
- **NoPE Paper**: https://arxiv.org/abs/2410.01926
- **Candle Documentation**: https://huggingface.github.io/candle/

---

This implementation stands on the shoulders of giants. Thank you to all the researchers, engineers, and open source contributors who make this work possible.
