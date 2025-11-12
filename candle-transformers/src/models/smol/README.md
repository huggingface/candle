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

## Related Models

### Granite-Docling
Document understanding VLM that originally used SmolLM-2 but now uses 
Granite 165M as its language backbone. See IBM's Docling project.

## References

- [SmolLM Blog Post](https://huggingface.co/blog/smollm)
- [SmolLM3 Announcement](https://huggingface.co/blog/smollm3)
- [NoPE Paper](https://arxiv.org/abs/2410.01926) - "Length Generalization of Causal Transformers without Position Encoding"
