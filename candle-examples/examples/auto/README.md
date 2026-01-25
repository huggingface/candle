# AutoModelForCausalLM

Load any supported causal language model by name, without manually specifying the architecture.

## Supported Models

- `llama` (Llama 2, Llama 3, etc.)
- `mistral` (Mistral 7B, etc.)
- `phi3` (Phi-3)
- `qwen2` (Qwen2, Qwen2.5)
- `gemma` (Gemma)

## Usage

```bash
# Qwen2
cargo run --example auto --release -- \
    --model "Qwen/Qwen2-0.5B-Instruct" \
    --prompt "Hello, I am"
```
