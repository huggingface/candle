# candle-gpt-oss: GPT with Mixture of Experts and Attention Sinks

GPT-OSS is a transformer model featuring Mixture of Experts (MoE) layers and attention sinks for efficient text generation. This implementation supports:

- **Mixture of Experts**: Token-level routing to specialized expert networks
- **Attention Sinks**: Special attention mechanism for improved context handling
- **Layer-wise attention types**: Mix of full attention and sliding window attention

## Running the example

```bash
$ cargo run --example gpt-oss --release -- --prompt "The future of AI is"

avx: true, neon: false, simd128: true, f16c: true
temp: 0.70 repeat-penalty: 1.10 repeat-last-n: 64
retrieved the files in 1.2s
Model: gpt2
Loaded model in 3.4s

Starting text generation...
Prompt: The future of AI is
Generated text:
==================================================
The future of AI is incredibly promising, with advances in machine learning enabling new possibilities across industries. From healthcare to autonomous vehicles, AI systems are becoming more sophisticated and capable of handling complex tasks.
==================================================
Generated 50 tokens in 2.1s (23.8 token/s)
```

```bash
$ cargo run --example gpt-oss --release --features cuda -- \
    --prompt "Write a simple function in Python" --sample-len 100

Write a simple function in Python

def greet(name):
    """
    A simple function that greets a person by name.
    
    Args:
        name (str): The name of the person to greet
        
    Returns:
        str: A greeting message
    """
    return f"Hello, {name}! Nice to meet you."

# Example usage
result = greet("Alice")
print(result)  # Output: Hello, Alice! Nice to meet you.
```