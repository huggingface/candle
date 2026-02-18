## candle-rwkv

The [RWKV model](https://wiki.rwkv.com/) is a recurrent neural network model with performance on par with transformer architectures. This example supports RWKV v5, v6, and v7 (including v7a and v7b variants).

### RWKV v7

RWKV v7 "Goose" models are available in sizes from 0.1B to 13.3B parameters.
They support 12 languages: English, Chinese, French, Spanish, German, Portuguese, Russian, Italian, Japanese, Korean, Vietnamese, and Arabic.

```bash
cargo run --example rwkv --release -- \
  --which rwkv7-g1d-0.1b \
  --prompt "The Eiffel tower is in the city of"
```

#### Sizes and Variants

Model names follow the upstream convention: `rwkv7{variant}-g{generation}{dataset}-{size}`

**Base models (rwkv7-g1d):** `rwkv7-g1d-0.1b`, `rwkv7-g1d-0.4b`, `rwkv7-g1d-1.5b`, `rwkv7-g1d-2.9b`, `rwkv7-g1d-7.2b`, `rwkv7-g1d-13.3b`

**Variants:**

| Variant | Description |
|---------|-------------|
| `rwkv7a-g1d-0.1b` | Adds **DeepEmbed** — token-dependent gating in the FFN layer for better context awareness |
| `rwkv7b-g1b-0.1b` | Adds **Deep Embedding Attention (DEA)** — a full quadratic attention mechanism alongside RWKV's linear attention (uses g1**b** dataset) |

```bash
# v7a with DeepEmbed (better at context-dependent tasks)
cargo run --example rwkv --release -- \
  --which rwkv7a-g1d-0.1b --template chat \
  --prompt "Summarize this: The quick brown fox jumps over the lazy dog."

# v7b with DEA (combines RNN efficiency with transformer-like attention)
cargo run --example rwkv --release -- \
  --which rwkv7b-g1b-0.1b --template chat \
  --prompt "What is 2+2?"
```

The base v7 models are fastest. v7a adds minimal overhead. v7b is slower but can handle tasks requiring precise token relationships.

#### Prompt templates

Use `--template` to apply RWKV's recommended prompt formats:

```bash
# Chat mode (with optional --system prompt)
cargo run --example rwkv --release -- \
  --which rwkv7-g1d-1.5b \
  --template chat \
  --system "You are a helpful assistant." \
  --prompt "What is the capital of France?"

# Think mode (for hard prompts)
cargo run --example rwkv --release -- \
  --which rwkv7-g1d-1.5b \
  --template think \
  --prompt "Solve: 23 * 47"

# Fake think (recommended for best quality)
cargo run --example rwkv --release -- \
  --which rwkv7-g1d-1.5b \
  --template fake-think \
  --prompt "Explain quantum entanglement"

# Fill-in-middle (FIM) - for G1c and newer models, works for text & code & everything
# --prompt: text before the gap (what you have so far)
# --suffix: text after the gap (known ending)
# Model generates text connecting prompt → suffix
cargo run --example rwkv --release -- \
  --which rwkv7-g1d-0.4b \
  --template fim \
  --prompt "When I was young, I only liked to" \
  --suffix "and that's how first I got interested in AI research."
```

#### Multilingual examples

```bash
# Chinese
cargo run --example rwkv --release -- \
  --which rwkv7-g1d-1.5b --template chat \
  --prompt "埃菲尔铁塔在哪个城市？"

# Japanese
cargo run --example rwkv --release -- \
  --which rwkv7-g1d-1.5b --template chat \
  --prompt "エッフェル塔はどの都市にありますか？"

# French
cargo run --example rwkv --release -- \
  --which rwkv7-g1d-1.5b --template chat \
  --prompt "Dans quelle ville se trouve la tour Eiffel?"
```

#### Sampling presets

Use `--preset` for recommended sampling configurations:

```bash
# Chat preset (default params: temp 1.0, top_p 0.5, presence 2.0, frequency 0.1)
cargo run --example rwkv --release -- \
  --which rwkv7-g1d-1.5b --template chat --preset chat \
  --prompt "Tell me about the RWKV architecture"

# Creative preset (temp 0.6, top_p 0.7, presence 2.0, frequency 0.2)
cargo run --example rwkv --release -- \
  --which rwkv7-g1d-1.5b --template chat --preset creative \
  --prompt "Write a short poem about a rainy evening"
```

Or configure parameters individually:

| Parameter | Chat | Creative | Description |
|-----------|------|----------|-------------|
| `--temperature` | 1.0 | 0.6 | Sampling temperature |
| `--top-p` | 0.5 | 0.7 | Nucleus sampling cutoff |
| `--alpha-presence` | 2.0 | 2.0 | Flat penalty for any seen token |
| `--alpha-frequency` | 0.1 | 0.2 | Penalty proportional to token count |
| `--alpha-decay` | 0.99 | 0.99 | Exponential decay of token counts per step |

#### Stop sequences

Use `--stop` to end generation when a specific text is produced:

```bash
# Stop when the model tries to generate the next user turn
cargo run --example rwkv --release -- \
  --which rwkv7-g1d-1.5b --template chat \
  --prompt "Tell me a joke" \
  --stop "B: "
```

#### Performance options

Use `--dtype` for faster inference with half precision:

```bash
# BF16 (recommended - faster, good numerical stability)
cargo run --example rwkv --release -- \
  --which rwkv7-g1d-1.5b --dtype bf16 \
  --prompt "Hello world"

# F16 (fastest on some hardware)
cargo run --example rwkv --release -- \
  --which rwkv7-g1d-1.5b --dtype f16 \
  --prompt "Hello world"
```

For GPU acceleration, enable the appropriate feature:

```bash
# Apple Silicon (Metal)
cargo run --example rwkv --release --features metal -- \
  --which rwkv7-g1d-1.5b --dtype bf16 --prompt "Hello"

# NVIDIA GPU (CUDA)
cargo run --example rwkv --release --features cuda -- \
  --which rwkv7-g1d-1.5b --dtype bf16 --prompt "Hello"
```

### RWKV v5/v6

Older (depreciated) models are also supported, including
Eagle 7B ([blog post](https://blog.rwkv.com/p/eagle-7b-soaring-past-transformers)):

```bash
cargo run --example rwkv --release -- \
  --which eagle7b \
  --prompt "The smallest prime is "

cargo run --example rwkv --release -- \
  --which world6-1b6 \
  --prompt "The smallest prime is "
```