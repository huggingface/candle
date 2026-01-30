# candle-mamba2: Mamba2 implementation

Candle implementation of _Mamba2_ [1] inference. Mamba2 introduces the State Space
Duality (SSD) framework which unifies structured SSMs and attention variants.

- [1]. [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060)

## Running the example

```bash
cargo run --example mamba2 --release -- --prompt "Mamba is the"
```

## Supported models

| Model | HuggingFace ID |
|-------|----------------|
| Mamba2-130m | `AntonV/mamba2-130m-hf` |
| Mamba2-370m | `AntonV/mamba2-370m-hf` |
| Mamba2-780m | `AntonV/mamba2-780m-hf` |
| Mamba2-1.3b | `AntonV/mamba2-1.3b-hf` |
| Mamba2-2.7b | `AntonV/mamba2-2.7b-hf` |

## Verification

Outputs match the PyTorch transformers `Mamba2ForCausalLM` reference implementation.

### mamba2-130m

```bash
cargo run --example mamba2 --release -- \
  --prompt "Mamba is the" \
  --which mamba2-130m \
  --sample-len 20 \
  --repeat-penalty 1.0
```

Expected output:
```
Mamba is the most popular and popular game in the world. It is a game where you can play with your friends
```

### mamba2-370m

```bash
cargo run --example mamba2 --release -- \
  --prompt "Mamba is the" \
  --which mamba2-370m \
  --sample-len 20 \
  --repeat-penalty 1.0
```

Expected output:
```
Mamba is the first game in the series to feature a new character, the Mamba, who is a female version
```
