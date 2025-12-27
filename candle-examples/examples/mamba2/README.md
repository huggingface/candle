# candle-mamba2: Mamba2 implementation

Candle implementation of _Mamba2_ [1] inference. Mamba2 introduces the State Space
Duality (SSD) framework which unifies structured SSMs and attention variants.

- [1]. [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060)

## Running the example

```bash
cargo run --example mamba2 --release -- --prompt "Mamba is the"
```
