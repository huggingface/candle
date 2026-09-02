# candle-mamba: Mamba implementation

Candle implementation of *Mamba* [1] inference only. Mamba is an alternative to
the transformer architecture. It leverages State Space Models (SSMs) with the
goal of being computationally efficient on long sequences. The implementation is
based on [mamba.rs](https://github.com/LaurentMazare/mamba.rs).

- [1]. [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752).

Compared to the mamba-minimal example, this version is far more efficient but
would only work for inference.
## Running the example

```bash
$ cargo run --example mamba --release -- --prompt "Mamba is the"
```

