# candle-mamba3: Mamba-3 implementation

Candle implementation of _Mamba-3_ [1] inference. Mamba-3 introduces exponential-trapezoidal
discretization, complex-valued state space models with RoPE-based state updates, and
multi-input multi-output (MIMO) SSMs for improved inference efficiency.

- [1]. [Mamba-3: Improved Sequence Modeling using State Space Principles](https://arxiv.org/abs/2602.18424)

## Running the example

```bash
cargo run --example mamba3 --release -- --prompt "Mamba is the"
```

## Usage

```bash
cargo run --example mamba3 --release -- \
  --prompt "The meaning of life is" \
  --use-prefill \
  --chunk-size 64
```

## Features

- SISO (single-input single-output) and MIMO (multi-input multi-output) variants
- Chunked prefill for efficient prompt processing
- Exponential-trapezoidal discretization
- Complex-valued / RoPE state updates
- GPU support via CUDA kernels
