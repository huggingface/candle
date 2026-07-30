# Installation

## 1. Create a new rust app or library

```bash
cargo new myapp
cd myapp
```

## 2. Add the correct candle version

### Standard

```bash
cargo add --git https://github.com/huggingface/candle.git candle-core
```

### CUDA

First, make sure that Cuda is correctly installed.
- `nvcc --version` should print information about your Cuda compiler driver.
- `nvidia-smi --query-gpu=compute_cap --format=csv` should print your GPUs compute capability, e.g. something
like:

```bash
compute_cap
8.9
```

You can also compile the Cuda kernels for a specific compute cap using the 
`CUDA_COMPUTE_CAP=<compute cap>` environment variable.

If any of the above commands errors out, please make sure to update your Cuda version.

Add the `candle-core` crate with the cuda feature:

```bash
cargo add --git https://github.com/huggingface/candle.git candle-core --features "cuda"
```

### ROCm

ROCm is the AMD GPU backend. Make sure ROCm 6.2 or newer is installed and on
`PATH`:
- `hipcc --version` should print information about your HIP compiler driver.
- `rocminfo | grep gfx` should print your GPU architecture, e.g. `gfx1101`.

Unlike CUDA, the kernels are not compiled during `cargo build`: `hipcc` compiles
them on first use and caches the result under `~/.cache/candle-rocm`, so the
same binary runs on any architecture. Set `CANDLE_ROCM_ARCH=<arch>` if
auto-detection is unavailable or picks the wrong GPU.

Add the `candle-core` crate with the rocm feature:

```bash
cargo add --git https://github.com/huggingface/candle.git candle-core --features "rocm"
```

If MIOpen is installed, `--features "miopen"` runs convolutions through it
instead of the default im2col plus rocBLAS GEMM.

### MKL

You can also see the `mkl` feature which can get faster inference on CPU.

Add the `candle-core` crate with the mkl feature:

```bash
cargo add --git https://github.com/huggingface/candle.git candle-core --features "mkl"
```

### Metal

Metal is exclusive to MacOS.

Add the `candle-core` crate with the metal feature:

```bash
cargo add --git https://github.com/huggingface/candle.git candle-core --features "metal"
```

## 3. Building

Run `cargo build` to make sure everything can be correctly built.

```bash
cargo build
```
