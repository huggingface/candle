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


**With wgpu support**

You can also see the `wgpu` feature which could be interesting to get faster inference with (Vulkan, Dx12, Metal or WebGpu). [Using wgpu](../wgpu/)
