# Installation

**With Cuda support**:

1. First, make sure that Cuda is correctly installed.
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

2. Create a new app and add [`candle-core`](https://github.com/huggingface/candle/tree/main/candle-core) with Cuda support.

Start by creating a new cargo:

```bash
cargo new myapp
cd myapp
```

Make sure to add the `candle-core` crate with the cuda feature:

```bash
cargo add --git https://github.com/huggingface/candle.git candle-core --features "cuda"
```

Run `cargo build` to make sure everything can be correctly built.

```bash
cargo build
```

**Without Cuda support**:

Create a new app and add [`candle-core`](https://github.com/huggingface/candle/tree/main/candle-core) as follows:

```bash
cargo new myapp
cd myapp
cargo add --git https://github.com/huggingface/candle.git candle-core
```

Finally, run `cargo build` to make sure everything can be correctly built.

```bash
cargo build
```

**With mkl support**

You can also see the `mkl` feature which could be interesting to get faster inference on CPU. [Using mkl](./advanced/mkl.md)
