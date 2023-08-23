# Installation

- **With Cuda support**:

1. First, make sure that Cuda is correctly installed.
- `nvcc --version` should print your information about your Cuda compiler driver.
- `nvidia-smi --query-gpu=compute_cap --format=csv` should print your GPUs compute capability, e.g. something
like:
```
compute_cap
8.9
```

If any of the above commands errors out, please make sure to update your Cuda version.

2. Create a new app and add [`candle-core`](https://github.com/huggingface/candle/tree/main/candle-core) with Cuda support

Start by creating a new cargo:

```bash
cargo new myapp
cd myapp
```

Make sure to add the `candle-core` crate with the cuda feature:

```
cargo add --git https://github.com/huggingface/candle.git candle-core --features "cuda"
```

Run `cargo build` to make sure everything can be correctly built.

```
cargo run
```

**Without Cuda support**:

Create a new app and add [`candle-core`](https://github.com/huggingface/candle/tree/main/candle-core) as follows:


Start by creating a new app:

```bash
cargo new myapp
cd myapp
cargo add --git https://github.com/huggingface/candle.git candle-core
```

At this point, candle will be built **without** CUDA support.
To get CUDA support use the `cuda` feature
```bash
cargo add --git https://github.com/huggingface/candle.git candle-core --features cuda
```

You can check everything works properly:

```bash
cargo build
```

**With mkl support**

You can also see the `mkl` feature which could be interesting to get faster inference on CPU. [Using mkl](./advanced/mkl.md)
