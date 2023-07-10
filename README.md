# candle
ML framework for Rust

```rust
let a = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
let b = Tensor::zeros((3, 4), DType::F32, &Device::Cpu)?;

let c = a.matmul(&b)?;
```

## Features

- Simple syntax (looks and like PyTorch)
- CPU and Cuda backends (and M1 support)
- Enable serverless (CPU), small and fast deployments
- Model training
- Distributed computing (NCCL).
- Models out of the box (Llama, Whisper, Falcon, ...)
- Emphasis on enabling users to use custom ops/kernels

## Structure

- [candle-core](./candle-core): Core ops, devices, and `Tensor` struct definition
- [candle-nn](./candle-nn/): Facilities to build real models
- [candle-examples](./candle-examples/): Real-world like examples on how to use the library in real settings
- [candle-kernels](./candle-kernels/): CUDA custom kernels

## How to use ?

Check out our [examples](./candle-examples/examples/):

- [Whisper](./candle-examples/examples/whisper/)
- [Llama](./candle-examples/examples/llama/)
- [Bert](./candle-examples/examples/bert/) (Useful for sentence embeddings)
- [Falcon](./candle-examples/examples/falcon/)



## FAQ

- Why Candle?

Candle stems from the need to reduce binary size in order to *enable serverless*
possible by making the whole engine smaller than PyTorch very large library volume

And simply *removing Python* from production workloads.
Python can really add overhead in more complex workflows and the [GIL](https://www.backblaze.com/blog/the-python-gil-past-present-and-future/) is a notorious source of headaches.

Rust is cool, and a lot of the HF ecosystem already has Rust crates [safetensors](https://github.com/huggingface/safetensors) and [tokenizers](https://github.com/huggingface/tokenizers).

### Missing symbols when compiling with the mkl feature.

If you get some missing symbols when compiling binaries/tests using the mkl
features, e.g.:
```
  = note: /usr/bin/ld: (....o): in function `blas::sgemm':
          .../blas-0.22.0/src/lib.rs:1944: undefined reference to `sgemm_' collect2: error: ld returned 1 exit status

  = note: some `extern` functions couldn't be found; some native libraries may need to be installed or have their path specified
  = note: use the `-l` flag to specify native libraries to link
  = note: use the `cargo:rustc-link-lib` directive to specify the native libraries to link with Cargo (see https://doc.rust-lang.org/cargo/reference/build-scripts.html#cargorustc-link-libkindname)
```

This is likely due to some missing linker flag that enable the mkl library. You
can try adding the following at the top of your binary:
```
extern crate intel_mkl_src;
```
