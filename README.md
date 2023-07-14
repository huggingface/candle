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

Cheatsheet:

|            | Using PyTorch                            | Using Candle                                                     |
|------------|------------------------------------------|------------------------------------------------------------------|
| Creation   | `torch.Tensor([[1, 2], [3, 4]])`         | `Tensor::new(`                                                   |
|            |                                          | `  &[[1f32, 2.]], [3., 4.]],`                                    |
|            |                                          | `  &Device::Cpu)?`                                               |
| Indexing   | `tensor[:, :4]`                          | `tensor.i((.., ..4))?`                                           |
| Operations | `tensor.view((2, 2))`                    | `tensor.reshape((2, 2))?`                                        |
| Operations | `a.matmul(b)`                            | `a.matmul(&b)?`                                                  |
| Arithmetic | `a + b`                                  | `&a + &b`                                                        |
| Device     | `tensor.to(device="cuda")`               | `tensor.to_device(&Device::Cuda(0))?`                            |
| Dtype      | `tensor.to(dtype=torch.float16)`         | `tensor.to_dtype(&DType::F16)?`                                  |
| Saving     | `torch.save({"A": A}, "model.bin")`      | `tensor.save_safetensors("A", "model.safetensors")?`             |
| Loading    | `weights = torch.load("model.bin")`      | TODO (see the examples for now)                                  |


Check out our [examples](./candle-examples/examples/):

- [Whisper](./candle-examples/examples/whisper/)
- [Llama](./candle-examples/examples/llama/)
- [Bert](./candle-examples/examples/bert/) (Useful for sentence embeddings)
- [Falcon](./candle-examples/examples/falcon/)



## FAQ

### Why Candle?

Candle stems from the need to reduce binary size in order to *enable serverless*
possible by making the whole engine smaller than PyTorch very large library volume.
This enables creating runtimes on a cluster much faster.

And simply *removing Python* from production workloads.
Python can really add overhead in more complex workflows and the [GIL](https://www.backblaze.com/blog/the-python-gil-past-present-and-future/) is a notorious source of headaches.

Rust is cool, and a lot of the HF ecosystem already has Rust crates [safetensors](https://github.com/huggingface/safetensors) and [tokenizers](https://github.com/huggingface/tokenizers).


### Other ML frameworks

- [dfdx](https://github.com/coreylowman/dfdx) is a formidable crate, with shapes being included
  in types preventing a lot of headaches by getting compiler to complain about shape mismatch right off the bat
  However we found that some features still require nightly and writing code can be a bit dauting for non rust experts.

  We're leveraging and contributing to other core crates for the runtime so hopefully both crates can benefit from each
  other

- [burn](https://github.com/burn-rs/burn) is a general crate that can leverage multiple backends so you can choose the best
  engine for your workload

- [tch-rs](https://github.com/LaurentMazare/tch-rs.git) Bindings to the torch library in Rust. Extremely versatile, but they 
  do bring in the entire torch library into the runtime. The main contributor of `tch-rs` is also involved in the development
  of `candle`.

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

### How to know where an error comes from.

You can set `RUST_BACKTRACE=1` to be provided with backtraces when a candle
error is generated.
