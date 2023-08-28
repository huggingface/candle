# candle
[![discord server](https://dcbadge.vercel.app/api/server/hugging-face-879548962464493619)](https://discord.com/channels/879548962464493619/1136218819447238726)
[![Latest version](https://img.shields.io/crates/v/candle-core.svg)](https://crates.io/crates/candle-core)
[![Documentation](https://docs.rs/candle-core/badge.svg)](https://docs.rs/candle-core)
![License](https://img.shields.io/crates/l/candle-core.svg)

Candle is a minimalist ML framework for Rust with a focus on performance (including GPU support) 
and ease of use. Try our online demos: 
[whisper](https://huggingface.co/spaces/lmz/candle-whisper),
[LLaMA2](https://huggingface.co/spaces/lmz/candle-llama2),
[yolo](https://huggingface.co/spaces/lmz/candle-yolo).

## Get started

Make sure that you have [`candle-core`](https://github.com/huggingface/candle/tree/main/candle-core) correctly installed as described in [**Installation**](https://huggingface.github.io/candle/guide/installation.html).

Let's see how to run a simple matrix multiplication.
Write the following to your `myapp/src/main.rs` file:
```rust
use candle_core::{Device, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    let a = Tensor::randn(0f32, 1., (2, 3), &device)?;
    let b = Tensor::randn(0f32, 1., (3, 4), &device)?;

    let c = a.matmul(&b)?;
    println!("{c}");
    Ok(())
}
```

`cargo run` should display a tensor of shape `Tensor[[2, 4], f32]`.


Having installed `candle` with Cuda support, simply define the `device` to be on GPU:

```diff
- let device = Device::Cpu;
+ let device = Device::new_cuda(0)?;
```

For more advanced examples, please have a look at the following section.

## Check out our examples

Check out our [examples](./candle-examples/examples/):

- [Whisper](./candle-examples/examples/whisper/): speech recognition model.
- [LLaMA and LLaMA-v2](./candle-examples/examples/llama/): general LLM.
- [Falcon](./candle-examples/examples/falcon/): general LLM.
- [Bert](./candle-examples/examples/bert/): useful for sentence embeddings.
- [StarCoder](./candle-examples/examples/bigcode/): LLM specialized to code
  generation.
- [Stable Diffusion](./candle-examples/examples/stable-diffusion/): text to
  image generative model.
- [DINOv2](./candle-examples/examples/dinov2/): computer vision model trained
  using self-supervision (can be used for imagenet classification, depth
  evaluation, segmentation).
- [Quantized LLaMA](./candle-examples/examples/quantized/): quantized version of
  the LLaMA model using the same quantization techniques as
  [llama.cpp](https://github.com/ggerganov/llama.cpp).
- [yolo-v3](./candle-examples/examples/yolo-v3/) and
  [yolo-v8](./candle-examples/examples/yolo-v8/): object detection and pose
  estimation models.
Run them using the following commands:
```
cargo run --example whisper --release
cargo run --example llama --release
cargo run --example falcon --release
cargo run --example bert --release
cargo run --example bigcode --release
cargo run --example stable-diffusion --release -- --prompt "a rusty robot holding a fire torch"
cargo run --example dinov2 --release -- --image path/to/myinput.jpg
cargo run --example quantized --release
cargo run --example yolo-v3 --release -- myimage.jpg
cargo run --example yolo-v8 --release -- myimage.jpg # for pose estimation, add --task pose 
```

In order to use **CUDA** add `--features cuda` to the example command line. If
you have cuDNN installed, use `--features cudnn` for even more speedups.

There are also some wasm examples for whisper and
[llama2.c](https://github.com/karpathy/llama2.c). You can either build them with
`trunk` or try them online:
[whisper](https://huggingface.co/spaces/lmz/candle-whisper),
[llama2](https://huggingface.co/spaces/lmz/candle-llama2).

For LLaMA2, run the following command to retrieve the weight files and start a
test server:
```bash
cd candle-wasm-examples/llama2-c
wget https://huggingface.co/spaces/lmz/candle-llama2/resolve/main/model.bin
wget https://huggingface.co/spaces/lmz/candle-llama2/resolve/main/tokenizer.json
trunk serve --release --port 8081
```
And then head over to
[http://localhost:8081/](http://localhost:8081/).

<!--- ANCHOR: features --->

## Features

- Simple syntax, looks and feels like PyTorch.
    - Model training.
    - Embed user-defined ops/kernels, such as [flash-attention v2](https://github.com/huggingface/candle/blob/89ba005962495f2bfbda286e185e9c3c7f5300a3/candle-flash-attn/src/lib.rs#L152).
- Backends.
    - Optimized CPU backend with optional MKL support for x86 and Accelerate for macs.
    - CUDA backend for efficiently running on GPUs, multiple GPU distribution via NCCL.
    - WASM support, run your models in a browser.
- Included models.
    - LLMs: LLaMA v1 and v2, Falcon, StarCoder.
    - Whisper (multi-lingual support).
    - Stable Diffusion.
    - Computer Vision: DINOv2, EfficientNet, yolo-v3, yolo-v8.
- File formats: load models from safetensors, npz, ggml, or PyTorch files.
- Serverless (on CPU), small and fast deployments.
- Quantization support using the llama.cpp quantized types.

<!--- ANCHOR_END: features --->

## How to use

<!--- ANCHOR: cheatsheet --->
Cheatsheet:

|            | Using PyTorch                            | Using Candle                                                     |
|------------|------------------------------------------|------------------------------------------------------------------|
| Creation   | `torch.Tensor([[1, 2], [3, 4]])`         | `Tensor::new(&[[1f32, 2.], [3., 4.]], &Device::Cpu)?`           |
| Creation   | `torch.zeros((2, 2))`                    | `Tensor::zeros((2, 2), DType::F32, &Device::Cpu)?`               |
| Indexing   | `tensor[:, :4]`                          | `tensor.i((.., ..4))?`                                           |
| Operations | `tensor.view((2, 2))`                    | `tensor.reshape((2, 2))?`                                        |
| Operations | `a.matmul(b)`                            | `a.matmul(&b)?`                                                  |
| Arithmetic | `a + b`                                  | `&a + &b`                                                        |
| Device     | `tensor.to(device="cuda")`               | `tensor.to_device(&Device::new_cuda(0)?)?`                            |
| Dtype      | `tensor.to(dtype=torch.float16)`         | `tensor.to_dtype(&DType::F16)?`                                  |
| Saving     | `torch.save({"A": A}, "model.bin")`      | `candle::safetensors::save(&HashMap::from([("A", A)]), "model.safetensors")?` |
| Loading    | `weights = torch.load("model.bin")`      | `candle::safetensors::load("model.safetensors", &device)`        |

<!--- ANCHOR_END: cheatsheet --->


## Structure

- [candle-core](./candle-core): Core ops, devices, and `Tensor` struct definition
- [candle-nn](./candle-nn/): Tools to build real models
- [candle-examples](./candle-examples/): Examples of using the library in realistic settings
- [candle-kernels](./candle-kernels/): CUDA custom kernels
- [candle-datasets](./candle-datasets/): Datasets and data loaders.
- [candle-transformers](./candle-transformers): transformers-related utilities.
- [candle-flash-attn](./candle-flash-attn): Flash attention v2 layer.

## FAQ

### Why should I use Candle?

Candle's core goal is to *make serverless inference possible*. Full machine learning frameworks like PyTorch
are very large, which makes creating instances on a cluster slow. Candle allows deployment of lightweight
binaries.

Secondly, Candle lets you *remove Python* from production workloads. Python overhead can seriously hurt performance,
and the [GIL](https://www.backblaze.com/blog/the-python-gil-past-present-and-future/) is a notorious source of headaches.

Finally, Rust is cool! A lot of the HF ecosystem already has Rust crates, like [safetensors](https://github.com/huggingface/safetensors) and [tokenizers](https://github.com/huggingface/tokenizers).


### Other ML frameworks

- [dfdx](https://github.com/coreylowman/dfdx) is a formidable crate, with shapes being included
  in types. This prevents a lot of headaches by getting the compiler to complain about shape mismatches right off the bat.
  However, we found that some features still require nightly, and writing code can be a bit daunting for non rust experts.

  We're leveraging and contributing to other core crates for the runtime so hopefully both crates can benefit from each
  other.

- [burn](https://github.com/burn-rs/burn) is a general crate that can leverage multiple backends so you can choose the best
  engine for your workload.

- [tch-rs](https://github.com/LaurentMazare/tch-rs.git) Bindings to the torch library in Rust. Extremely versatile, but they 
  bring in the entire torch library into the runtime. The main contributor of `tch-rs` is also involved in the development
  of `candle`.

### Common Errors

#### Missing symbols when compiling with the mkl feature.

If you get some missing symbols when compiling binaries/tests using the mkl
or accelerate features, e.g. for mkl you get:
```
  = note: /usr/bin/ld: (....o): in function `blas::sgemm':
          .../blas-0.22.0/src/lib.rs:1944: undefined reference to `sgemm_' collect2: error: ld returned 1 exit status

  = note: some `extern` functions couldn't be found; some native libraries may need to be installed or have their path specified
  = note: use the `-l` flag to specify native libraries to link
  = note: use the `cargo:rustc-link-lib` directive to specify the native libraries to link with Cargo
```
or for accelerate:
```
Undefined symbols for architecture arm64:
            "_dgemm_", referenced from:
                candle_core::accelerate::dgemm::h1b71a038552bcabe in libcandle_core...
            "_sgemm_", referenced from:
                candle_core::accelerate::sgemm::h2cf21c592cba3c47 in libcandle_core...
          ld: symbol(s) not found for architecture arm64
```

This is likely due to a missing linker flag that was needed to enable the mkl library. You
can try adding the following for mkl at the top of your binary:
```rust
extern crate intel_mkl_src;
```
or for accelerate:
```rust
extern crate accelerate_src;
```

#### Cannot run the LLaMA examples: access to source requires login credentials

```
Error: request error: https://huggingface.co/meta-llama/Llama-2-7b-hf/resolve/main/tokenizer.json: status code 401
```

This is likely because you're not permissioned for the LLaMA-v2 model. To fix
this, you have to register on the huggingface-hub, accept the [LLaMA-v2 model
conditions](https://huggingface.co/meta-llama/Llama-2-7b-hf), and set up your
authentication token. See issue
[#350](https://github.com/huggingface/candle/issues/350) for more details.

#### Tracking down errors

You can set `RUST_BACKTRACE=1` to be provided with backtraces when a candle
error is generated.
