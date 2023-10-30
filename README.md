# candle
[![discord server](https://dcbadge.vercel.app/api/server/hugging-face-879548962464493619)](https://discord.gg/hugging-face-879548962464493619)
[![Latest version](https://img.shields.io/crates/v/candle-core.svg)](https://crates.io/crates/candle-core)
[![Documentation](https://docs.rs/candle-core/badge.svg)](https://docs.rs/candle-core)
![License](https://img.shields.io/crates/l/candle-core.svg)

Candle is a minimalist ML framework for Rust with a focus on performance (including GPU support) 
and ease of use. Try our online demos: 
[whisper](https://huggingface.co/spaces/lmz/candle-whisper),
[LLaMA2](https://huggingface.co/spaces/lmz/candle-llama2),
[T5](https://huggingface.co/spaces/radames/Candle-T5-Generation-Wasm),
[yolo](https://huggingface.co/spaces/lmz/candle-yolo),
[Segment
Anything](https://huggingface.co/spaces/radames/candle-segment-anything-wasm).

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

These online demos run entirely in your browser:
- [yolo](https://huggingface.co/spaces/lmz/candle-yolo): pose estimation and
  object recognition.
- [whisper](https://huggingface.co/spaces/lmz/candle-whisper): text to speech.
- [LLaMA2](https://huggingface.co/spaces/lmz/candle-llama2): text generation.
- [T5](https://huggingface.co/spaces/radames/Candle-T5-Generation-Wasm): text generation.
- [Phi-v1.5](https://huggingface.co/spaces/radames/Candle-Phi-1.5-Wasm): text generation.
- [Segment Anything Model](https://huggingface.co/spaces/radames/candle-segment-anything-wasm): Image segmentation.
- [BLIP](https://huggingface.co/spaces/radames/Candle-BLIP-Image-Captioning): image captioning.

We also provide a some command line based examples using state of the art models:

- [LLaMA and LLaMA-v2](./candle-examples/examples/llama/): general LLM.
- [Falcon](./candle-examples/examples/falcon/): general LLM.
- [Phi-v1 and Phi-v1.5](./candle-examples/examples/phi/): a 1.3b general LLM with performance on par with LLaMA-v2 7b.
- [StableLM-3B-4E1T](./candle-examples/examples/stable-lm/): a 3b general LLM
  pre-trained on 1T tokens of English and code datasets.
- [Mistral7b-v0.1](./candle-examples/examples/mistral/): a 7b general LLM with
  performance larger than all publicly available 13b models as of 2023-09-28.
- [StarCoder](./candle-examples/examples/bigcode/): LLM specialized to code generation.
- [Replit-code-v1.5](./candle-examples/examples/replit-code/): a 3.3b LLM specialized for code completion.
- [Quantized LLaMA](./candle-examples/examples/quantized/): quantized version of
  the LLaMA model using the same quantization techniques as
  [llama.cpp](https://github.com/ggerganov/llama.cpp).

<img src="https://github.com/huggingface/candle/raw/main/candle-examples/examples/quantized/assets/aoc.gif" width="600">
  
- [Stable Diffusion](./candle-examples/examples/stable-diffusion/): text to
  image generative model, support for the 1.5, 2.1, and SDXL 1.0 versions.

<img src="https://github.com/huggingface/candle/raw/main/candle-examples/examples/stable-diffusion/assets/stable-diffusion-xl.jpg" width="200">

- [Wuerstchen](./candle-examples/examples/wuerstchen/): another text to
  image generative model.

<img src="https://github.com/huggingface/candle/raw/main/candle-examples/examples/wuerstchen/assets/cat.jpg" width="200">

- [yolo-v3](./candle-examples/examples/yolo-v3/) and
  [yolo-v8](./candle-examples/examples/yolo-v8/): object detection and pose
  estimation models.

<img src="https://github.com/huggingface/candle/raw/main/candle-examples/examples/yolo-v8/assets/bike.od.jpg" width="200"><img src="https://github.com/huggingface/candle/raw/main/candle-examples/examples/yolo-v8/assets/bike.pose.jpg" width="200">
- [segment-anything](./candle-examples/examples/segment-anything/): image
  segmentation model with prompt.

<img src="https://github.com/huggingface/candle/raw/main/candle-examples/examples/segment-anything/assets/sam_merged.jpg" width="200">

- [Whisper](./candle-examples/examples/whisper/): speech recognition model.
- [T5](./candle-examples/examples/t5), [Bert](./candle-examples/examples/bert/),
  [JinaBert](./candle-examples/examples/jina-bert/) : useful for sentence embeddings.
- [DINOv2](./candle-examples/examples/dinov2/): computer vision model trained
  using self-supervision (can be used for imagenet classification, depth
  evaluation, segmentation).
- [BLIP](./candle-examples/examples/blip/): image to text model, can be used to
  generate captions for an image.
- [Marian-MT](./candle-examples/examples/marian-mt/): neural machine translation
  model, generates the translated text from the input text.

Run them using commands like:
```
cargo run --example quantized --release
```

In order to use **CUDA** add `--features cuda` to the example command line. If
you have cuDNN installed, use `--features cudnn` for even more speedups.

There are also some wasm examples for whisper and
[llama2.c](https://github.com/karpathy/llama2.c). You can either build them with
`trunk` or try them online:
[whisper](https://huggingface.co/spaces/lmz/candle-whisper),
[llama2](https://huggingface.co/spaces/lmz/candle-llama2),
[T5](https://huggingface.co/spaces/radames/Candle-T5-Generation-Wasm),
[Phi-v1.5](https://huggingface.co/spaces/radames/Candle-Phi-1.5-Wasm),
[Segment Anything Model](https://huggingface.co/spaces/radames/candle-segment-anything-wasm).

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

<!--- ANCHOR: useful_libraries --->

## Useful External Resources
- [`candle-tutorial`](https://github.com/ToluClassics/candle-tutorial): a
  very detailed tutorial showing how to convert a PyTorch model to Candle.
- [`optimisers`](https://github.com/KGrewal1/optimisers): a collection of optimisers
  including SGD with momentum, AdaGrad, AdaDelta, AdaMax, NAdam, RAdam, and RMSprop.
- [`candle-lora`](https://github.com/EricLBuehler/candle-lora): a LoRA implementation
  that conforms to the official `peft` implementation.

If you have an addition to this list, please submit a pull request.

<!--- ANCHOR_END: useful_libraries --->

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
    - Language Models.
        - LLaMA v1 and v2.
        - Falcon.
        - StarCoder.
        - Phi v1.5.
        - Mistral 7b v0.1.
        - StableLM-3B-4E1T.
        - Replit-code-v1.5-3B.
        - T5.
        - Bert.
    - Whisper (multi-lingual support).
    - Text to image.
        - Stable Diffusion v1.5, v2.1, XL v1.0.
        - Wurstchen v2.
    - Image to text.
        - BLIP.
    - Text to text.
        - Marian MT (Machine Translation).
    - Computer Vision Models.
        - DINOv2, ConvMixer, EfficientNet, ResNet, ViT.
        - yolo-v3, yolo-v8.
        - Segment-Anything Model (SAM).
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

#### Missing cute/cutlass headers when compiling flash-attn

```
  In file included from kernels/flash_fwd_launch_template.h:11:0,
                   from kernels/flash_fwd_hdim224_fp16_sm80.cu:5:
  kernels/flash_fwd_kernel.h:8:10: fatal error: cute/algorithm/copy.hpp: No such file or directory
   #include <cute/algorithm/copy.hpp>
            ^~~~~~~~~~~~~~~~~~~~~~~~~
  compilation terminated.
  Error: nvcc error while compiling:
```
[cutlass](https://github.com/NVIDIA/cutlass) is provided as a git submodule so you may want to run the following command to check it in properly.
```bash
git submodule update --init
```

#### Compiling with flash-attention fails

```
/usr/include/c++/11/bits/std_function.h:530:146: error: parameter packs not expanded with ‘...’:
```

This is a bug in gcc-11 triggered by the Cuda compiler. To fix this, install a different, supported gcc version - for example gcc-10, and specify the path to the compiler in the CANDLE_NVCC_CCBIN environment variable.
```
env CANDLE_NVCC_CCBIN=/usr/lib/gcc/x86_64-linux-gnu/10 cargo ...
```

#### Linking error on windows when running rustdoc or mdbook tests

```
Couldn't compile the test.
---- .\candle-book\src\inference\hub.md - Using_the_hub::Using_in_a_real_model_ (line 50) stdout ----
error: linking with `link.exe` failed: exit code: 1181
//very long chain of linking
 = note: LINK : fatal error LNK1181: cannot open input file 'windows.0.48.5.lib'
```

Make sure you link all native libraries that might be located outside a project target, e.g., to run mdbook tests, you should run:

```
mdbook test candle-book -L .\target\debug\deps\ `
-L native=$env:USERPROFILE\.cargo\registry\src\index.crates.io-6f17d22bba15001f\windows_x86_64_msvc-0.42.2\lib `
-L native=$env:USERPROFILE\.cargo\registry\src\index.crates.io-6f17d22bba15001f\windows_x86_64_msvc-0.48.5\lib
```

#### Extremely slow model load time with WSL

This may be caused by the models being loaded from `/mnt/c`, more details on
[stackoverflow](https://stackoverflow.com/questions/68972448/why-is-wsl-extremely-slow-when-compared-with-native-windows-npm-yarn-processing).

#### Tracking down errors

You can set `RUST_BACKTRACE=1` to be provided with backtraces when a candle
error is generated.
