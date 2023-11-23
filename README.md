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

For an example to run a simple matrix multiplication with CPU or CUDA, see [**this**](https://huggingface.github.io/candle/guide/matmul.html)
example. For more advanced examples, please have a look at the following section.

## Check out our examples

These online demos run entirely in your browser:
- [yolo](https://huggingface.co/spaces/lmz/candle-yolo): pose estimation and
  object recognition.
- [whisper](https://huggingface.co/spaces/lmz/candle-whisper): speech recognition.
- [LLaMA2](https://huggingface.co/spaces/lmz/candle-llama2): text generation.
- [T5](https://huggingface.co/spaces/radames/Candle-T5-Generation-Wasm): text generation.
- [Phi-v1.5](https://huggingface.co/spaces/radames/Candle-Phi-1.5-Wasm): text generation.
- [Segment Anything Model](https://huggingface.co/spaces/radames/candle-segment-anything-wasm): Image segmentation.
- [BLIP](https://huggingface.co/spaces/radames/Candle-BLIP-Image-Captioning): image captioning.

We also provide some command line based examples using state of the art models, including:

- [LLaMA and LLaMA-v2](./candle-examples/examples/llama/): general LLM.
- [Falcon](./candle-examples/examples/falcon/): general LLM.
- [Phi-v1 and Phi-v1.5](./candle-examples/examples/phi/): a 1.3b general LLM with performance on par with LLaMA-v2 7b.
- [StableLM-3B-4E1T](./candle-examples/examples/stable-lm/): a 3b general LLM
  pre-trained on 1T tokens of English and code datasets.
- [Mistral7b-v0.1](./candle-examples/examples/mistral/): a 7b general LLM with
  performance larger than all publicly available 13b models as of 2023-09-28.
- [StarCoder](./candle-examples/examples/bigcode/): LLM specialized to code generation.
- [Replit-code-v1.5](./candle-examples/examples/replit-code/): a 3.3b LLM specialized for code completion.
- [Yi-6B / Yi-34B](./candle-examples/examples/yi/): two bilingual
  (English/Chinese) general LLMs with 6b and 34b parameters.
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

See [**this guide**](https://huggingface.github.io/candle/guide/examples.html) for how to run these examples.

## Useful External Resources
- [`candle-tutorial`](https://github.com/ToluClassics/candle-tutorial): A
  very detailed tutorial showing how to convert a PyTorch model to Candle.
- [`candle-lora`](https://github.com/EricLBuehler/candle-lora): Efficient and ergonomic LoRA implementation for Candle. `candle-lora` has      
  out-of-the-box LoRA support for many models from Candle, which can be found [here](https://github.com/EricLBuehler/candle-lora/tree/master/candle-lora-transformers/examples).
- [`optimisers`](https://github.com/KGrewal1/optimisers): A collection of optimisers
  including SGD with momentum, AdaGrad, AdaDelta, AdaMax, NAdam, RAdam, and RMSprop.
- [`candle-vllm`](https://github.com/EricLBuehler/candle-vllm): Efficient platform for inference and
  serving local LLMs including an OpenAI compatible API server.
- [`candle-ext`](https://github.com/mokeyish/candle-ext): An extension library to Candle that provides PyTorch functions not currently available in Candle.
- [`kalosm`](https://github.com/floneum/floneum/tree/master/interfaces/kalosm): A multi-modal meta-framework in Rust for interfacing with local pre-trained models with support for controlled generation, custom samplers, in-memory vector databases, audio transcription, and more.
- [`candle-sampling`](https://github.com/EricLBuehler/candle-sampling): Sampling techniques for Candle.

If you have an addition to this list, please submit a pull request.

## Features

- **PyTorch-like look and feel**
- **WASM support**: Run your models in a browser.
- **Variety of included models**: Language models, quantized support, text to image, image to text, computer vision and more!
- **File formats**: Load models from a variety of formats including safetensors, npz, ggml, and PyTorch.
- **Quantization**: Using the llama.cpp quantized types.

Please see a full feature list [**here**](https://huggingface.github.io/candle/guide/examples.html).

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


## FAQ

See the full FAQ [**here**](https://huggingface.github.io/candle/guide/faq.html).


### Why should I use Candle?

Candle's core goal is to *make serverless inference possible*. Full machine learning frameworks like PyTorch
are very large, which makes creating instances on a cluster slow. Candle allows deployment of lightweight
binaries.

Secondly, Candle lets you *remove Python* from production workloads. Python overhead can seriously hurt performance,
and the [GIL](https://www.backblaze.com/blog/the-python-gil-past-present-and-future/) is a notorious source of headaches.

Finally, Rust is cool! A lot of the HF ecosystem already has Rust crates, like [safetensors](https://github.com/huggingface/safetensors) and [tokenizers](https://github.com/huggingface/tokenizers).


### Common Errors

See [**this page**](https://huggingface.github.io/candle/guide/common_errors.html) for some common errors and fixes.
