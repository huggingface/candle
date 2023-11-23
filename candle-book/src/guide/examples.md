# Example Models

## Examples
Candle provides many example models which can be run either in your browser or your command line.

These online demos run entirely in your browser:
- [yolo](https://huggingface.co/spaces/lmz/candle-yolo): pose estimation and
  object recognition.
- [whisper](https://huggingface.co/spaces/lmz/candle-whisper): speech recognition.
- [LLaMA2](https://huggingface.co/spaces/lmz/candle-llama2): text generation.
- [T5](https://huggingface.co/spaces/radames/Candle-T5-Generation-Wasm): text generation.
- [Phi-v1.5](https://huggingface.co/spaces/radames/Candle-Phi-1.5-Wasm): text generation.
- [Segment Anything Model](https://huggingface.co/spaces/radames/candle-segment-anything-wasm): Image segmentation.
- [BLIP](https://huggingface.co/spaces/radames/Candle-BLIP-Image-Captioning): image captioning.

Command line based examples using state of the art models:

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

## Running examples

These examples can be run using commands like:
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
