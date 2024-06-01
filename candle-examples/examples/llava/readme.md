# candle-llava

implement LLaVA using candle

This example is from [candle-llava](https://github.com/chenwanqq/candle-llava)

The code is based on [https://github.com/haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA), Hence the llava-hf version of config may perform differently.


The llava-hf models contain tokenizer.json, so if you want pure-rust experience, I suggest you to use llava-hf version.

## model zoo
* [liuhaotian/LLaVA](https://huggingface.co/liuhaotian)
* [llava-hf](https://huggingface.co/llava-hf)

Right now I have tested on liuhaotian/llava-v1.6-vicuna-7b and llava-hf/llava-v1.6-vicuna-7b-hf. The memory use might have room for optimization.


## eval

### single-image
```bash
 cargo run --example llava --features cuda  # default args, use  llava-hf/llava-v1.6-vicuna-7b-hf, default-image is llava_logo.png, prompt is "is this a cat?"
```

## task
- [x] Download the corresponding weights from Hugging Face

- [x] Load the model weights and configs
   - [x] general llava config(need to rethink what is necessary)
   - [x] Vision tower(CLIP)
      - [x] image processor(partial, the format of 'size' and 'crop size' not fully compatible with python transformer)
   - [ ] LLM
      - [x] llama/vicuna
      - [ ] mistral  

- [x] image preprocess
   - [x] clip image processor
   - [x] 'anyres' image preprocess
   - [x] 'pad' image preprocess

- [x] conv template (partial, only implement conv_llava_v1 and conv_chatml_direct, which is enough for LLaVA v1.6)

- [x] Model structure Implementation
   - [x] Vision tower
   - [x] LLM
      - [x] modify of llama code
         - [x] output embedding result
         - [x] generate from embed tensors

- [x] model forward
   - [x] Vision tower
      - [x] feature select
   - [x] LLM
   - [x] process of multiple images
      - [x] read multiple images
      - [x] multiple images patch process
   - [x] concat of image features and text features
   - [x] truncate of the concat features

- [ ] main process
   - [x] load model
   - [x] load image
   - [x] load text
   - [x] tokenize text
   - [x] forward
      - [x] single image
   - [x] output
   - [x] KV cache
   - [ ] conversation mode
   - [ ] (long term) web?

- [ ] quantization
   - [ ] 4-bit
   - [ ] 8-bit

- [ ] (long term)  Expand candle operators, including:
   - [ ] split
   - [ ] nonzero
   - [ ] where

- [x] **top priority** migrate to support llava-hf series model
   - [x] determine whether it is a llava-hf model
   - [x] translate of config
   - [x] translate of model
   - [x] take care of constant such as image_token_index
   - [x] modify of image processor config

- [ ] LoRA
- [ ] contribution to other projects
   - [ ] [huggingface/candle](https://github.com/huggingface/candle)
   - [ ] [EricLBuehler/mistral.rs](https://github.com/EricLBuehler/mistral.rs)
- [ ] memory optimization for LLaVA 1.6 version
- [ ] (long term)model training 
c
  
## Tokenizer Setup  
```bash  
conda create -n llava python=3.10  
pip install transformers protobuf
```
## Download using mirror (for Chinese users)  
```bash
pip install -U huggingface_hub  
export HF_ENDPOINT=https://hf-mirror.com  
huggingface-cli download --resume-download liuhaotian/llava-v1.6-vicuna-7b
```
## Limitations
* Tested only on liuhaotian/llava-v1.6-vicuna-7b version
