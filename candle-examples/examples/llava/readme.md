# candle-llava

implement LLaVA using candle

This example is from [candle-llava](https://github.com/chenwanqq/candle-llava)

The code is based on [https://github.com/haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA), Hence the llava-hf version of config may perform differently.

## model zoo
* [liuhaotian/LLaVA](https://huggingface.co/liuhaotian)
* [llava-hf](https://huggingface.co/llava-hf)

Right now I have tested on liuhaotian/llava-v1.6-vicuna-7b and llava-hf/llava-v1.6-vicuna-7b-hf. The memory use might have room for optimization.

## Tokenizer Setup  
The llava-hf models contain tokenizer.json, so you can just use it(with -hf flag). For original llava models, you can use the following code to generate the tokenizer.json file.

```bash  
conda create -n llava python=3.10  
pip install transformers protobuf
conda activate llava
python -c "from transformers import AutoTokenizer;tokenizer=AutoTokenizer.from_pretrained('liuhaotian/llava-v1.6-vicuna-7b');tokenizer.save_pretrained('tokenizer')"
```
Then the tokenizer.json file should be  tokenizer/tokenizer.json(which is the default path).


## eval

```bash
cargo run --example llava --features cuda -- --image-file "llava_logo.png" --prompt "is this a cat?" --hf # default args, use  llava-hf/llava-v1.6-vicuna-7b-hf. image-file is required^_^
cargo run --example llava --features cuda -- --model-path liuhaotian/llava-v1.6-vicuna-7b  --image-file "llava_logo.png" --prompt "is this a cat?" # use liuhaotian/llava-v1.6-vicuna-7b, tokenizer setup should be done
```

## Major Limitations
1. Currently only support llama-2/vicuna llm. Haven't supoort Mistral yet.
2. There are some ops like split, nonzero and where are not supported by candle.
3. Lack of quantization and LoRA support.

  

