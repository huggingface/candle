# candle-quantized-glm4

Candle implementation of various quantized GLM4-0414 models.

## Running an example

Run local gguf file (with local tokenizer.json)

```bash
$ cargo run --example quantized-glm4 --release --features cuda -- --tokenizer /home/data/GLM-4-9B-0414/tokenizer.json --model /home/data/GLM-4-9B-0414-Q4_K_M.gguf  --prompt "How are you today?"
```

Run local gguf file with tokenizer.json downloaded form huggingface

```bash
$ cargo run --example quantized-glm4 --release --features cuda -- --which q4k9b --model /home/data/GLM-4-9B-0414-Q4_K_M.gguf  --prompt "How are you today?"
```


Run with model-id (download from huggingface)

```bash
$ cargo run --example quantized-glm4 --release --features cuda -- --which q4k9b  --prompt "How are you today?"
```

Options for `which` [q2k9b, q2k32b, q4k9b, q4k32b]

Example output:

```
avx: true, neon: false, simd128: false, f16c: true
temp: 0.80 repeat-penalty: 1.10 repeat-last-n: 64
loaded 523 tensors (6.16GB) in 0.86s
model built

I'm just a computer program, so I don't have feelings or emotions. However, I'm functioning well and ready to assist you with any questions or tasks you might have. How can I help you today?

  10 prompt tokens processed: 67.12 token/s
  44 tokens generated: 45.28 token/s
```