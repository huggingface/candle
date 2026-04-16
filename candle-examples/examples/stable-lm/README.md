# candle-stable-lm

StableLM-3B-4E1T is a 3 billion parameter decoder-only language model
pre-trained on 1 trillion tokens of diverse English and code datasets for 4
epochs. See the [HuggingFace Hub Model
Card](https://huggingface.co/stabilityai/stablelm-3b-4e1t).

Note that this model is gated so you will have to request access on the Hub in
order to be able to use it.

Other available models are Stable-Code-3B, StableLM-2 and Zephyr variants.

## Running some example

```bash
$ cargo run --example stable-lm --release --features cuda -- --prompt 'What is the most efficient programming language in use?' --sample-len 150
avx: true, neon: false, simd128: false, f16c: true
temp: 0.00 repeat-penalty: 1.10 repeat-last-n: 64
retrieved the files in 126.593µs
loaded the model in 3.474148965s
What is the most efficient programming language in use?
The answer to this question depends on what you mean by "efficient". If you're talking about speed, then C++ and Java are probably your best bets. But if you're talking about ease of development, then Python is probably the way to go.
Python is a high-level, interpreted language that is easy to learn and use. It has a large community of developers who are always working on new features and improvements.
C++ is a low-level, compiled language that can be used for both desktop applications and web development. It's more difficult to learn than Python but offers greater control over the code.
Java is another high-level language that is popular with programmers because it runs on many different platforms (including Android phones
150 tokens generated (37.61 token/s)
```
