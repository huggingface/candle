# candle-olmo: Open Language Models designed to enable the science of language models

OLMo is a series of Open Language Models designed to enable the science of language models.

- **Project Page:** https://allenai.org/olmo
- **Papers:** [OLMo](https://arxiv.org/abs/2402.00838) [OLMo 2](https://arxiv.org/abs/2501.00656)
- **Technical blog post:** https://blog.allenai.org/olmo-open-language-model-87ccfc95f580
- **W&B Logs:** https://wandb.ai/ai2-llm/OLMo-1B/reports/OLMo-1B--Vmlldzo2NzY1Njk1
<!-- - **Press release:** TODO -->

## Running the example

```bash
$ cargo run --example olmo --release  -- --prompt "It is only with the heart that one can see rightly"

avx: true, neon: false, simd128: false, f16c: true
temp: 0.20 repeat-penalty: 1.10 repeat-last-n: 64
retrieved the files in 354.977µs
loaded the model in 19.87779666s
It is only with the heart that one can see rightly; what is essential is invisible to the eye.
```

Various model sizes are available via the `--model` argument.

```bash
$ cargo run --example olmo --release  -- --model 1.7-7b --prompt 'It is only with the heart that one can see rightly'

avx: true, neon: false, simd128: false, f16c: true
temp: 0.20 repeat-penalty: 1.10 repeat-last-n: 64
retrieved the files in 1.226087ms
loaded the model in 171.274578609s
It is only with the heart that one can see rightly; what is essential is invisible to the eye.”
~ Antoine de Saint-Exupery, The Little Prince
I am a big fan of this quote. It reminds me that I need to be open and aware of my surroundings in order to truly appreciate them.
```

