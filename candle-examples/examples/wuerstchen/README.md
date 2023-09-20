# candle-wuerstchen: Efficient Pretraining of Text-to-Image Models

![anthropomorphic cat dressed as a fire fighter](./assets/cat.jpg)

The `wuerstchen` example is a port of the [diffusers
implementation](https://github.com/huggingface/diffusers/tree/19edca82f1ff194c07317369a92b470dbae97f34/src/diffusers/pipelines/wuerstchen) for Würstchen v2.
The candle implementation reproduces the same structure/files for models and
pipelines. Useful resources:

- [Official implementation](https://github.com/dome272/Wuerstchen).
- [Arxiv paper](https://arxiv.org/abs/2306.00637).
- Blog post: [Introducing Würstchen: Fast Diffusion for Image Generation](https://huggingface.co/blog/wuerstchen).

## Getting the weights

The weights are automatically downloaded for you from the [HuggingFace
Hub](https://huggingface.co/) on the first run. There are various command line
flags to use local files instead, run with `--help` to learn about them.

## Running some example.

```bash
cargo run --example wuerstchen --release --features cuda,cudnn -- \
  --prompt "Anthropomorphic cat dressed as a fire fighter"
```

The final image is named `sd_final.png` by default.
