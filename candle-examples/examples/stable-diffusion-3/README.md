# candle-stable-diffusion-3: Candle Implementation of Stable Diffusion 3 Medium

![](assets/stable-diffusion-3.jpg)

*A cute rusty robot holding a candle torch in its hand, with glowing neon text \"LETS GO RUSTY\" displayed on its chest, bright background, high quality, 4k*

Stable Diffusion 3 Medium is a text-to-image model based on Multimodal Diffusion Transformer (MMDiT) architecture.

- [huggingface repo](https://huggingface.co/stabilityai/stable-diffusion-3-medium)
- [research paper](https://arxiv.org/pdf/2403.03206)
- [announcement blog post](https://stability.ai/news/stable-diffusion-3-medium)

## Getting access to the weights

The weights of Stable Diffusion 3 Medium is released by Stability AI under the Stability Community License. You will need to accept the conditions and acquire a license by visiting the [repo on HuggingFace Hub](https://huggingface.co/stabilityai/stable-diffusion-3-medium) to gain access to the weights for your HuggingFace account.

On the first run, the weights will be automatically downloaded from the Huggingface Hub. You might be prompted to configure a [Huggingface User Access Tokens](https://huggingface.co/docs/hub/en/security-tokens) (recommended) on your computer if you haven't done that before. After the download, the weights will be [cached](https://huggingface.co/docs/datasets/en/cache) and remain accessible locally.

## Running the model

```shell
cargo run --example stable-diffusion-3 --release --features=cuda -- \
  --height 1024 --width 1024 \
  --prompt 'A cute rusty robot holding a candle torch in its hand, with glowing neon text \"LETS GO RUSTY\" displayed on its chest, bright background, high quality, 4k'
```

To display other options available,

```shell
cargo run --example stable-diffusion-3 --release --features=cuda -- --help
```

If GPU supports, Flash-Attention is a strongly recommended feature as it can greatly improve the speed of inference, as MMDiT is a transformer model heavily depends on attentions. To utilize [candle-flash-attn](https://github.com/huggingface/candle/tree/main/candle-flash-attn) in the demo, simply add the `flash-attn` to the feature list:

```shell
cargo run --example stable-diffusion-3 --release --features=cuda,flash-attn -- ..
```

## Performance Benchmark

Below benchmark is done by generating 1024-by-1024 image from 28 steps of Euler sampling and measure the average speed (iteration per seconds).

[candle](https://github.com/huggingface/candle) and [candle-flash-attn](https://github.com/huggingface/candle/tree/main/candle-flash-attn) is based on the commit of [0d96ec3](https://github.com/huggingface/candle/commit/0d96ec31e8be03f844ed0aed636d6217dee9c7bc).

System specs (Desktop PCIE 5 x8/x8 dual-GPU setup):

- Operating System: Ubuntu 23.10
- CPU: i9 12900K w/o overclocking.
- RAM: 64G dual-channel DDR5 @ 4800 MT/s

| Speed (iter/s) | w/o flash-attn | w/ flash-attn |
| -------------- | -------------- | ------------- |
| RTX 3090 Ti    | 0.83           | 2.15          |
| RTX 4090       | 1.72           | 4.06          |
