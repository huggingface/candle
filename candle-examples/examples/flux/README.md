# candle-flux: image generation with latent rectified flow transformers

![rusty robot holding a candle](./assets/flux-robot.jpg)

Flux is a 12B rectified flow transformer capable of generating images from text
descriptions,
[huggingface](https://huggingface.co/black-forest-labs/FLUX.1-schnell),
[github](https://github.com/black-forest-labs/flux),
[blog post](https://blackforestlabs.ai/announcing-black-forest-labs/).


## Running the model

```bash
cargo run --features cuda --example flux -r -- \
    --height 1024 --width 1024 \
    --prompt "a rusty robot walking on a beach holding a small torch, the robot has the word "rust" written on it, high quality, 4k"
```

