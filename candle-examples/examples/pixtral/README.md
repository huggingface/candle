# pixtral

Pixtral-12B is a 12B text+vision model.

[Blog Post](https://mistral.ai/news/pixtral-12b/) -
[HF Model Card](https://huggingface.co/mistralai/Pixtral-12B-2409) -
[HF Community Model Card](https://huggingface.co/mistral-community/pixtral-12b).

```bash
cargo run --profile=release-with-debug --features cuda --example pixtral -- \
    --image candle-examples/examples/flux/assets/flux-robot.jpg
```

```
Describe the image.
In the center of the image shows a small, detailed robot with glowing green eyes
stands on a beach, holding a 3D. It has a lit candle, giving off to the sand,
with clouds, sea, and sky background.

The image features a small robot is predominantly orange and blue and white,
with the robot is made of metal, with a small, with clouds, giving a 3d with
human-like eyes, beach, holding a candle. Its body is detailed, appears to be
making it appears, on the sand, holding a small, with waves, giving off. sky,
back as if its with clouds, detailed robot, be it seems, metal and metal,
giving, with, human-like eyes, be on the sea, back of clouds in a candlelit sky.
```
