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

The image depicts a charming, rustic robot standing on a sandy beach at sunset.
The robot has a vintage, steampunk aesthetic with visible gears and mechanical
parts. It is holding a small lantern in one hand, which emits a warm glow, and
its other arm is extended forward as if reaching out or guiding the way. The
robot's body is adorned with the word "RUST" in bright orange letters, adding to
its rustic theme.

The background features a dramatic sky filled with clouds, illuminated by the
setting sun, casting a golden hue over the scene. Gentle waves lap against the
shore, creating a serene and picturesque atmosphere. The overall mood of the
image is whimsical and nostalgic, evoking a sense of adventure and tranquility.
```
