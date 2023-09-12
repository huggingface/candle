# candle-dinov2

[DINOv2](https://github.com/facebookresearch/dinov2) is a computer vision model.
In this example, it is used as an ImageNet classifier: the model returns the
probability for the image to belong to each of the 1000 ImageNet categories.

## Running some example

```bash
cargo run --example dinov2 --release -- --image candle-examples/examples/yolo-v8/assets/bike.jpg

> mountain bike, all-terrain bike, off-roader: 43.67%
> bicycle-built-for-two, tandem bicycle, tandem: 33.20%
> crash helmet            : 13.23%
> unicycle, monocycle     : 2.44%
> maillot                 : 2.42%
```

![Leading group, Giro d'Italia 2021](../yolo-v8/assets/bike.jpg)
