# candle-beit

[Beit](https://arxiv.org/abs/2106.08254) is a computer vision model.
In this example, it is used as an ImageNet classifier: the model returns the
probability for the image to belong to each of the 1000 ImageNet categories.

## Running some example

```bash
cargo run --example beit --release -- --image candle-examples/examples/yolo-v8/assets/bike.jpg

> mountain bike, all-terrain bike, off-roader: 56.16%
> bicycle-built-for-two, tandem bicycle, tandem: 3.08%
> maillot                 : 2.23%
> alp                     : 0.88%
> crash helmet            : 0.85%

```

![Leading group, Giro d'Italia 2021](../yolo-v8/assets/bike.jpg)
