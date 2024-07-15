# candle-eva2

[EVA-02](https://arxiv.org/abs/2303.11331) is a computer vision model.
In this example, it is used as an ImageNet classifier: the model returns the
probability for the image to belong to each of the 1000 ImageNet categories.

## Running some example

```bash
cargo run --example eva2 --release -- --image candle-examples/examples/yolo-v8/assets/bike.jpg

> mountain bike, all-terrain bike, off-roader: 37.09%
> maillot                 : 8.30%
> alp                     : 2.13%
> bicycle-built-for-two, tandem bicycle, tandem: 0.84%
> crash helmet            : 0.73%


```

![Leading group, Giro d'Italia 2021](../yolo-v8/assets/bike.jpg)
