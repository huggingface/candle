# candle-mobileone

[MobileOne: An Improved One millisecond Mobile Backbone](https://arxiv.org/abs/2206.04040).

This candle implementation uses a pre-trained MobileOne network for inference. The
classification head has been trained on the ImageNet dataset and returns the
probabilities for the top-5 classes.

## Running an example

```
$ cargo run --example mobileone --release -- --image candle-examples/examples/yolo-v8/assets/bike.jpg --which s2

loaded image Tensor[dims 3, 224, 224; f32]
model built
mountain bike, all-terrain bike, off-roader: 79.33%
bicycle-built-for-two, tandem bicycle, tandem: 15.32%
crash helmet            : 2.58%
unicycle, monocycle     : 1.70%
alp                     : 0.21%

```
