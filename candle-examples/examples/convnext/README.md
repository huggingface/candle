# candle-convnext

[A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) and
[ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808).

This candle implementation uses a pre-trained ConvNeXt network for inference. The
classification head has been trained on the ImageNet dataset and returns the
probabilities for the top-5 classes.

## Running an example

```
$ cargo run --example convnext --release -- --image candle-examples/examples/yolo-v8/assets/bike.jpg --which tiny

loaded image Tensor[dims 3, 224, 224; f32]
model built
mountain bike, all-terrain bike, off-roader: 84.09%
bicycle-built-for-two, tandem bicycle, tandem: 4.15%
maillot                 : 0.74%
crash helmet            : 0.54%
unicycle, monocycle     : 0.44%

```
