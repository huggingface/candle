# candle-mobilenetv4

[MobileNetV4 - Universal Models for the Mobile Ecosystem](https://arxiv.org/abs/2404.10518)
This candle implementation uses pre-trained MobileNetV4 models from timm for inference.
The classification head has been trained on the ImageNet dataset and returns the probabilities for the top-5 classes.

## Running an example

```
$ cargo run --example mobilenetv4 --release  -- --image candle-examples/examples/yolo-v8/assets/bike.jpg --which medium
loaded image Tensor[dims 3, 256, 256; f32]
model built
unicycle, monocycle     : 20.18%
mountain bike, all-terrain bike, off-roader: 19.77%
bicycle-built-for-two, tandem bicycle, tandem: 15.91%
crash helmet            : 1.15%
tricycle, trike, velocipede: 0.67%
```
