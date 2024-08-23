# candle-fastvit

[FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization](https://arxiv.org/abs/2303.14189).
This candle implementation uses a pre-trained FastViT network for inference. The
classification head has been trained on the ImageNet dataset and returns the
probabilities for the top-5 classes.

## Running an example

```
$ cargo run --example fastvit --release -- --image candle-examples/examples/yolo-v8/assets/bike.jpg --which sa12

loaded image Tensor[dims 3, 256, 256; f32]
model built
mountain bike, all-terrain bike, off-roader: 43.45%
bicycle-built-for-two, tandem bicycle, tandem: 14.16%
unicycle, monocycle     : 4.12%
crash helmet            : 2.26%
alp                     : 1.40%
```
