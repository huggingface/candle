# hiera

[Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles](https://arxiv.org/abs/2306.00989)
This candle implementation uses pre-trained Hiera models from timm for inference.
The classification head has been trained on the ImageNet dataset and returns the probabilities for the top-5 classes.

## Running an example

```
$ cargo run --example  hiera --release  -- --image candle-examples/examples/yolo-v8/assets/bike.jpg --which tiny
loaded image Tensor[dims 3, 224, 224; f32]
model built
mountain bike, all-terrain bike, off-roader: 71.15%
unicycle, monocycle     : 7.11%
knee pad                : 4.26%
crash helmet            : 1.48%
moped                   : 1.07%
```
