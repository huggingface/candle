# candle-efficientvit

[EfﬁcientViT: Memory Efﬁcient Vision Transformer with Cascaded Group Attention](https://arxiv.org/abs/2305.07027).

This candle implementation uses a pre-trained EfficientViT (from Microsoft Research Asia) network for inference.
The classification head has been trained on the ImageNet dataset and returns the probabilities for the top-5 classes.

## Running an example

```
$ cargo run --example efficientvit --release -- --image candle-examples/examples/yolo-v8/assets/bike.jpg --which m1

loaded image Tensor[dims 3, 224, 224; f32]
model built
mountain bike, all-terrain bike, off-roader: 69.80%
unicycle, monocycle     : 13.03%
bicycle-built-for-two, tandem bicycle, tandem: 9.28%
crash helmet            : 2.25%
alp                     : 0.46%
```
