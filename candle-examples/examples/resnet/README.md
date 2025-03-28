# candle-resnet

A candle implementation of inference using a pre-trained [ResNet](https://arxiv.org/abs/1512.03385).
This uses a classification head trained on the ImageNet dataset and returns the
probabilities for the top-5 classes.

## Running an example

```
$ cargo run --example resnet --release -- --image candle-examples/examples/yolo-v8/assets/bike.jpg

loaded image Tensor[dims 3, 224, 224; f32]
model built
tiger, Panthera tigris  : 90.21%
tiger cat               : 8.93%
lion, king of beasts, Panthera leo: 0.35%
leopard, Panthera pardus: 0.16%
jaguar, panther, Panthera onca, Felis onca: 0.09%
```
