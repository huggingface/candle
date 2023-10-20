# candle-vit

Vision Transformer (ViT) model implementation following the lines of
[vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)
This uses a classification head trained on the ImageNet dataset and returns the
probabilities for the top-5 classes.

## Running an example

```
$ cargo run --example vit --release -- --image tiger.jpg

loaded image Tensor[dims 3, 224, 224; f32]
model built
tiger, Panthera tigris  : 100.00%
tiger cat               : 0.00%
jaguar, panther, Panthera onca, Felis onca: 0.00%
leopard, Panthera pardus: 0.00%
lion, king of beasts, Panthera leo: 0.00%
```
