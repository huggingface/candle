# candle-blip

The
[blip-image-captioning](https://huggingface.co/Salesforce/blip-image-captioning-base)
model can generate captions for an input image.

## Running on an example

```bash
cargo run --example blip --release -- --image candle-examples/examples/yolo-v8/assets/bike.jpg
```

```
Running on CPU, to run on GPU, build this example with `--features cuda`
loaded image Tensor[dims 3, 384, 384; f32]
model built
several cyclists are riding down a road with cars behind them%
```
![Leading group, Giro d'Italia 2021](../yolo-v8/assets/bike.jpg)
