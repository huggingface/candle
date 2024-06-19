# candle-dinov2

[Depth Anything V2] is a  model for Monocular depth estimation which builds on the [DINOv2](https://github.com/facebookresearch/dinov2) vision transformer.

This example first instantiates the DINOv2 model and then proceeds to create DepthAnythingV2 and run it. 

## Running some example

```bash
cargo run --example depth_anything_v2 --release -- --image candle-examples/examples/yolo-v8/assets/bike.jpg

```

