## VGG Model Implementation

This example demonstrates the implementation of VGG models (VGG13, VGG16, VGG19) using the Candle library.

The VGG models are defined in `candle-transformers/src/models/vgg.rs`. The main function in `candle-examples/examples/vgg/main.rs` loads an image, selects the VGG model based on the provided argument, and applies the model to the loaded image.

You can run the example with the following command:

```bash
cargo run --example vgg --release -- --image ../yolo-v8/assets/bike.jpg --which vgg13
```

In the command above, `--image` specifies the path to the image file and `--which` specifies the VGG model to use (vgg13, vgg16, or vgg19).
