# yolo-v5

Candle implementation of [YOLOv5](https://github.com/ultralytics/yolov5) for object detection.

## Running an example

```bash
$ cargo run --example yolo-v5 --release -- candle-examples/examples/yolo-v5/assets/bike.jpg
```

## Command line flags

- `--cpu`: Run on CPU rather than on GPU.
- `--model`: Path to the model weights in safetensors format. If not specified, the weights are downloaded from the HuggingFace hub.
- `--image`: Path to the input image on which to run object detection.
- `--which`: Select the model variant to be used, `n`, `s`, `m`, or `l` by increasing size and quality.
- `--confidence-threshold`: Confidence threshold for detected objects, default is `0.25`.
- `--nms-threshold`: IoU threshold for non-max suppression, default is `0.45`.
