# candle-yolo-v26: NMS-Free Object Detection with Attention

This is a port of [Ultralytics
YOLO26](https://docs.ultralytics.com/models/yolo26). YOLO26 introduces several
architectural improvements over YOLOv8:

- **C3k2 blocks** replace C2f, with configurable branch types (Bottleneck, C3k,
  or attention-based)
- **C2PSA** (Positional Scaling Attention) module with multi-head self-attention
  and DWConv positional encoding
- **NMS-free detection** via end-to-end one2one head with top-k selection
  (no non-maximum suppression needed)
- **SPPF with shortcut** and configurable pooling count

## Running some example

```bash
cargo run --example yolo-v26 --release -- path/to/image.jpg
```

This prints details about the detected objects and generates a `image.pp.jpg`
file with annotated bounding boxes.

### Command-line flags

- `--which`: select the model variant, `n`, `s`, `m`, `l`, or `x` by increasing
  size and accuracy. Default: `n`.
- `--confidence-threshold`: minimum confidence for detections. Default: `0.25`.
- `--legend-size`: the size of the label text. `0` disables labels. Default:
  `14`.
- `--model`: use a local safetensors file rather than downloading from the hub.
- `--cpu`: force CPU inference.
- `--tracing`: enable chrome tracing output.

### Model variants

| Model | Params | GFLOPs |
|-------|--------|--------|
| n     | 2.6M   | 6.1    |
| s     | 10.0M  | 22.8   |
| m     | 21.9M  | 75.4   |
| l     | 26.3M  | 93.8   |
| x     | 59.0M  | 209.5  |
