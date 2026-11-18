# candle-segment-anything-2: Segment-Anything Model 2 (SAM 2)

This example implements the single image path of
[Segment Anything 2](https://github.com/facebookresearch/sam2), the equivalent of the upstream
`SAM2ImagePredictor`. Compared to [SAM 1](../segment-anything/README.md) the backbone is a
hierarchical Hiera trunk with an FPN neck, and the mask decoder additionally predicts an object
score and fuses the high resolution encoder features into its upscaling path.

The video path (memory attention, memory encoder and the object pointer bookkeeping) is not
implemented.

## Running some examples

```bash
cargo run --example segment-anything-2 --release -- \
  --image candle-examples/examples/yolo-v8/assets/bike.jpg \
  --which tiny --point 0.6,0.6
```

Prompts are given in normalized coordinates, where `0.5,0.5` is the middle of the image:

- `--point x,y` a click that should be part of the mask, can be repeated.
- `--neg-point x,y` a click that should be part of the background, can be repeated.
- `--bbox x0,y0,x1,y1` a box prompt, which SAM 2 feeds through the point path as a pair of
  labelled corners.
- `--multimask` returns the three mask candidates rather than a single mask, the one with the
  highest predicted IoU is the one that gets saved.

`--which` selects the model size, one of `tiny`, `small`, `base-plus` and `large`. The weights are
downloaded from the hub, `--model` can be used to point at a local `sam2.1_hiera_*.pt` checkpoint
or at a safetensors file using the same tensor names.
