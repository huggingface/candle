# candle-lfm2-vl: LFM2-VL (Liquid Foundation Model, vision-language)

[LFM2-VL](https://huggingface.co/LiquidAI/LFM2-VL-450M) pairs the hybrid LFM2 language model
with a SigLIP2 NaFlex vision encoder. Images are processed at their native aspect ratio:
small images are resized as a whole, large ones are split into 512x512 tiles plus a
thumbnail of the whole image.

## Running the example

```bash
cargo run --example lfm2-vl --release -- \
    --image candle-examples/examples/yolo-v8/assets/bike.jpg \
    --prompt "Describe this image."
```

> The image captures a vibrant scene of a professional bicycle race, likely the Tour de France,
> with numerous cyclists and their motorcycles in motion. The photograph is taken from the side
> of the road, showcasing a diverse group of riders, ...

On macOS with Metal, add `--features metal`; with CUDA, `--features cuda`.

## Supported Models

| `--which` | Model |
|---|---|
| `lfm2-vl-450m` | [LFM2-VL-450M](https://huggingface.co/LiquidAI/LFM2-VL-450M) |
| `lfm2-vl-1.6b` | [LFM2-VL-1.6B](https://huggingface.co/LiquidAI/LFM2-VL-1.6B) |
| `lfm2-vl-3b` | [LFM2-VL-3B](https://huggingface.co/LiquidAI/LFM2-VL-3B) |
| `lfm2.5-vl-1.6b` | [LFM2.5-VL-1.6B](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B) |

## Notes

- Images are resized with the `image` crate, whose filters differ slightly from the
  torchvision resize used by `transformers`. With the same pixel values, the logits match
  `transformers` to about `1e-4`, but long generations can differ by a few words.
- Decoding is greedy by default; use `--temperature` and `--top-p` to sample.
