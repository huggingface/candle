# candle-florence2

[Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks](https://arxiv.org/abs/2311.06242) by Microsoft.

Florence-2 is a unified vision-language model that handles diverse vision tasks
(captioning, detection, segmentation, OCR, grounding) through a single architecture.
It uses a DaViT (Dual Attention Vision Transformer) backbone with alternating
spatial window attention and channel attention, followed by a BART-like
encoder-decoder language model.

This example demonstrates the DaViT vision encoder, which extracts visual features
from images. The features are projected to the language model's embedding space
for downstream text generation.

## Running on an example

Extract visual features from an image:

```bash
cargo run --example florence2 --release -- --image candle-examples/examples/yolo-v8/assets/bike.jpg
```

```
Florence-2 config: vision_stages=4, text_d_model=768, projection_dim=768
  DaViT: dim_embed=[128, 256, 512, 1024], depths=[1, 1, 9, 1], window_size=12
Encoding image (768x768)...
Visual features shape: [1, 2, 768]
```

![Leading group, Giro d'Italia 2021](../yolo-v8/assets/bike.jpg)
