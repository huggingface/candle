## SigLIP2 NaFlex (variable-shape forward)

Demonstrates the variable-aspect-ratio forward pass of SigLIP2 NaFlex, where
each input in the batch may have a different `(h_patches, w_patches)` shape.
Inputs are padded to `max_num_patches` along the patch dimension, and a
`pixel_attention_mask` distinguishes real patches from padding.

### Running

```
$ cargo run --release --features accelerate --example siglip2-naflex-variable

Variable-shape forward succeeded.
spatial_shapes: [[16, 16], [12, 12]]
output shape: [2, 768]
```

The example uses synthetic zero-input pixel values and exercises the forward
path without downloading model weights or test images.
