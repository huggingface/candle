## SigLIP2 NaFlex (reference comparison)

**Validation tooling.** Tensor-level reference comparison between candle's
`siglip2_naflex::VisionModel` and PyTorch reference outputs from the
HuggingFace transformers `Siglip2VisionModel` on the same inputs. Used during
development to confirm correctness; not a user-facing example.

### Generating reference fixtures

Reference fixtures are produced from PyTorch + the HF NaFlex preprocessor.
A reference safetensors file should contain four tensors: `pixel_values`,
`pixel_attention_mask`, `spatial_shapes`, and `vision_pooler_output`.

### Running

```
$ cargo run --release --features accelerate --example siglip2-naflex-refcompare -- \
    /path/to/naflex_vision_reference.safetensors

VisionModel loaded.
=== variable-shape ===
max abs diff: 1.685321e-5
mean abs diff: 2.788678e-6
  Row 0: cosine=1.000000  ||expected||=7.6479  ||candle||=7.6479
  Row 1: cosine=1.000000  ||expected||=8.2744  ||candle||=8.2744
```

Path can also come from `SIGLIP2_NAFLEX_REFERENCE` env var.
