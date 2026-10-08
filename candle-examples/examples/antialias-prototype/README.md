## Antialias prototype

**Validation tooling.** Standalone validation of the bilinear-with-antialias
interpolation algorithm used by `siglip2_naflex` against PyTorch reference
fixtures. Not a user-facing example.

The candle-core `upsample_bilinear2d` lacks an antialias filter; the
`siglip2_naflex` model file includes a userland helper that approximates
`F.interpolate(mode="bilinear", align_corners=False, antialias=True)` via
matmul of triangle-weight matrices. This example exercises the same algorithm
across multiple scale factors and compares to PyTorch reference outputs.

### Running

```
$ cargo run --release --features accelerate --example antialias-prototype -- \
    /path/to/antialias_reference.safetensors

Loading reference fixtures from: ...
Loaded 21 tensors

=== siglip2_16x16_to_17x15 ===
  antialias prototype vs PT       max=2.861e-6  mean=4.382e-7
  candle bilinear vs PT           max=2.089e-1  mean=2.800e-2
  candle bilinear vs PT (no AA)   max=3.353e-6  mean=4.922e-7
  PASS (tol=1e-4)
...
Worst max abs diff across all cases: 3.219e-6
All 8 cases pass at tol 1e-4
```

The reference safetensors file is produced by a Python script using
`torch.nn.functional.interpolate(mode="bilinear", antialias=True)` over
representative input shapes. Eight cases cover SigLIP2 NaFlex resize
ranges (16x16 to 17x15, 19x13, 13x17, 8x8, 24x24) plus synthetic shapes.
