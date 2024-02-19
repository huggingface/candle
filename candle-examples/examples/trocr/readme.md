# candle-trocr

`TrOCR` is a transformer OCR Model. In this example it is used to
transcribe image text. See the associated [model
card](https://huggingface.co/microsoft/trocr-base-printed) for details on
the model itself.

Supported models include:

- `--which base`: small handwritten OCR model.
- `--which large`: large handwritten OCR model.
- `--which base-printed`: small printed OCR model.
- `--which large-printed`: large printed OCR model.

## Running an example

```bash
cargo run --example trocr --release -- --image candle-examples/examples/trocr/assets/trocr.png
cargo run --example trocr --release -- --which large --image candle-examples/examples/trocr/assets/trocr.png
cargo run --example trocr --release -- --which base-printed --image candle-examples/examples/trocr/assets/noto.png
cargo run --example trocr --release -- --which large-printed --image candle-examples/examples/trocr/assets/noto.png
```

### Outputs

```
industry , Mr. Brown commented icily . " Let us have a
industry , " Mr. Brown commented icily . " Let us have a
THE QUICK BROWN FOR JUMPS OVER THE LAY DOG
THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG
```
