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
cargo run --example trocr --release --  --image candle-examples/examples/trocr/assets/trocr.png
```

```
<s> industry , Mr. Brown commented icily . " Let us have a</s>
```
