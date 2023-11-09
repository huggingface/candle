# candle-trocr

`TrOCR` is a transformer OCR Model. In this example it is used to
transcribe image text. See the associated [model
card](https://huggingface.co/microsoft/trocr-base-printed) for details on
the model itself.

## Running an example

```bash
cargo run --example trocr --release --  --which base --cpu --image assets/trocr.png
```

```
<s> industry , Mr. Brown commented icily . " Let us have a</s>
```
