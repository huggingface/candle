# candle-trocr

`TrOCR` is a transformer OCR Model. In this example it is used to
transcribe image text. See the associated [model
card](https://huggingface.co/microsoft/trocr-base-printed) for details on
the model itself.

## Running an example

```bash
cargo run --example trocr --release --  --which base-hand-written --cpu --image candle-examples/examples/trocr/assets/trocr.png
cargo run --example trocr --release --  --which large-hand-written --cpu --image candle-examples/examples/trocr/assets/trocr.png
cargo run --example trocr --release --  --which base-printed --cpu --image candle-examples/examples/trocr/assets/printed-number.jpg
cargo run --example trocr --release --  --which large-printed --cpu --image candle-examples/examples/trocr/assets/printed-number.jpg
```

```
<s> industry , Mr. Brown commented icily . " Let us have a</s>
```
