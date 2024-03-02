# candle-metavoice

MetaVoice-1B is a text-to-speech model trained on 100K hours of speech, more
details on the [model
card](https://huggingface.co/metavoiceio/metavoice-1B-v0.1).

## Run an example

```bash
cargo run --example metavoice --release -- \\
  --prompt "This is a demo of text to speech by MetaVoice-1B, an open-source foundational audio model."
```
