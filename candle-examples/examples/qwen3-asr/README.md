# candle-qwen3-asr

Initial scaffold for a Candle example based on `qwen3-asr-rs`.

This first commit wires the example entrypoint and CLI shape. Functional model
loading/inference is added in follow-up commits to keep the migration reviewable.

## Run scaffold

```bash
cargo run --example qwen3-asr --features symphonia -- --help
```
