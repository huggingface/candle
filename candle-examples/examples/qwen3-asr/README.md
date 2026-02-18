# candle-qwen3-asr

Qwen3-ASR example ported from `qwen3-asr-rs` into Candle examples.

This migration stage includes:
- offline transcription
- streaming transcription
- model/tokenizer loading from Hugging Face or local path

Current limitation:
- for this first migration pass, the input audio must already be 16kHz mono.

## Run

Offline (downloads `sample:jfk` if no input is provided):

```bash
cargo run --example qwen3-asr --features symphonia --release -- --model-id Qwen/Qwen3-ASR-0.6B
```

Streaming:

```bash
cargo run --example qwen3-asr --features symphonia --release -- \
  -- --model-id Qwen/Qwen3-ASR-0.6B --stream --chunk-size-sec 0.5 --print-intermediate
```
