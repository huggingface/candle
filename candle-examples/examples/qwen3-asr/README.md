# candle-qwen3-asr

Qwen3-ASR example ported from `qwen3-asr-rs` into Candle examples.

This migration stage includes:
- offline transcription
- streaming transcription
- model/tokenizer loading from Hugging Face or local path

Input notes:
- wav input is decoded via `symphonia`
- non-16kHz audio is automatically resampled to 16kHz for ASR

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
