# silero-vad: Voice Activity Detection

[Silero VAD (v5)](https://github.com/snakers4/silero-vad) detects voice activity in streaming audio.

This example uses the models available in the hugging face [onnx-community/silero-vad](https://huggingface.co/onnx-community/silero-vad).

## Running the example

```bash
$ arecord -t raw -f S16_LE -r 16000 -c 1 -d 5 - | cargo run --example silero-vad --release --features onnx -- --sample-rate 16000
```

