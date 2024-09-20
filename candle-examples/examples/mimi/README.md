# candle-mimi

[Mimi](https://huggingface.co/kyutai/mimi) is a state of the art audio
compression model using an encoder/decoder architecture with residual vector
quantization. The candle implementation supports streaming meaning that it's
possible to encode or decode a stream of audio tokens on the flight to provide
low latency interaction with an audio model.

## Running one example

Generating some audio tokens from an audio files.
```bash
wget https://github.com/metavoiceio/metavoice-src/raw/main/assets/bria.mp3
cargo run --example mimi --features mimi --release -- audio-to-code bria.mp3 bria.safetensors
```

And decoding the audio tokens back into a sound file.
```bash
cargo run --example mimi --features mimi --release -- code-to-audio bria.safetensors bria.wav
```
