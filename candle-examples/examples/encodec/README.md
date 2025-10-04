# candle-endocec

[EnCodec](https://huggingface.co/facebook/encodec_24khz) is a high-quality audio
compression model using an encoder/decoder architecture with residual vector
quantization.

## Running one example

```bash
cargo run --example encodec --features encodec --release -- code-to-audio \
    candle-examples/examples/encodec/jfk-codes.safetensors \
    jfk.wav
```

This decodes the EnCodec tokens stored in `jfk-codes.safetensors` and generates
an output wav file containing the audio data.

Instead of `code-to-audio` one can use:
- `audio-to-audio in.mp3 out.wav`: encodes the input audio file then decodes it to a wav file.
- `audio-to-code in.mp3 out.safetensors`: generates a safetensors file
  containing EnCodec tokens for the input audio file.

If the audio output file name is set to `-`, the audio content directly gets
played on default audio output device. If the audio input file is set to `-`, the audio
gets recorded from the default audio input.
