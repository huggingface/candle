# candle-parler-tts

[Parler-TTS](https://huggingface.co/parler-tts/parler-tts-large-v1) is a large
text-to-speech model with 2.2B parameters trained on ~45K hours of audio data.
The voice can be controlled by a text prompt.

## Run an example

```bash
cargo run --example parler-tts -r -- \
  --prompt "Hey, how are you doing today?"
```

In order to specify some prompt for the voice, use the `--description` argument.
```bash
cargo run --example parler-tts -r -- \
  --prompt "Hey, how are you doing today?" \
  --description "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."
```


https://github.com/user-attachments/assets/1b16aeac-70a3-4803-8589-4563279bba33

