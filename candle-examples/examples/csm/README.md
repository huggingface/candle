# Conversational Speech Model (CSM)

CSM is a speech generation model from Sesame,
[SesameAILabs/csm](https://github.com/SesameAILabs/csm).

It can generate a conversational speech between two different speakers.
The speakers turn are delimited by the `|` character in the prompt.

```bash
cargo run --example csm --features cuda -r -- \
    --voices candle-examples/examples/csm/voices.safetensors  \
    --prompt "Hey how are you doing?|Pretty good, pretty good. How about you?"
```

