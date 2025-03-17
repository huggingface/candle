# candle-gemma: 2b and 7b LLMs from Google DeepMind

[Gemma](https://ai.google.dev/gemma/docs) is a collection of lightweight open
models published by Google Deepmind with a 2b and a 7b variant for the first
version, and a 2b and a 9b variant for v2.

## Running the example

```bash
$ cargo run --example gemma --features cuda -r -- \
    --prompt "Here is a proof that square root of 2 is not rational: "

Here is a proof that square root of 2 is not rational:

Let us assume it to be rational. Then, we can write √2 = p/q where q ≠ 0 and p and q are integers with no common factors other than 1. Squaring both sides gives us (p/q)^2 = 2 or p^2/q^2 = 2. This implies that p^2 is divisible by 2, which means that p must be even. Let us write p = 2m where m is an integer. Substituting this in the above equation we get:

(p^2)/q^2 = 2 or (4m^2)/q^2 = 2 or q^2/2m^2 = 1 which implies that q^2 must be divisible by 2, and hence q is even. This contradicts our assumption that p and q have no common factors other than 1. Hence we conclude that √2 cannot be rational.
```

## Access restrictions

In order to use the v1 examples, you have to accept the license on the
[HuggingFace Hub Gemma repo](https://huggingface.co/google/gemma-7b) and set up
your access token via the [HuggingFace cli login
command](https://huggingface.co/docs/huggingface_hub/guides/cli#huggingface-cli-login).


