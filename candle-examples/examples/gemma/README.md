# candle-gemma: 2b and 7b LLMs from Google DeepMind

[Gemma](https://ai.google.dev/gemma/docs) is a collection of lightweight open
models published by Google Deepmind with a 2b and a 7b variant.

In order to use the example below, you have to accept the license on the
[HuggingFace Hub Gemma repo](https://huggingface.co/google/gemma-7b) and set up
your access token via the [HuggingFace cli login
command](https://huggingface.co/docs/huggingface_hub/guides/cli#huggingface-cli-login).

## Running the example

```bash
$ cargo run --example gemma --release -- --prompt "fn count_primes(max_n: usize)"
fn count_primes(max_n: usize) -> usize {
    let mut primes = vec![true; max_n];
    for i in 2..=max_n {
        if primes[i] {
            for j in i * i..max_n {
                primes[j] = false;
             }
         }
    }
    primes.len()
}
```

