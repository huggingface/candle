# candle-mistral: 7b LLM with Apache 2.0 licensed weights

Mistral-7B-v0.1 is a pretrained generative LLM with 7 billion parameters. It outperforms all the publicly available 13b models
as of 2023-09-28. Weights (and the original Python model code) are released under the permissive Apache 2.0 license.

- [Blog post](https://mistral.ai/news/announcing-mistral-7b/) from Mistral announcing the model release.
- [Model card](https://huggingface.co/mistralai/Mistral-7B-v0.1) on the
  HuggingFace Hub.

```bash
$ cargo run --example mistral --release --features cuda -- --prompt 'Write helloworld code in Rust' --sample-len 150

Generated text:
Write helloworld code in Rust
=============================

This is a simple example of how to write "Hello, world!" program in Rust.

## Compile and run

``bash
$ cargo build --release
   Compiling hello-world v0.1.0 (/home/user/rust/hello-world)
    Finished release [optimized] target(s) in 0.26s
$ ./target/release/hello-world
Hello, world!
``

## Source code

``rust
fn main() {
    println!("Hello, world!");
}
``

## License

This example is released under the terms
```
