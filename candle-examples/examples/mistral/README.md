# candle-mistral: 7b LLM with Apache 2.0 licensed weights

Mistral-7B-v0.1 is a pretrained generative LLM with 7 billion parameters. It outperforms all the publicly available 13b models
as of 2023-09-28. Weights (and the original Python model code) are released under the permissive Apache 2.0 license.

- [Blog post](https://mistral.ai/news/announcing-mistral-7b/) from Mistral announcing the model release.
- [Model card](https://huggingface.co/mistralai/Mistral-7B-v0.1) on the
  HuggingFace Hub.
This example supports the initial model as well as a quantized variant.

## Running the example

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

## Running the quantized version of the model

```bash
$ cargo run --example mistral --features accelerate --release -- \
$   --prompt "Here is a sample quick sort implementation in rust " --quantized -n 400
avx: false, neon: true, simd128: false, f16c: false
temp: 0.00 repeat-penalty: 1.10 repeat-last-n: 64
retrieved the files in 562.292µs
loaded the model in 1.100323667s
Here is a sample quick sort implementation in rust

``rust
fn quick_sort(arr: &mut [i32]) {
    if arr.len() <= 1 {
        return;
    }

    let pivot = arr[0];
    let mut left = vec![];
    let mut right = vec![];

    for i in 1..arr.len() {
        if arr[i] < pivot {
            left.push(arr[i]);
        } else {
            right.push(arr[i]);
        }
    }

    quick_sort(&mut left);
    quick_sort(&mut right);

    let mut i = 0;
    for _ in &left {
        arr[i] = left.pop().unwrap();
        i += 1;
    }

    for _ in &right {
        arr[i] = right.pop().unwrap();
        i += 1;
    }
}
``
226 tokens generated (10.91 token/s)
```
