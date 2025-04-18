# candle-falcon

Falcon is a general large language model.

## Running an example

Make sure to include the `--use-f32` flag if using CPU, because there isn't a BFloat16 implementation yet.
```
cargo run --example falcon --release -- --prompt "Flying monkeys are" --use-f32
```