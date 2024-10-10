# candle-rnn: Recurrent Neural Network

This example demonstrates how to use the `candle_nn::rnn` crate to run LSTM and GRU, including bidirection and multi-layer.

## Running the example

```bash
$ cargo run --example rnn --release
```

Choose LSTM or GRU via the `--model` argument, number of layers via `--layer`, and to enable bidirectional via `--bidirection`.

```bash
$ cargo run --example rnn --release -- --model lstm --layers 3 --bidirection
```
