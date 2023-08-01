# Installation

Start by creating a new app:

```bash
cargo new myapp
cd myapp
cargo add --git https://github.com/LaurentMazare/candle.git candle
```

At this point, candle will be built **without** CUDA support.
To get CUDA support use the `cuda` feature
```bash
cargo add --git https://github.com/LaurentMazare/candle.git candle --features cuda
```

You can check everything works properly:

```bash
cargo build
```


You can also see feature `mkl` which could be interesting to get faster inference on CPU. [Using mkl](./advanced/mkl.md)
