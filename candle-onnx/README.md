# candle-onnx

This crate adds ONNX support to candle

## Multi-device evaluation (pipeline parallelism)

`simple_eval` runs the whole graph on a single device. To split one graph across
several devices (e.g. the first transformer blocks on `cuda:0` and the rest on
`cuda:1`, so a model that does not fit on one GPU still runs), use
`simple_eval_with_placement` with a `Placement`:

```rust
let placement = candle_onnx::Placement::new(&cuda0)
    .with_prefix("/model/layers.16/", &cuda1);
let outputs = candle_onnx::simple_eval_with_placement(&model, inputs, &placement)?;
```

A node runs on the device of the longest matching node-name prefix, otherwise it
inherits the device of its first input, otherwise the fallback device. Weights
are materialized lazily on the device that first consumes them and cross-device
copies are inserted automatically at stage boundaries. This is pipeline
parallelism (not tensor/expert parallelism), and cross-device transfers
currently require CUDA (Metal-to-Metal copies are unimplemented in candle-core).

## FAQ

#### Missing protoc installation when compiling candle-onnx

The candle-onnx dependency prost-build no longer comes bundled with prost
binaries. This could cause the following error when attempting to compile
candle-onnx:

```
error: failed to run custom build command for `candle-onnx`
Caused by: // (...)
  Could not find `protoc` installation and this build crate cannot proceed without this knowledge.
```

To fix this issue install protoc on your system and make it available in your
system `PATH`. See the [protoc
documentation](https://grpc.io/docs/protoc-installation/) for more information.
