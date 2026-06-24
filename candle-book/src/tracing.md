# Tracing

Tracing is a powerful tool for identifying performance issues and bottlenecks in code.

> Profiling on GPUs is trickier due to asynchronous execution, see the [GPU section](#gpu).

## Overview

Candle uses the [tracing](https://docs.rs/tracing/latest/tracing/) crate for instrumentation.

To try it out, run an example in `candle-examples` with the `--tracing` flag. 
This generates a trace file, typically named `trace-<timestamp>.json`. 
You can view the trace in Chrome by navigating to `chrome://tracing/`, clicking **Load**, and selecting the generated trace file.

## Adding Tracing

Candle includes built-in tracing for many internal operations, using [spans](https://docs.rs/tracing/latest/tracing/struct.Span.html) to mark key points of execution.

To add custom tracing in your code, you can define a span like this:

```rust
let span = tracing::span!(tracing::Level::TRACE, name);
```

Then, to record the span during execution, create a guard:

```rust
let _enter = span.enter();
```

This guard will record the span's duration, from when it is created to when it is dropped, into a global data structure managed by the tracing crate.

## Recording and Saving a Trace

To capture and save trace data, you need to configure the tracing system with an output format. Candle uses the [tracing_subscriber](https://docs.rs/tracing-subscriber/latest/tracing_subscriber/) and [tracing_chrome](https://docs.rs/tracing-chrome/latest/tracing_chrome/) crates.

The snippet below sets up a Chrome compatible recorder that logs all tracing activity between creation and drop of the guard:

```rust
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::prelude::*;

let _guard = {
    let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
    tracing_subscriber::registry().with(chrome_layer).init();
    guard
};
```

## GPU

When using CUDA, Metal, or other asynchronous GPU backends, tracing may produce misleading timing data because operations are queued rather than executed immediately.

### CUDA

For CUDA-specific profiling, you have two options:

1. Set the environment variable `CUDA_LAUNCH_BLOCKING=1` which forces synchronous execution. This makes trace timings more accurate, at the cost of reduced performance.
2. Use [NVIDIA's Nsight Systems](https://developer.nvidia.com/nsight-systems) (`nsys profile` and `nsys-ui`) which are designed specifically for profiling asynchronous CUDA executions.

We recommend using NVIDIA's Nsight Systems when possible, as it offers accurate performance data without altering typical execution patterns. In contrast, setting the `CUDA_LAUNCH_BLOCKING` environment variable forces synchronous execution, which can significantly alter execution behavior.

#### Performance Profiling with NVIDIA Nsight Systems

1. Generate an `.nsys-rep` file containing performance data ([docs](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#example-single-command-lines))
   - Run `nsys profile --trace cuda,nvtx,osrt --gpu-metrics-device=all --output profile_run ./target/debug/... --prompt "whatever "`
1. Open the generated `.nsys-rep` report file in Nsight Systems GUI
    - File > Open