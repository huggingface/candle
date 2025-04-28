# Tracing

Tracing is a powerful tool for identifying performance issues and bottlenecks in code.

> When using CUDA, Metal, or other devices with asynchronous execution, be aware that tracing might not accurately identify performance issues. GPU operations are typically queued and executed asynchronously, which can lead to misleading timing information in traces.
> 
> If are debugging CUDA specifically, you can set the environment variable `CUDA_LAUNCH_BLOCKING=1` which forces synchronous execution. This makes trace timings more accurate, at the cost of reduced performance.

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

