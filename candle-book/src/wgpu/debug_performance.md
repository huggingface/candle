# Debugging WGPU Performance

There are three main ways to measure and debug the performance of `wgpu` devices:

1. Tracing
2. Benchmarking
3. Recording all queued operations

---

## 1. Tracing

You can use tracing to measure internal durations, for example:

- How long it takes to create buffers
- How long it takes to create bind groups
- How long it takes to encode commands
- How long it takes to wait for the GPU to be ready

To add tracing to a project, see the [`tracing`](../tracing.md) page.

In addition:

- `candle-core` has a dependency on `tracing` with  
  `features = ["release_max_level_off"]` in its `Cargo.toml`.
- With this configuration, tracing is effectively disabled in release builds.
- To use tracing, you must either:
  - run a debug build, **or**
  - remove the `features = ["release_max_level_off"]` entry from the `tracing` dependency in `Cargo.toml`.

---

## 2. Benchmarking

The directory:

```text
candle-core/benches/benchmarks/...
```

contains various benchmarks.

For example, `matmul_wgpu` can be used to:

* override the `matmul` implementation used, and
* test the performance of different `matmul` implementations under different scenarios.

Use these benches to compare implementations and understand performance characteristics on your hardware.

---

## 3. Recording All Queued Operations

To debug the performance of the WGPU shaders in a model, you can measure, record and replay all WGPU commands.

### Step 1: Enable the `wgpu_debug` feature

Compile the crate with the `wgpu_debug` feature enabled.
This causes all commands executed during the model’s runtime to be recorded.

### Step 2: Log debugging information to files

At the end of the model execution, call `log_debuginfo_to_file` to write all recorded commands into multiple files:

```rust
#[cfg(feature = "wgpu_debug")]
{
    device
        .as_wgpu_device()
        .unwrap()
        .log_debuginfo_to_file("{OUTPUT_PATH}", "MODEL_NAME", "VERSION_NAME")?;
    // Example:
    // log_debuginfo_to_file("", "llama2c", "5.0")?;
}
```

This will create 4 files:
 - `*_measurements.json` shows performance metrics for all used shaders and pipelines.
 - `*_shaders.json` shows all created shaders and there Constants used, this may be usefull to detect problems where a lot of shaders where created(shader creation comes with a overhead) and instead instead of constands pipeline parameters may be used instead.
 - `*_used_consts.json` Map of ConstsId to actual Consts used, this file is needed to replay the used commands later on.
 - `*_used_pipelines.json` Has all Pipeline, Bindgroup, Consts, and Buffer Information needed to replay the commands later on.


### Step 3: Analyze the generated debug files

You can:

* Analyze the generated files directly, **or**
* Use the helper script for automated benchmarking:

Run the script at:

```text
candle-wasm-examples/candle-test/src/bin/candle-test .rs
```
e.g:
```bash
cargo run --bin candle-test --release                # to run in native
cargo xtask run-wasm -- --release --bin candle-test  # to run in browser
```

At the top of this script, set the corresponding `DEBUG` constants to point to your generated `*_used_consts` and `*_used_pipelines` files. The script will:

* Benchmark each command in the debug dump, and
* Print all commands sorted in reverse order of their total execution duration (slowest first).

This makes it easy to identify performance bottlenecks.

> Optionally, this process can also be run in a browser, depending on your environment and setup.