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

## 3. Recording and Replaying WGPU Operations

To debug and optimize the performance of WGPU shaders in a model, you can **record**, **inspect**, and **replay** all WGPU commands used during execution.

### 3.1 Features: `wgpu_debug` and `wgpu_debug_serialize`

There are two related features:

#### `wgpu_debug`

Enable this feature to record all WGPU commands executed during the model’s runtime.

```bash
# Example
cargo build --features wgpu_debug
```

When `wgpu_debug` is enabled:

* All queued WGPU commands (pipelines, bind groups, buffers, dispatches, etc.) are recorded.
* At the end of execution, you can dump them to disk using `log_debuginfo_to_file` (see Step 2).
* The recorded data can later be **replayed** to benchmark or debug performance.

#### `wgpu_debug_serialize`

This feature is more lightweight:

* It **does not** record any commands at runtime.
* Instead, it adds `serde::Serialize` derives (and related metadata) to pipelines, bind groups, shader info, etc.
* This is useful when you want to **load and work with** the files produced by a `wgpu_debug` run (for example, to simulate or analyze them in another process or crate), without enabling full command recording again.

Typical workflow:

1. Run your model once with `wgpu_debug` enabled to generate the debug files.
2. In another tool/binary/crate, enable `wgpu_debug_serialize` to **deserialize and inspect** those recorded files, replay commands, or run simulations.

---

### Step 2: Log debugging information to files

At the end of the model execution (with `wgpu_debug` enabled), call `log_debuginfo_to_file` to write all recorded information into a set of files:

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

This will create four files:

* `*_measurements.json`
  Contains performance metrics for all used shaders and pipelines.

* `*_shaders.json`
  Contains all created shaders and their used constants.
  This is useful to detect situations where **many slightly different shaders** are created (shader creation is expensive) and where using **pipeline parameters instead of constants** might be more efficient.

* `*_used_consts.json`
  Maps `ConstsId` to the actual constants used.
  This file is required when **replaying** the recorded commands.

* `*_used_pipelines.json`
  Contains all pipeline, bind group, constants, and buffer information needed to **replay** the recorded commands.

---

### Step 3: Analyze and Replay the Generated Debug Files

You can:

* Inspect the generated JSON files manually, **or**
* Use the provided helper script for automated benchmarking and analysis.

The script is located at:

```text
candle-wasm-examples/candle-test/src/bin/candle-test.rs
```

Example invocations:

```bash
# Run natively
cargo run --bin candle-test --release

# Run in the browser (via wasm)
cargo xtask run-wasm -- --release --bin candle-test
```

At the top of `candle-test.rs`, configure the relevant `DEBUG` constants to point to your generated `*_used_consts.json` and `*_used_pipelines.json` files. Once configured, the script will:

* Replay and benchmark each recorded command, and
* Print all commands sorted in **reverse order of total execution duration** (slowest first).

This makes it straightforward to spot performance bottlenecks and problematic shaders/pipelines.

> Depending on your setup, you can run this analysis either natively or in the browser, using the same recorded debug data.
