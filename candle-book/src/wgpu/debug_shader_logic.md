# Debugging WGPU Shader Logic

This page describes how to debug **incorrect results and shader logic errors** in the WGPU backend by recording a **complete execution trace** of GPU commands.

This mechanism is intended for **correctness debugging** (unit tests, small reproductions), not for performance profiling.

Unlike `wgpu_debug`, which records only **pipeline statistics and timing**, this mechanism records:

* Full shader source code
* All dispatched pipelines
* Copies of all input and output buffers
* Dispatch order and parameters

Because it records full buffers and shaders, the generated data can be **very large** and should only be used with small test cases.

---

## Enabling

This feature is available with the `wgpu_debug` feature:

```bash
cargo build --features="wgpu wgpu_debug"
```

---

## Recording Commands

Wrap the code you want to debug:

```rust
#[cfg(feature = "wgpu_debug")]
{
    let wgpu = device.as_wgpu_device()?;
    wgpu.inner_device().start_recording_commands();

    // Run the operations you want to debug here (prefer small unit tests).

    wgpu.inner_device()
        .stop_recording_commands(&"PATH TO ZIP FILE TO WRITE ALL DISPATCHES TO")?;
}
```

Everything executed between `start_recording_commands` and `stop_recording_commands` is recorded into a ZIP file.

---

## Synchronization

Make sure all work has completed before stopping the recording:

* Synchronize the device, or
* Read back a buffer from the GPU

Otherwise, the recording may be incomplete.

---

## Difference to `wgpu_debug`

* `wgpu_debug` (performance):

  * Records pipeline names, call counts, timing
  * No shader code or buffers

* Full command recording (this page):

  * Records full shader code and all buffers
  * Intended for debugging incorrect results

---

## Notes

* Use only for **small tests** — recordings can become very large.
* Intended for native debugging (not supported in the browser).