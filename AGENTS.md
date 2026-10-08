# Candle Development Notes

## Shader-Slang Debug Printing

The Slang compute shaders support `printf()` for debug output. To enable it:

### 1. Build with the `SPV_KHR_non_semantic_info` capability

Set the `SLANGCFLAGS` environment variable before building:

```sh
export SLANGCFLAGS="-capability SPV_KHR_non_semantic_info"
cargo build --release --features vulkan
```

The `build.rs` in `candle-vulkan-kernels` reads `SLANGCFLAGS` and passes
each whitespace-separated token as an extra flag to `slangc`.

### 2. Enable the Vulkan validation layer at runtime

The Khronos validation layer intercepts shader `printf` output and routes it
to stdout. Set these environment variables before running:

```sh
export CANDLE_VULKAN_VALIDATION=1        # enables VK_LAYER_KHRONOS_validation in candle
export VK_LAYER_PRINTF_ENABLE=1          # layer: capture shader printf
export VK_LAYER_PRINTF_TO_STDOUT=1       # layer: route to stdout
```

`CANDLE_VULKAN_VALIDATION` is checked in
`candle-core/src/vulkan_backend/device.rs`; when set, the instance is
created with the `VK_LAYER_KHRONOS_validation` layer enabled.

### 3. Use `printf` in the shader

```slang
if (tid.x == 0u) {
    printf("value=%f index=%u\n", some_value, tid.x);
}
```

Output appears on stdout interleaved with normal program output.

### Notes

- `printf` adds overhead; use it only for debugging, not in production paths.
- The `VK_LAYER_PRINTF_BUFFER_SIZE` env var (default 1024) controls the
  per-message buffer size; increase it for long messages.
- The `radv is not a conformant Vulkan implementation` warning is expected
  and can be ignored.
