# Candle Development Notes

## Shader-Slang Debug Printing

The Slang compute shaders support `printf()` for debug output.

### 1. Use `printf` in the shader

```slang
if (tid.x == 0u) {
    printf("value=%f index=%u\n", some_value, tid.x);
}
```

`slangc` automatically adds the `SPV_KHR_non_semantic_info` SPIR-V
capability when a shader uses `printf`, so no special build flags are
needed. (The `build.rs` in `candle-vulkan-kernels` passes any
whitespace-separated tokens from the `SLANGCFLAGS` environment variable
as extra flags to `slangc`, for other purposes.)

### 2. Enable the validation layer at runtime

The Khronos validation layer intercepts the `NonSemantic.DebugPrintf`
instructions and routes the `printf` output. Set these environment
variables before running:

```sh
export VK_LAYER_PRINTF_ENABLE=1          # candle: enables the layer + device extension; layer: capture shader printf
export VK_LAYER_PRINTF_TO_STDOUT=1       # layer: route to stdout
```

`VK_LAYER_PRINTF_ENABLE` is checked in
`candle-core/src/vulkan_backend/device.rs`; when set, the instance is
created with the `VK_LAYER_KHRONOS_validation` layer enabled and the
device is created with the `VK_KHR_shader_non_semantic_info` extension
enabled.

### 3. Verify with the integration test

The `test_printf` test shader prints the lowest and highest representable
value of every supported data type (u8, u32, i16, i32, i64, bf16, f16,
f32, f64) as "Expected ..., actual ..." lines, and the integration test
checks that each line carries the correct value:

```sh
VK_LAYER_PRINTF_ENABLE=1 VK_LAYER_PRINTF_TO_STDOUT=1 \
  cargo test --features vulkan --test vulkan_integration_tests \
  -- test_vulkan_shader_printf --exact
```

Without `VK_LAYER_PRINTF_ENABLE` the test still runs and validates the
kernel's data path, but no `printf` output is produced or checked.

### Notes

- `printf` adds overhead; use it only for debugging, not in production paths.
- The `VK_LAYER_PRINTF_BUFFER_SIZE` env var (default 1024) controls the
  per-message buffer size; increase it for long messages.
- The `radv is not a conformant Vulkan implementation` warning is expected
  and can be ignored.
