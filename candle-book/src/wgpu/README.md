# Wgpu Installation
To use the wgpu backend, you must enable the wgpu feature.
In the code, use the `new_wgpu` function to create a new wgpu device.
```rust
//on the browser, the async method must be used.
let device = Device::new_wgpu_async(0).await?

//or

let device = Device::new_wgpu(0)?

//or

//Pass additional configuration, e.g. the wgpu backend to be used (vulkan, dx12 or metal).
let device = Device::new_wgpu_config_async(0, config).await? 

//or

let device = Device::new_wgpu_config(0, config)?
```

## GPU Storage Query Limitation in WGPU

Currently, WGPU does not provide a way to query the available storage of a GPU device. As a result, the Candle implementation for WGPU cannot determine the number of buffers that can be cached or when existing buffers should be deleted.

To address this limitation, the `buffer_cached_max_allowed_size` property is included in the device creation configuration. This property allows users to specify the maximum amount of memory, in bytes, that Candle is permitted to allocate for buffers. By default, this value is set to 8 GB.

## Feature Support Table

| Feature                     | Support Status                                  | Notes                                                              |
|-----------------------------|-------------------------------------------------|--------------------------------------------------------------------|
| **<span style="color:#1E90FF">Data Types</span>**                                  |                                       |                                                                    |
| f32                         | ✅ Supported                                     |                                                                    |
| u32                         | ✅ Supported                                     |                                                                    |
| u8                          | ⚠️ Only Output of Cmp                            | *Only f32, I32 and U32 are available in a webGpu shader            |
| i64                         | ⚠️ Supported Native                              |                                                                    |
| f64                         | ⚠️ Supported Native                              |                                                                    |
| f16                         | ❌ Not Supported                                 |                                                                    |
| bf16                        | ❌ Not Supported                                 |                                                                    |
| **<span style="color:#1E90FF">Operations</span>**              |                                                 |   All operations support non-contiguous arrays                                                                   |
| Unary Operations            | ✅ Supported                                     |                                                   |
| Binary Operations           | ✅ Supported                                     |                                |
| MatMul                      | ✅ Supported                                     |                                                                    |
| Reduce Operations           | ✅ Supported                                     | Sum, Min, Max, (ArgMax, ArgMin works only if continues Dimensions are reduced)                                     |
| Conv2d                      | ✅ Supported                                     |                                                                    |
| Conv2dTranspose             | ✅ Supported                                     |       |
| Conv1d                      | ✅ Supported                                     |                                                                    |
| Conv1dTranspose             | ✅ Supported                                     |                                                                    |
| Index Select                | ✅ Supported                                     |                                                                    |
| Where_cond                  | ✅ Supported                                     |                                                                    |
| Pool2dMax                   | ✅ Supported                               |                                                                    |
| Pool2dAvg                   | ✅ Supported                               |                                                                    |
| Upsample                    | ✅ Supported                               |                                                                    |
| Gather                      | ✅ Supported                               |                                                                    |
| Scatter_add                 | ✅ Supported                               |                                                                    |
| Index_add                   | ✅ Supported                              |                                                                    |
| **<span style="color:#1E90FF">Not Implemented</span>**        |                                                 |                                                                    |
| ArgSort                     | ❌ Not Implemented                               |                                                                    |
| Quantized Matrices          | ❌ Not Supported?                                 |                                                                    |



# Usage in the Browser
It is not possible to synchronously request a device, read a gpu memory or synchronise the device in the browser. 
There are synchronous methods for these operations, but as these methods will just block the current thread, they will not work in the browser and will fail. 
If you want to target the browser, you need to use the async methods.

The following code demonstrates how to use wgpu in the browser:
```rust
use candle_core::{Device, Tensor};

//use the await method to create a device, this must be asynchronous
let device = Device::new_wgpu_async(0, config).await? 

let a = Tensor::randn(0f32, 1., (2, 3), &device)?;
let b = Tensor::randn(0f32, 1., (3, 4), &device)?;

let c = a.matmul(&b)?;

//If we want to synchronise with the device, we must use the async function.
device.synchonize_async();

//We need to asynchronously copy the gpu buffer back to the cpu, 
//to_device() will not work
let c = c.to_device_async(&Device:Cpu).await?;
//or c.to_vec2_async().await?
console_log!("{c}");
Ok(())
```
** Note that the above limitation only applies if the browser is targeted; a native program can still use the same sync functions.

# Example Projects
All example projects, as well as the WAM examples, can be used with the wgpu backend. 

In order to use **WGPU** add `--features wgpu` to the example command line.
e.g:
```bash
cargo run --example stable-diffusion --release --features="wgpu" -- --prompt "Anthropomorphic cat dressed as a fire fighter" --sd-version v1-5
```


# known problems
- not all dtypes are supported: f32, u32 is implemented for most and u8 for a cmp and whereCond. 
  f64 or i64 is supported for native programs. WebGpu has no support for f64 or i64 or u8 dtypes<br>
  (There is a f16 extension in the webGpu Spec, but this is currently not supported by wgpu(https://github.com/gfx-rs/wgpu/issues/4384))
- Reduce Implementation error: When using ArgMin, ArgMax with non continues reduction dimensions will probably not work. e.g if dim 0 and 2 are reduced. The current implementation will first reduce dim 2, and afterwards dim 0. This approach will not work for ArgMin/ArgMax as after the first reduction the type and source values changed.
- Buffer size limitation: 
  Depending on the driver used, it may not be possible to create a large enough buffer. 
  Also, you may be able to create a large buffer, but not be able to bind to the entire buffer in a single operation.
- Browser performance worse than native:
  The shaders have been optimized for an NVIDIA GPU using a native Vulkan driver. 
  Performance may not be optimal on other platforms or GPUs. Browser performance has been shown to be slower than native.
