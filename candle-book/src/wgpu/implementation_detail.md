# Implementation details:

## Kernels:
This implementation uses a custom wgsl kernel system in candle-wgpu-kernels.
For the syntax look at 
At compile time, files ending in `.pwgsl`(TODO link to "pwgsl_filed.md") are processed by the build.rs and included with the following DTYPE-Variants:
["F32", "U32", "I64", "F64", "F16", "U8"] defining the TYPE name as global defines. 


In addition, a rust module is defined for each .pwgsl shader file, which contains information about the compute shader functions contained in that file. When called from the candle_backend, these automatically generated mappings are used to call the kernel.

In addition, the build.rs further truncates the wgsl files (removes all spaces, truncates variable names, constant override names or function names and removes unused global variables and functions).

# Implementation Details  

## Kernels  

This implementation utilizes a custom WGSL kernel system provided by `candle-wgpu-kernels`.  

For details on the syntax, see **[pwgsl_files.md]**.  

### Kernel Preprocessing  

At compile time, `.pwgsl` files are processed by `build.rs` and included with the following **`DTYPE` variants**, which define the `TYPE` name as global preprocessor defines:  
- `"F32"`  
- `"U32"`  
- `"I64"`  
- `"F64"`  
- `"F16"`  
- `"U8"`  

### Rust Module Generation  

For each `.pwgsl` shader file, a corresponding **Rust module** is automatically generated.  
This module contains metadata about the compute shader functions defined in the file.  
When a kernel is invoked from `candle_backend`, these auto-generated mappings ensure the correct function is called.  

### WGSL Optimization  

Additionally, `build.rs` performs **WGSL optimization**, which includes:  
- **Whitespace removal** – Stripping unnecessary spaces to reduce file size.  
- **Variable name truncation** – Shortening variable names, constant overrides, and function names.  
- **Dead code elimination** – Removing unused global variables and functions.  

## Cache system:
All called wgpu functions are not executed directly, but first queued in an internal queue inside the WgpuDevice object. 
All previously queued functions are only flushed to the GPU when a buffer is requested to be read, the device is synchronised, or data is copied from the CPU to the wgpu device. 

When flushed, previously created buffers and bindgroups are reused using a custom implemented cache system. (For example, to generate an image using Wuerstchen, more than 2_000_000 commands will be queued, the current cache system will only create about 8000 buffers and 100_000 bindgroups for these commands (instead of creating 2_000_000 bindgroups and output buffers for each command)).

Objects: 
BufferReference(an object representing a virtual buffer. It may or may not be currently associated with an actual CachedBuffer)
CachedBuffer(an object representing a Wgpu::Buffer)
CachedBindgroup(An object representing a wgpu::bindgroup)

All these 3 objects are held in a separate vec storage. 
Objects can be read or written using a reference (an index into the vec and a timestamp value).
When an entry is deleted, the timestamp value at that index is incremented to ensure that no further entries are made.