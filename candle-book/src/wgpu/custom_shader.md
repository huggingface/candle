# Custom shader

In "candle-core/examples/wgpu_basics.rs" is a sample project that shows how to write a custom WGPU shader.

1. Define a LoaderIndex. 
When you add your custom pipeline to the queue, you need to use a unique identifier to distinguish your shader from the default shader provided with Candle, or the shaders of other modules. 
The macro `create_loader` will generate a unique index for this purpose at compile time.
```rust  
wgpu_compute_layer::create_loader!(MyCustomLoader);
```

2. Define a ShaderLoader
A ShaderLoader is an object that implements the `ShaderLoader' trait. 
It is responsible for returning the source code of a .wgsl shader file, as well as the name of the entry point.

```rust
impl wgpu_compute_layer::ShaderLoader for MyCustomLoader{
    fn load(&self, _ : wgpu_compute_layer::ShaderIndex) -> &str {
        return "YOUR SHADER CODE GOES HERE";
    }

    fn get_entry_point(&self, _ : wgpu_compute_layer::PipelineIndex) -> &str {
        return "ENTRY POINT NAME GOES HERE"
    }
}
```
Instead of creating multiple ShaderLoaders, your ShaderLoader can handle multiple files using the index parameter. (Up to 65536 can be handled).
Each file can also have multiple compute entry points, which you can differentiate using the PipelineIndex parameter.

3. Add the ShaderLoader to the WGPU device.
You can get a reference to the WgpuDevice from WgpuStorage.device() (e.g. inside a CustomOp), 
or by pattern matching the candle device. 
```rust
    wgpu_device.add_wgpu_shader_loader(MyCustomLoader::LOADER_INDEX, || {MyCustomLoader{}});
```
This will add your shader loader at the specified index. 
For example, your index created by the create_loader macro is 13. 
Later on when we enqueue a custom shader we will use this index to tell the wgpu backend that we want to enqueue one of our custom shaders.
For example, if you enqueue (Loader=13, Shader=0, EntryPoint=0), the wgpu system will look for that pipeline in a hashmap. If it does not find it, it will ask the shader loader at index 13 for the first shader and the name of the first entry point of that shader.
  
4. Queue your shader:

    To add a pipeline to the queue, we need to use the following commands:
    1. Define a reference to the metastructure. 
        Here we can pass additional meta information for the operation
        ```rust
        let mut queue = wgpu_device.get_queue();
        queue.add(42);
        queue.add(13);
        ..
        ```
    2.  Define the pipeline to use. 
        Use your ShaderLoaderIndex to define which pipeline and entry point to use.
        ```rust
        let pipeline = queue.get_pipeline(PipelineIndex::new(ShaderIndex::new(MyCustomLoader::LOADER_INDEX, 0), 0));
        //or
        let pipeline = queue.get_pipeline_const(PipelineIndex::new(ShaderIndex::new(MyCustomLoader::LOADER_INDEX, 0), 0), [42, 12]); //CONSTV_0 = 42, CONSTV_1 = 12
        ```
        It is also possible to define webgpu override const values using the get_pipeline_const function.
        Each time this const parameter changes, a new shader is compiled. The constant is compiled into the shader. This can improve performance. 
        But remember that shader compilation also takes time, so if the constant value changes frequently, you may want to add the value as a meta parameter instead.
        
        The names of the following const parameters must be `CONSTV_{N}`.
    
    3. Define the Bindgroup
        The bindgroup defines the input and output buffers for your operations.
        ```rust
        let bing_group = wgpu_device.create_bind_group_input0(*output_buffer.buffer(), candle_core::DType::U32.into());
        ```
        In general, there are 4 possible Bindgroup types:
        - Bindgroup0 - V_Dest(Binding 0), V_Meta(Binding 1)
        - Bindgroup1 - V_Dest(Binding 0), V_Meta(Binding 1), V_Input1(Binding 2)
        - Bindgroup2 - V_Dest(Binding 0), V_Meta(Binding 1), V_Input1(Binding 2), V_Input2(Binding 3)
        - Bindgroup3 - V_Dest(Binding 0), V_Meta(Binding 1), V_Input1(Binding 2), V_Input2(Binding 3), V_Input3(Binding 4)

    4. add the command to the queue:
        ```rust
        queue.enqueue_workgroups(
                    pipeline,
                    bind_group,
                    x,
                    y,
                    z,
                    workloadestimate //e.g. m*k*n for a matmul
                );
        ```