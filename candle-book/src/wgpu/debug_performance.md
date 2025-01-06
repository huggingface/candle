 
To debug the performance of the WGPU shaders in a model, follow these steps:

1. **Enable the `wgpu_debug` feature:**  
   Compile the crate with the `wgpu_debug` feature enabled. This will store all the commands executed during the model's runtime.

2. **Log debugging information to files:**  
   At the end of the model execution, call the `log_debuginfo_to_file` function to write all used commands into multiple files. Here's an example:  
   ```rust
   #[cfg(feature = "wgpu_debug")]
   {
       device
           .as_wgpu_device()
           .unwrap()
           .log_debuginfo_to_file("{OUTPUT_PATH}", "MODEL_NAME", "VERSION_NAME")?; 
       // Example: log_debuginfo_to_file("", "llama2c", "5.0")?;
   }
   ```

3. **Analyze the generated debug files:**  
   You can either analyze the generated files directly or use the following method for benchmarking:

   - Run the script located at `candle-wasm-examples/candle-test/src/bin/m.rs`.  
     Include the generated debug files(`*_d` and `*_e`) by setting the corresponding `DEBUG` constants at the beginning of the script.  
     This script benchmarks each command in the generated debug dump and prints all commands in reverse order of their total execution duration.  

   - Optionally, this process can also be run in a browser.