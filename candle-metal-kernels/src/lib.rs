use metal::{Buffer, Device, Function, Library, CompileOptions};
use std::collections::HashMap;
use std::sync::RwLock;

static UNARY: &'static str = include_str!("unary.metal");

pub enum Error {}

pub struct Kernels {
    libraries: RwLock<HashMap<&'static str, Library>>,
    funcs: RwLock<HashMap<String, Function>>,
}

impl Kernels {
    pub fn new() -> Self {
        let libraries = RwLock::new(HashMap::new());
        let funcs = RwLock::new(HashMap::new());
        Self { libraries, funcs }
    }
    pub fn call_unary(
        &self,
        device: &Device,
        name: &str,
        input: &Buffer,
        output: &mut Buffer,
        length: usize,
    ) -> Result<(), Error> {
        if let Some(func) = self
            .funcs
            .read()
            .expect("Failed to acquire kernel lock")
            .get(name)
        {
            call_unary(func, input, output, length);
        } else {
            let func = self
                .libraries
                .write()
                .expect("Failed to acquire lock")
                .entry("unary")
                .or_insert_with(|| {
                    device
                        .new_library_with_source(UNARY, &CompileOptions::new())
                        .expect("Failed to load unary library")
                })
                .get_function(name, None)
                .expect("Could not find unary function");
            self.funcs
                .write()
                .expect("Failed to acquire lock")
                .insert(name.to_string(), func.clone());
            call_unary(&func, input, output, length);
        }
        Ok(())
    }
}

fn call_unary(func: &Function, input: &Buffer, output: &Buffer, length: usize) {
    todo!("Call unary");
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal::{
        ComputePipelineDescriptor, MTLResourceOptions, MTLResourceUsage, MTLSize,
    };

    fn approx(v: Vec<f32>, digits: i32) -> Vec<f32>{
        let b = 10f32.powi(digits);
        v.iter().map(|t| f32::round(t * b) / b).collect()
    }

    #[test]
    fn cos() {
        let v = vec![1.0f32, 2.0, 3.0];
        let option = metal::MTLResourceOptions::CPUCacheModeDefaultCache;
        let device = Device::system_default().unwrap();
        let command_queue = device.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let input = device.new_buffer_with_data(
            v.as_ptr() as *const core::ffi::c_void,
            (v.len() * core::mem::size_of::<f32>()) as u64,
            option,
        );
        let output = device.new_buffer((v.len() * core::mem::size_of::<f32>()) as u64, option);
        let library = device
            .new_library_with_source(UNARY, &CompileOptions::new())
            .expect("Failed to load unary library");
        let func = library.get_function("cos", None).unwrap();
        let argument_encoder = func.new_argument_encoder(0);
        let arg_buffer = device.new_buffer(
            argument_encoder.encoded_length(),
            MTLResourceOptions::empty(),
        );
        argument_encoder.set_argument_buffer(&arg_buffer, 0);
        argument_encoder.set_buffer(0, &input, 0);
        argument_encoder.set_buffer(1, &output, 0);
        let pipeline_state_descriptor = ComputePipelineDescriptor::new();
        pipeline_state_descriptor.set_compute_function(Some(&func));

        let pipeline_state = device
            .new_compute_pipeline_state_with_function(
                pipeline_state_descriptor.compute_function().unwrap(),
            )
            .unwrap();

        encoder.set_compute_pipeline_state(&pipeline_state);
        encoder.set_buffer(0, Some(&arg_buffer), 0);

        encoder.use_resource(&input, MTLResourceUsage::Read);
        encoder.use_resource(&output, MTLResourceUsage::Write);

        let width = 16;

        let thread_group_count = MTLSize {
            width,
            height: 1,
            depth: 1,
        };

        let thread_group_size = MTLSize {
            width: (v.len() as u64 + width) / width,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
        let results = output.read_to_vec::<f32>(v.len());
        assert_eq!(approx(results, 4), vec![0.5403, -0.4161, -0.99]);
        assert_eq!(approx(expected, 4), vec![0.5403, -0.4161, -0.99]);
    }
}
