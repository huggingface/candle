use metal::{Buffer, CompileOptions, Device, Function, Library};
use std::collections::HashMap;
use std::sync::RwLock;

pub const INDEXING: &str = include_str!("indexing.metal");
pub const UNARY: &str = include_str!("unary.metal");

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

fn call_unary(_func: &Function, _input: &Buffer, _output: &Buffer, _length: usize) {
    todo!("Call unary");
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal::{
        CompileOptions, ComputePipelineDescriptor, Device, FunctionConstantValues, MTLDataType,
        MTLResourceOptions, MTLResourceUsage, MTLSize, NSUInteger,
    };
    use std::ffi::c_void;
    use std::mem;

    pub fn void_ptr<T>(v: &T) -> *const c_void {
        (v as *const T).cast()
    }
    fn approx(v: Vec<f32>, digits: i32) -> Vec<f32> {
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
            v.as_ptr() as *const c_void,
            (v.len() * mem::size_of::<f32>()) as u64,
            option,
        );
        let output = device.new_buffer((v.len() * mem::size_of::<f32>()) as u64, option);
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

    #[test]
    fn index_add() {
        let device = Device::system_default().expect("no device found");

        let options = CompileOptions::new();
        let library = device.new_library_with_source(INDEXING, &options).unwrap();

        let left = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let right = [1.0f32; 15];
        let index = [0u32, 4, 2];

        let ids_dim_size = index.len() as u32;

        // Are these reversed?
        let src_dim_size: u32 = 9;
        let dst_dim_size: u32 = 15;
        let left_size: u32 = 3;
        let right_size: u32 = 3;

        let fcv = FunctionConstantValues::new();
        fcv.set_constant_value_at_index(void_ptr(&ids_dim_size), MTLDataType::UInt, 0);
        fcv.set_constant_value_at_index(void_ptr(&src_dim_size), MTLDataType::UInt, 1);
        fcv.set_constant_value_at_index(void_ptr(&dst_dim_size), MTLDataType::UInt, 2);
        fcv.set_constant_value_at_index(void_ptr(&left_size), MTLDataType::UInt, 3);
        fcv.set_constant_value_at_index(void_ptr(&right_size), MTLDataType::UInt, 4);

        let function = library.get_function("index_add", Some(fcv)).unwrap();
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .unwrap();
        let options = MTLResourceOptions::StorageModeShared;

        let ids_size = (index.len() * mem::size_of::<u32>()) as NSUInteger;
        let input_size = (left.len() * mem::size_of::<f32>()) as NSUInteger;
        let output_size = (right.len() * mem::size_of::<f32>()) as NSUInteger;

        let ids = device.new_buffer_with_data(void_ptr(&index), ids_size, options);
        let inputs = device.new_buffer_with_data(void_ptr(&left), input_size, options);
        let outputs = device.new_buffer_with_data(void_ptr(&right), output_size, options);

        let command_queue = device.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        let thread_group_memory_length = output_size;
        encoder.set_threadgroup_memory_length(0, thread_group_memory_length as NSUInteger);

        encoder.use_resource(&ids, MTLResourceUsage::Read);
        encoder.use_resource(&inputs, MTLResourceUsage::Read);
        encoder.use_resource(&outputs, MTLResourceUsage::Write);

        encoder.set_buffer(0, Some(&ids), 0);
        encoder.set_buffer(1, Some(&inputs), 0);
        encoder.set_buffer(2, Some(&outputs), 0);
        let width = 16;

        let thread_group_count = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };

        let thread_group_size = MTLSize {
            width,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let expected = vec![
            2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 8.0, 9.0, 10.0, 1.0, 1.0, 1.0, 5.0, 6.0, 7.0,
        ];
        let result = outputs.read_to_vec::<f32>(right.len());
        assert_eq!(result, expected);
    }
}
