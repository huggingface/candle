use metal::{Buffer, CompileOptions, Device, Function, Library};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::RwLock;

pub const AFFINE: &str = include_str!("affine.metal");
pub const INDEXING: &str = include_str!("indexing.metal");
pub const UNARY: &str = include_str!("unary.metal");

static LIBRARY_SOURCES: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut l = HashMap::new();
    l.insert("affine", AFFINE);
    l.insert("indexing", INDEXING);
    l.insert("unary", UNARY);
    l
});

#[derive(thiserror::Error, Debug)]
pub enum MetalKernelError {
    #[error("Could not lock kernel map: {0}")]
    LockError(String),
    #[error("Error while loading library: {0}")]
    LoadLibraryError(String),
    #[error("Error while loading function: {0}")]
    LoadFunctionError(String),
}

impl<T> From<std::sync::PoisonError<T>> for MetalKernelError {
    fn from(e: std::sync::PoisonError<T>) -> Self {
        Self::LockError(e.to_string())
    }
}

type KernelMap<T> = HashMap<&'static str, T>;
type Libraries = KernelMap<Library>;
type Functions = KernelMap<Function>;

#[derive(Debug)]
pub struct Kernels {
    libraries: RwLock<Libraries>,
    funcs: RwLock<Functions>,
}

impl Kernels {
    pub fn new() -> Self {
        let libraries = RwLock::new(Libraries::new());
        let funcs = RwLock::new(Functions::new());
        Self { libraries, funcs }
    }

    pub fn init(device: &Device) -> Result<Self, MetalKernelError> {
        let kernels = Self::new();
        kernels.load_libraries(device)?;
        Ok(kernels)
    }

    fn load_libraries(&self, device: &Device) -> Result<(), MetalKernelError> {
        for name in LIBRARY_SOURCES.keys() {
            self.load_library(device, name)?;
        }
        Ok(())
    }

    fn get_library_source(&self, name: &'static str) -> Option<&'static str> {
        LIBRARY_SOURCES.get(name).cloned()
    }

    pub fn load_library(
        &self,
        device: &Device,
        name: &'static str,
    ) -> Result<Library, MetalKernelError> {
        let mut libraries = self.libraries.write()?;
        if let Some(lib) = libraries.get(name) {
            Ok(lib.clone())
        } else {
            let source = self.get_library_source(name).ok_or_else(|| {
                MetalKernelError::LoadLibraryError(format!("No source found for {}", name))
            })?;
            let lib = device
                .new_library_with_source(source, &CompileOptions::new())
                .map_err(|e| MetalKernelError::LoadLibraryError(e.to_string()))?;
            libraries.insert(name, lib.clone());
            Ok(lib)
        }
    }

    pub fn load_function(
        &self,
        device: &Device,
        library_name: &'static str,
        name: &'static str,
    ) -> Result<Function, MetalKernelError> {
        let mut funcs = self.funcs.write()?;
        if let Some(func) = funcs.get(name) {
            Ok(func.clone())
        } else {
            let func = self
                .load_library(device, library_name)?
                .get_function(name, None)
                .map_err(|e| MetalKernelError::LoadFunctionError(e.to_string()))?;
            funcs.insert(name, func.clone());
            Ok(func)
        }
    }

    pub fn call_unary(
        &self,
        device: &Device,
        library_name: &'static str,
        name: &'static str,
        input: &Buffer,
        output: &mut Buffer,
        length: usize,
    ) -> Result<(), MetalKernelError> {
        let func = self.load_function(device, library_name, name)?;
        call_unary(&func, input, output, length);
        Ok(())
    }
}

fn call_unary(_func: &Function, _input: &Buffer, _output: &Buffer, _length: usize) {
    todo!("Call unary");
}

pub fn void_ptr<T>(v: &T) -> *const c_void {
    (v as *const T).cast()
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use metal::{
        CompileOptions, ComputePipelineDescriptor, Device, MTLResourceOptions, MTLSize, NSUInteger,
    };
    use std::mem;

    fn device() -> Device {
        Device::system_default().unwrap()
    }

    fn approx(v: Vec<f32>, digits: i32) -> Vec<f32> {
        let b = 10f32.powi(digits);
        v.iter().map(|t| f32::round(t * b) / b).collect()
    }

    fn approx_f16(v: Vec<f16>, digits: i32) -> Vec<f32> {
        let b = 10f32.powi(digits);
        v.iter().map(|t| f32::round(t.to_f32() * b) / b).collect()
    }

    fn run_cos<T: Clone>(v: &[T], name: &str) -> Vec<T> {
        let device = device();
        let options = MTLResourceOptions::StorageModeManaged;
        let command_queue = device.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let input = device.new_buffer_with_data(
            v.as_ptr() as *const core::ffi::c_void,
            (v.len() * core::mem::size_of::<T>()) as u64,
            options,
        );
        let output = device.new_buffer((v.len() * core::mem::size_of::<T>()) as u64, options);
        let library = device
            .new_library_with_source(UNARY, &CompileOptions::new())
            .expect("Failed to load unary library");
        let func = library.get_function(&format!("cos_{name}"), None).unwrap();
        let pipeline_state_descriptor = ComputePipelineDescriptor::new();
        pipeline_state_descriptor.set_compute_function(Some(&func));

        let pipeline = device
            .new_compute_pipeline_state_with_function(
                pipeline_state_descriptor.compute_function().unwrap(),
            )
            .unwrap();

        let dim: u32 = v.len() as u32;
        // let num_dims: u32 = 1;
        // let info = [v.len() as u32, 1];

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);

        encoder.set_bytes(0, 4, void_ptr(&dim));

        encoder.set_buffer(1, Some(&input), 0);
        encoder.set_buffer(2, Some(&output), 0);

        let width = v.len() as NSUInteger;

        let thread_group_count = MTLSize {
            width,
            height: 1,
            depth: 1,
        };

        let thread_group_size = MTLSize {
            width: pipeline.max_total_threads_per_threadgroup(),
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        output.read_to_vec::<T>(v.len())
    }

    #[test]
    fn cos_f32() {
        let v = vec![1.0f32, 2.0, 3.0];
        let results = run_cos(&v, "float");
        let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
        assert_eq!(approx(results, 4), vec![0.5403, -0.4161, -0.99]);
        assert_eq!(approx(expected, 4), vec![0.5403, -0.4161, -0.99]);
    }

    #[test]
    fn affine() {
        let device = device();

        let options = CompileOptions::new();
        let library = device.new_library_with_source(AFFINE, &options).unwrap();

        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let output = [2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let dim: u32 = 8;
        let num_dims: u32 = 4;
        let info = [1u32, 2, 3];
        let mul: f32 = 1.5;
        let add: f32 = 1.1;

        let function = library.get_function("affine", None).unwrap();
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .unwrap();
        let options = MTLResourceOptions::StorageModeManaged;

        let command_queue = device.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        let input_size = (input.len() * mem::size_of::<f32>()) as NSUInteger;
        let output_size = (output.len() * mem::size_of::<f32>()) as NSUInteger;

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_threadgroup_memory_length(0, output_size as NSUInteger);

        let inputs_buffer = device.new_buffer_with_data(void_ptr(&input), input_size, options);
        let outputs_buffer = device.new_buffer_with_data(void_ptr(&output), output_size, options);

        encoder.set_bytes(0, 4, void_ptr(&dim));
        encoder.set_bytes(1, 4, void_ptr(&num_dims));
        encoder.set_bytes(2, 4, void_ptr(&info));

        encoder.set_buffer(3, Some(&inputs_buffer), 0);
        encoder.set_buffer(4, Some(&outputs_buffer), 0);

        encoder.set_bytes(5, 4, void_ptr(&mul));
        encoder.set_bytes(6, 4, void_ptr(&add));

        let grid_size = MTLSize {
            width: output.len() as NSUInteger,
            height: 1,
            depth: 1,
        };

        let thread_group_size = MTLSize {
            width: pipeline.max_total_threads_per_threadgroup(),
            height: 1,
            depth: 1,
        };

        encoder.dispatch_threads(grid_size, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let expected = vec![2.6, 4.1, 5.6, 7.1, 8.6, 10.1, 11.6, 13.1];
        let result = outputs_buffer.read_to_vec::<f32>(output.len());
        println!("Result {:?}", result.as_ptr());
        assert_eq!(result, expected);
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
        let dst_dim_size: u32 = 15;
        let left_size: u32 = 3;
        let right_size: u32 = 3;

        let function = library.get_function("ia_u32_f32", None).unwrap();
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .unwrap();
        let options = MTLResourceOptions::StorageModeManaged;

        let command_queue = device.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        let ids_size = (index.len() * mem::size_of::<u32>()) as NSUInteger;
        let input_size = (left.len() * mem::size_of::<f32>()) as NSUInteger;
        let output_size = (right.len() * mem::size_of::<f32>()) as NSUInteger;

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_threadgroup_memory_length(0, output_size as NSUInteger);

        let index_buffer = device.new_buffer_with_data(void_ptr(&index), ids_size, options);
        let inputs_buffer = device.new_buffer_with_data(void_ptr(&left), input_size, options);
        let outputs_buffer = device.new_buffer_with_data(void_ptr(&right), output_size, options);

        encoder.set_buffer(0, Some(&index_buffer), 0);
        encoder.set_buffer(1, Some(&inputs_buffer), 0);
        encoder.set_buffer(2, Some(&outputs_buffer), 0);

        encoder.set_bytes(3, 4, void_ptr(&ids_dim_size));
        encoder.set_bytes(4, 4, void_ptr(&left_size));
        encoder.set_bytes(5, 4, void_ptr(&dst_dim_size));
        encoder.set_bytes(6, 4, void_ptr(&right_size));

        let grid_size = MTLSize {
            width: right.len() as NSUInteger,
            height: 1,
            depth: 1,
        };

        let thread_group_size = MTLSize {
            width: pipeline.max_total_threads_per_threadgroup(),
            height: 1,
            depth: 1,
        };

        encoder.dispatch_threads(grid_size, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let expected = vec![
            2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 8.0, 9.0, 10.0, 1.0, 1.0, 1.0, 5.0, 6.0, 7.0,
        ];
        let result = outputs_buffer.read_to_vec::<f32>(right.len());
        println!("Result {:?}", result.as_ptr());
        assert_eq!(result, expected);
    }

    #[test]
    fn cos_f16() {
        let v: Vec<f16> = [1.0f32, 2.0, 3.0]
            .iter()
            .map(|v| f16::from_f32(*v))
            .collect();
        let results = run_cos(&v, "half");
        let expected: Vec<f16> = v.iter().map(|v| f16::from_f32(v.to_f32().cos())).collect();
        assert_eq!(approx_f16(results, 4), vec![0.54, -0.4165, -0.9902]);
        assert_eq!(approx_f16(expected, 4), vec![0.5405, -0.4163, -0.9902]);
    }
}
