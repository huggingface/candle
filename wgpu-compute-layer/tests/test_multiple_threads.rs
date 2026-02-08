use std::{borrow::Cow, thread};

use wgpu_compute_layer::{DType, LoaderIndex, PipelineIndex, ShaderIndex, WgpuDevice};



#[derive(Debug)]
struct ThreadTestLoader;

wgpu_compute_layer::create_loader!(ThreadTestLoader);

fn create_dummy_shader_index() -> ShaderIndex {
    ShaderIndex::new(LoaderIndex(ThreadTestLoader::LOADER_INDEX.0), 0)
}

fn create_dummy_pipeline_index(index: u8) -> PipelineIndex {
    PipelineIndex::new(create_dummy_shader_index(), index)
}

impl wgpu_compute_layer::ShaderLoader for ThreadTestLoader {
    fn load(&self, _index: wgpu_compute_layer::ShaderIndex, _defines : &[(&str, String)]) -> Cow<'_, str> {
        "
        @group(0) @binding(0) var<storage, read_write> output : array<u32>;
        @group(0) @binding(1) var<storage> op_meta : array<u32>;
        @group(0) @binding(2) var<storage> input : array<u32>;
        @compute @workgroup_size(1) fn main() { output[0] += input[0] * 71;}
        @compute @workgroup_size(1) fn set_zero() { output[0] = 0;}
        ".into()
    }

    fn get_entry_point(&self, index: PipelineIndex) -> &str {
        match index.1 {
            0 => "main",
            1 => "set_zero",
            _ => panic!("Invalid pipeline index"),
        }
    }
}


fn create_device() -> WgpuDevice {
    let dev = WgpuDevice::create(Default::default()).expect("create device");
    dev.add_wgpu_shader_loader(ThreadTestLoader::LOADER_INDEX, || ThreadTestLoader {});
    dev
}

#[test]
fn test_multi_thread() {
    let dev = create_device();

    let mut handles = Vec::new();

    for i in 0..1000 {
        let dev= dev.clone();
        let handle = thread::spawn(move || {
            let output = dev.alloc_uninit_size(DType::U32, 1);
            let input = dev.alloc_from_slice(DType::U32, &[i]).expect("could not create buffer");
            {
                let mut q = dev.get_queue(); //set output to zero, the first buffer will be zero, but if a thread uses a reused buffer of onother thread it will not be zero
                let pipeline = q.get_pipeline(create_dummy_pipeline_index(1));
                let bind = dev.create_bind_group_input1(output.buffer(), input.buffer(), DType::U32.into());
                q.enqueue_workgroups(pipeline, bind, 1, 1, 1, 100000000); //we have a high workload_size, so we do not send all commands into one command buffer, but have to split them into multiple command buffers
            }
           
            for _ in 0..100{
                let mut q = dev.get_queue();
                let pipeline = q.get_pipeline(create_dummy_pipeline_index(0));

                let bind = dev.create_bind_group_input1(output.buffer(), input.buffer(), DType::U32.into());
                q.enqueue_workgroups(pipeline, bind, 1, 1, 1, 100000000); //we have a high workload_size, so we do not send all commands into one command buffer, but have to split them into multiple command buffers
            }
            let result : Vec<u32> = pollster::block_on(output.read_from_buffer_reference_async()).expect("could not read buffer");
            assert_eq!(result.len(), 1);
            assert_eq!(result[0], 100 * i * 71);
        });

        handles.push(handle);
    }

    for h in handles {
        match h.join(){
            Ok(_) => {},
            Err(e) => {
                if let Some(s) = e.downcast_ref::<&str>() {
                    println!("Thread panicked with message: {}", s);
                } else if let Some(s) = e.downcast_ref::<String>() {
                    println!("Thread panicked with message: {}", s);
                } else {
                    println!("Thread panicked with non-string payload");
                }
                panic!("Thread panicked");
            }
        }
    }
}
