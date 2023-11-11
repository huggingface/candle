use candle_metal_kernels::{call_cast_contiguous, Kernels};
use metal::objc::rc::autoreleasepool;
use metal::{Device, MTLResourceOptions};
use rand;
use std::any::type_name;
use std::time::Instant;

fn main() {
    let device = Device::system_default().unwrap();
    let kernels = Kernels::new();

    let f32_1k = (0..1000).map(|_| rand::random::<f32>()).collect::<Vec<_>>();
    let f32_10k = (0..10000)
        .map(|_| rand::random::<f32>())
        .collect::<Vec<_>>();
    let f32_100k = (0..100000)
        .map(|_| rand::random::<f32>())
        .collect::<Vec<_>>();

    let contiguous_kernels = ["cast_u32_f32"];

    println!(
        "{0: <5} | {1: <19} | {2: <6} | {3: <5} | {4: <11} | {5: <11}",
        "dtype", "kernel", "size", "runs", "total time", "avg time"
    );

    // f32
    run_cast_bench(&device, &kernels, &f32_1k, &contiguous_kernels);
    run_cast_bench(&device, &kernels, &f32_10k, &contiguous_kernels);
    run_cast_bench(&device, &kernels, &f32_100k, &contiguous_kernels);
}

fn run_cast_bench<T: Clone>(
    device: &Device,
    kernels: &Kernels,
    v: &[T],
    contiguous: &[&'static str],
) {
    let command_queue = device.new_command_queue();
    let options = MTLResourceOptions::StorageModeManaged;

    let iterations = 1000;
    let input = device.new_buffer_with_data(
        v.as_ptr() as *const core::ffi::c_void,
        core::mem::size_of_val(v) as u64,
        options,
    );
    let mut output = device.new_buffer(core::mem::size_of_val(v) as u64, options);

    // Contiguous
    for kernel_name in contiguous {
        let total_time = autoreleasepool(|| {
            let command_buffer = command_queue.new_command_buffer();
            let start = Instant::now();
            for _ in 0..iterations {
                call_cast_contiguous(
                    device,
                    &command_buffer,
                    kernels,
                    kernel_name,
                    v.len(),
                    &input,
                    &mut output,
                )
                .unwrap();
            }
            command_buffer.commit();
            command_buffer.wait_until_completed();

            start.elapsed()
        });
        println!(
            "{0: <5} | {1: <19} | {2: <6} | {3: <5} | {4: <11?} | {5: <11?}",
            type_name::<T>().split("::").last().unwrap(),
            kernel_name.to_string(),
            v.len(),
            iterations,
            total_time,
            total_time / iterations
        );
    }

    // Strided?
}
