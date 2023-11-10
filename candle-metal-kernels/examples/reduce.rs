use candle_metal_kernels::{call_reduce_contiguous, Kernels};
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

    let reduce_kernels = ["fast_sum_float", "softmax_float"];

    println!(
        "{0: <5} | {1: <19} | {2: <6} | {3: <5} | {4: <12} | {5}",
        "dtype", "kernel", "size", "runs", "total time", "avg time"
    );

    // f32
    run_reduce_bench(&device, &kernels, &f32_1k, reduce_kernels);
    run_reduce_bench(&device, &kernels, &f32_10k, reduce_kernels);
    run_reduce_bench(&device, &kernels, &f32_100k, reduce_kernels);
}

fn run_reduce_bench<T: Clone>(
    device: &Device,
    kernels: &Kernels,
    v: &[T],
    reduce_kernels: [&'static str; 2],
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
    // Ghost pass to ensure kernel load time is not included in benchmarks
    for kernel_name in reduce_kernels {
        autoreleasepool(|| {
            let command_buffer = command_queue.new_command_buffer();
            call_reduce_contiguous(
                &device,
                command_buffer,
                &kernels,
                kernel_name,
                v.len(),
                v.len(),
                &input,
                &mut output,
            )
            .unwrap();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        });
    }
    for kernel_name in reduce_kernels {
        let total_time = autoreleasepool(|| {
            let command_buffer = command_queue.new_command_buffer();
            let start = Instant::now();
            for _ in 0..iterations {
                call_reduce_contiguous(
                    &device,
                    command_buffer,
                    &kernels,
                    kernel_name,
                    v.len(),
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
            "{0: <5} | {1: <19} | {2: <6} | {3: <5} | {4: <12?} | {5:?}",
            type_name::<T>().split("::").last().unwrap(),
            kernel_name.to_string(),
            v.len(),
            iterations,
            total_time,
            total_time / iterations
        );
    }
}
