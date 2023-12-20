use candle_metal_kernels::{call_affine, Kernels};
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

    println!(
        "{0: <5} | {1: <19} | {2: <6} | {3: <5} | {4: <11} | {5: <11}",
        "dtype", "kernel", "size", "runs", "total time", "avg time"
    );

    // f32
    run_affine_bench(&device, &kernels, &f32_1k);
    run_affine_bench(&device, &kernels, &f32_10k);
    run_affine_bench(&device, &kernels, &f32_100k);
}

fn run_affine_bench<T: Clone>(device: &Device, kernels: &Kernels, v: &[T]) {
    let command_queue = device.new_command_queue();
    let options = MTLResourceOptions::StorageModeManaged;

    let iterations = 10000;
    let input = device.new_buffer_with_data(
        v.as_ptr() as *const core::ffi::c_void,
        core::mem::size_of_val(v) as u64,
        options,
    );
    let mut output = device.new_buffer(core::mem::size_of_val(v) as u64, options);

    let mul: f32 = 1.2345;
    let add: f32 = 2.3456;
    let total_time = autoreleasepool(|| {
        let command_buffer = command_queue.new_command_buffer();
        let start = Instant::now();
        for _ in 0..iterations {
            call_affine(
                &device,
                command_buffer,
                &kernels,
                "affine_float",
                v.len(),
                &input,
                &mut output,
                mul,
                add,
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
        "affine",
        v.len(),
        iterations,
        total_time,
        total_time / iterations
    );
}
