use candle_metal_kernels::{call_fill, Kernels};
use half::{bf16, f16};
use metal::objc::rc::autoreleasepool;
use metal::{Device, MTLResourceOptions, NSUInteger};
use std::any::type_name;
use std::time::Instant;

fn main() {
    let device = Device::system_default().unwrap();
    let kernels = Kernels::new();
    println!(
        "{: <10} | {: <6} | {: <5} | {: <12} | {}",
        "dtype", "size", "runs", "total time", "avg time"
    );

    run_fill_benches(&device, &kernels, "fill_u8", 0u8);
    run_fill_benches(&device, &kernels, "fill_u32", 0u32);
    run_fill_benches(&device, &kernels, "fill_i64", 0i64);
    run_fill_benches(&device, &kernels, "fill_bf16", bf16::from_f32(0.0));
    run_fill_benches(&device, &kernels, "fill_f16", f16::from_f32(0.0));
    run_fill_benches(&device, &kernels, "fill_f32", 0f32);
}

fn run_fill_benches<T: Copy>(device: &Device, kernels: &Kernels, kernel_name: &'static str, v: T) {
    run_fill_bench(device, kernels, kernel_name, v, 1000);
    run_fill_bench(device, kernels, kernel_name, v, 10000);
    run_fill_bench(device, kernels, kernel_name, v, 100000);
}

fn run_fill_bench<T: Copy>(
    device: &Device,
    kernels: &Kernels,
    kernel_name: &'static str,
    v: T,
    elem_count: usize,
) {
    let command_queue = device.new_command_queue();
    let options = MTLResourceOptions::StorageModeManaged;

    let iterations = 10000;

    let buffer_size = (elem_count * core::mem::size_of_val(&v)) as NSUInteger;
    // debug!("Allocate 1 - buffer size {size}");
    let mut buffer = device.new_buffer(buffer_size, MTLResourceOptions::StorageModeShared);

    // Ghost pass to ensure kernel load time is not included in benchmarks
    autoreleasepool(|| {
        let command_buffer = command_queue.new_command_buffer();
        candle_metal_kernels::call_fill(
            device,
            command_buffer,
            kernels,
            kernel_name,
            elem_count,
            &mut buffer,
            v,
        )
        .unwrap();

        command_buffer.commit();
        command_buffer.wait_until_completed();
    });

    let total_time = autoreleasepool(|| {
        let command_buffer = command_queue.new_command_buffer();
        let start = Instant::now();
        for _ in 0..iterations {
            candle_metal_kernels::call_fill(
                device,
                command_buffer,
                kernels,
                kernel_name,
                elem_count,
                &mut buffer,
                v,
            )
            .unwrap();
        }
        command_buffer.commit();
        command_buffer.wait_until_completed();

        start.elapsed()
    });
    println!(
        "{: <10} | {: <6} | {: <5} | {: <12?} | {:?}",
        type_name::<T>().split("::").last().unwrap(),
        elem_count,
        iterations,
        total_time,
        total_time / iterations
    );
}
