use candle_metal_kernels::{call_where_cond_strided, Kernels};
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

    let f32_strided_k = [
        "where_i64_f32",
        "where_u32_f32",
        "where_u8_f32",
        "where_i64_f16",
        "where_u32_f16",
        "where_u8_f16",
        "where_i64_u8",
        "where_u32_u8",
        "where_u8_u8",
        "where_i64_u32",
        "where_u32_u32",
        "where_u8_u32",
        "where_i64_i64",
        "where_u32_i64",
        "where_u8_i64",
    ];

    println!(
        "{0: <5} | {1: <19} | {2: <6} | {3: <5} | {4: <12} | {5:}",
        "dtype", "kernel", "size", "runs", "total time", "avg time"
    );

    // f32
    run_ternary_bench(&device, &kernels, &f32_1k, f32_strided_k);
    run_ternary_bench(&device, &kernels, &f32_10k, f32_strided_k);
    run_ternary_bench(&device, &kernels, &f32_100k, f32_strided_k);
}

fn run_ternary_bench<T: Clone>(
    device: &Device,
    kernels: &Kernels,
    v: &[T],
    // contiguous: [&'static str; 1],
    strided: [&'static str; 15],
) {
    let command_queue = device.new_command_queue();
    let options = MTLResourceOptions::StorageModeManaged;
    let iterations = 1000;

    let length = v.len() as i32;

    let shape = vec![2, v.len() / 2];

    // Stride is the biggest factor to the complexity of the kernel
    let stride: Vec<_> = (0..v.len() / 50).map(|i| i).collect();
    let cond = (0..length).map(|i| i % 2).collect::<Vec<_>>();
    let left = (0..length).step_by(1).collect::<Vec<_>>();
    let right = left.iter().map(|v| -*v).collect::<Vec<_>>();

    let cond = device.new_buffer_with_data(
        cond.as_ptr() as *const core::ffi::c_void,
        std::mem::size_of_val(&cond) as u64,
        options,
    );
    let left = device.new_buffer_with_data(
        left.as_ptr() as *const core::ffi::c_void,
        std::mem::size_of_val(&left) as u64,
        options,
    );
    let right = device.new_buffer_with_data(
        right.as_ptr() as *const core::ffi::c_void,
        std::mem::size_of_val(&right) as u64,
        options,
    );
    let mut output = device.new_buffer(core::mem::size_of_val(v) as u64, options);

    // Ghost pass to ensure kernel load time is not included in benchmarks
    for kernel_name in strided {
        autoreleasepool(|| {
            let command_buffer = command_queue.new_command_buffer();
            call_where_cond_strided(
                device,
                command_buffer,
                &kernels,
                &kernel_name,
                &shape,
                &cond,
                (stride.as_slice(), 0),
                &left,
                (stride.as_slice(), 0),
                &right,
                (stride.as_slice(), 0),
                &mut output,
            )
            .unwrap();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        });
    }
    for kernel_name in strided {
        let total_time = autoreleasepool(|| {
            let command_buffer = command_queue.new_command_buffer();
            let start = Instant::now();
            for _ in 0..iterations {
                call_where_cond_strided(
                    device,
                    command_buffer,
                    &kernels,
                    &kernel_name,
                    &shape,
                    &cond,
                    (stride.as_slice(), 0),
                    &left,
                    (stride.as_slice(), 0),
                    &right,
                    (stride.as_slice(), 0),
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
