use candle_metal_kernels::{binary, call_binary_contiguous, call_binary_strided, Kernels};
use half::{bf16, f16};
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

    let f16_map = |v: &[f32]| v.iter().map(|v| f16::from_f32(*v)).collect::<Vec<_>>();
    let f16_1k = f16_map(&f32_1k);
    let f16_10k = f16_map(&f32_10k);
    let f16_100k = f16_map(&f32_100k);

    let bf16_map = |v: &[f32]| v.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>();
    let bf16_1k = bf16_map(&f32_1k);
    let bf16_10k = bf16_map(&f32_10k);
    let bf16_100k = bf16_map(&f32_100k);

    let f32_ckernels = [
        binary::contiguous::add::FLOAT,
        binary::contiguous::sub::FLOAT,
        binary::contiguous::mul::FLOAT,
        binary::contiguous::div::FLOAT,
    ];
    let f32_skernels = [
        binary::strided::add::FLOAT,
        binary::strided::sub::FLOAT,
        binary::strided::mul::FLOAT,
        binary::strided::div::FLOAT,
    ];
    let f16_ckernels = [
        binary::contiguous::add::HALF,
        binary::contiguous::sub::HALF,
        binary::contiguous::mul::HALF,
        binary::contiguous::div::HALF,
    ];
    let f16_skernels = [
        binary::strided::add::HALF,
        binary::strided::sub::HALF,
        binary::strided::mul::HALF,
        binary::strided::div::HALF,
    ];
    let bf16_ckernels = [
        binary::contiguous::add::BFLOAT,
        binary::contiguous::sub::BFLOAT,
        binary::contiguous::mul::BFLOAT,
        binary::contiguous::div::BFLOAT,
    ];
    let bf16_skernels = [
        binary::strided::add::BFLOAT,
        binary::strided::sub::BFLOAT,
        binary::strided::mul::BFLOAT,
        binary::strided::div::BFLOAT,
    ];

    println!(
        "{0: <5} | {1: <19} | {2: <6} | {3: <5} | {4: <11} | {5: <11}",
        "dtype", "kernel", "size", "runs", "total time", "avg time"
    );

    // f32
    run_binary_bench(&device, &kernels, &f32_1k, f32_ckernels, f32_skernels);
    run_binary_bench(&device, &kernels, &f32_10k, f32_ckernels, f32_skernels);
    run_binary_bench(&device, &kernels, &f32_100k, f32_ckernels, f32_skernels);

    // f16
    run_binary_bench(&device, &kernels, &f16_1k, f16_ckernels, f16_skernels);
    run_binary_bench(&device, &kernels, &f16_10k, f16_ckernels, f16_skernels);
    run_binary_bench(&device, &kernels, &f16_100k, f16_ckernels, f16_skernels);

    // bf16
    run_binary_bench(&device, &kernels, &bf16_1k, bf16_ckernels, bf16_skernels);
    run_binary_bench(&device, &kernels, &bf16_10k, bf16_ckernels, bf16_skernels);
    run_binary_bench(&device, &kernels, &bf16_100k, bf16_ckernels, bf16_skernels);
}

fn run_binary_bench<T: Clone>(
    device: &Device,
    kernels: &Kernels,
    v: &[T],
    contiguous: [binary::contiguous::Kernel; 4],
    strided: [binary::strided::Kernel; 4],
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
                call_binary_contiguous(
                    device,
                    &command_buffer,
                    kernels,
                    kernel_name,
                    v.len(),
                    &input,
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

    // Strided
    let shape = vec![2, 5_000];
    let strides = vec![2, 1];
    let offset = 0;
    for kernel_name in strided {
        let total_time = autoreleasepool(|| {
            let command_buffer = command_queue.new_command_buffer();
            let start = Instant::now();
            for _ in 0..iterations {
                call_binary_strided(
                    device,
                    command_buffer,
                    &kernels,
                    kernel_name,
                    &shape,
                    &input,
                    &strides,
                    offset,
                    &input,
                    &strides,
                    offset,
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
}
