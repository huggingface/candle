use candle_metal_kernels::{call_unary_contiguous, call_unary_strided, unary, Kernels};
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
        unary::contiguous::sin::FLOAT,
        unary::contiguous::cos::FLOAT,
        unary::contiguous::exp::FLOAT,
        unary::contiguous::sqr::FLOAT,
        unary::contiguous::sqrt::FLOAT,
        unary::contiguous::neg::FLOAT,
        unary::contiguous::copy::FLOAT,
    ];
    let f32_skernels = [
        unary::strided::sin::FLOAT,
        unary::strided::cos::FLOAT,
        unary::strided::exp::FLOAT,
        unary::strided::sqr::FLOAT,
        unary::strided::sqrt::FLOAT,
        unary::strided::neg::FLOAT,
        unary::strided::copy::FLOAT,
    ];
    let f16_ckernels = [
        unary::contiguous::sin::HALF,
        unary::contiguous::cos::HALF,
        unary::contiguous::exp::HALF,
        unary::contiguous::sqr::HALF,
        unary::contiguous::sqrt::HALF,
        unary::contiguous::neg::HALF,
        unary::contiguous::copy::HALF,
    ];
    let f16_skernels = [
        unary::strided::sin::HALF,
        unary::strided::cos::HALF,
        unary::strided::exp::HALF,
        unary::strided::sqr::HALF,
        unary::strided::sqrt::HALF,
        unary::strided::neg::HALF,
        unary::strided::copy::HALF,
    ];
    let bf16_ckernels = [
        unary::contiguous::sin::BFLOAT,
        unary::contiguous::cos::BFLOAT,
        unary::contiguous::exp::BFLOAT,
        unary::contiguous::sqr::BFLOAT,
        unary::contiguous::sqrt::BFLOAT,
        unary::contiguous::neg::BFLOAT,
        unary::contiguous::copy::BFLOAT,
    ];
    let bf16_skernels = [
        unary::strided::sin::BFLOAT,
        unary::strided::cos::BFLOAT,
        unary::strided::exp::BFLOAT,
        unary::strided::sqr::BFLOAT,
        unary::strided::sqrt::BFLOAT,
        unary::strided::neg::BFLOAT,
        unary::strided::copy::BFLOAT,
    ];

    println!(
        "{0: <5} | {1: <19} | {2: <6} | {3: <5} | {4: <11} | {5: <11}",
        "dtype", "kernel", "size", "runs", "total time", "avg time"
    );

    // f32
    run_unary_bench(&device, &kernels, &f32_1k, f32_ckernels, f32_skernels);
    run_unary_bench(&device, &kernels, &f32_10k, f32_ckernels, f32_skernels);
    run_unary_bench(&device, &kernels, &f32_100k, f32_ckernels, f32_skernels);

    // f16
    run_unary_bench(&device, &kernels, &f16_1k, f16_ckernels, f16_skernels);
    run_unary_bench(&device, &kernels, &f16_10k, f16_ckernels, f16_skernels);
    run_unary_bench(&device, &kernels, &f16_100k, f16_ckernels, f16_skernels);

    // bf16
    run_unary_bench(&device, &kernels, &bf16_1k, bf16_ckernels, bf16_skernels);
    run_unary_bench(&device, &kernels, &bf16_10k, bf16_ckernels, bf16_skernels);
    run_unary_bench(&device, &kernels, &bf16_100k, bf16_ckernels, bf16_skernels);
}

fn run_unary_bench<T: Clone>(
    device: &Device,
    kernels: &Kernels,
    v: &[T],
    contiguous: [unary::contiguous::Kernel; 7],
    strided: [unary::strided::Kernel; 7],
) {
    let command_queue = device.new_command_queue();
    let options = MTLResourceOptions::StorageModeManaged;

    let iterations = 10000;
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
                call_unary_contiguous(
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
            kernel_name.0,
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
    for kernel_name in &strided {
        let total_time = autoreleasepool(|| {
            let command_buffer = command_queue.new_command_buffer();
            let start = Instant::now();
            for _ in 0..iterations {
                call_unary_strided(
                    device,
                    command_buffer,
                    &kernels,
                    kernel_name,
                    &shape,
                    &input,
                    &strides,
                    offset,
                    &mut output,
                    0,
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
            kernel_name.0,
            v.len(),
            iterations,
            total_time,
            total_time / iterations
        );
    }
}
