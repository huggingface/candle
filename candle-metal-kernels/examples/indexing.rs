use candle_metal_kernels::{call_index_add, Kernels};
use half::{bf16, f16};
use metal::objc::rc::autoreleasepool;
use metal::{Device, MTLResourceOptions, NSUInteger};
use rand;
use std::any::type_name;
use std::time::Instant;

macro_rules! ia_name {
    ($idx_type:literal, $dtype:literal) => {
        concat!("ia_", $idx_type, "_", $dtype)
    };
}

macro_rules! ia_kernels {
    ($dtype:literal) => {
        [
            ia_name!("u8", $dtype),
            ia_name!("u32", $dtype),
            ia_name!("i64", $dtype),
        ]
    };
}

struct IdsData {
    u8: Vec<u8>,
    u32: Vec<u32>,
    i64: Vec<i64>,
}

fn main() {
    let device = Device::system_default().unwrap();

    let kernels = Kernels::new();
    let u8_1k = (0..1000).map(|_| rand::random::<u8>()).collect::<Vec<_>>();
    let u8_10k = (0..10000).map(|_| rand::random::<u8>()).collect::<Vec<_>>();

    let u8_100k = (0..100000)
        .map(|_| rand::random::<u8>())
        .collect::<Vec<_>>();
    let f32_map = |v: &[u8]| v.iter().map(|v| *v as f32).collect::<Vec<_>>();
    let f32_1k = f32_map(&u8_1k);
    let f32_10k = f32_map(&u8_10k);

    let f32_100k = f32_map(&u8_100k);
    let f16_map = |v: &[f32]| v.iter().map(|v| f16::from_f32(*v)).collect::<Vec<_>>();
    let f16_1k = f16_map(&f32_1k);
    let f16_10k = f16_map(&f32_10k);

    let f16_100k = f16_map(&f32_100k);
    let bf16_map = |v: &[f32]| v.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>();
    let bf16_1k = bf16_map(&f32_1k);
    let bf16_10k = bf16_map(&f32_10k);

    let bf16_100k = bf16_map(&f32_100k);
    let u32_map = |v: &[u8]| v.iter().map(|v| *v as u32).collect::<Vec<_>>();
    let u32_1k = u32_map(&u8_1k);
    let u32_10k = u32_map(&u8_10k);

    let u32_100k = u32_map(&u8_100k);
    let i64_map = |v: &[u8]| v.iter().map(|v| *v as i64).collect::<Vec<_>>();
    let i64_1k = i64_map(&u8_1k);
    let i64_10k = i64_map(&u8_10k);

    let i64_100k = i64_map(&u8_100k);

    let f32_kernels = ia_kernels!("f32");
    let f16_kernels = ia_kernels!("f16");
    let bf16_kernels = ia_kernels!("bf16");
    let u32_kernels = ia_kernels!("u32");
    let i64_kernels = ia_kernels!("u32");

    println!(
        "{0: <5} | {1: <11} | {2: <6} | {3: <5} | {4: <12} | {5}",
        "dtype", "kernel", "size", "runs", "total time", "avg time"
    );

    let ids_data_1k = IdsData {
        u8: u8_1k.clone(),
        u32: u32_1k.clone(),
        i64: i64_1k.clone(),
    };
    let ids_data_10k = IdsData {
        u8: u8_10k.clone(),
        u32: u32_10k.clone(),
        i64: i64_10k.clone(),
    };
    let ids_data_100k = IdsData {
        u8: u8_100k.clone(),
        u32: u32_100k.clone(),
        i64: i64_100k.clone(),
    };

    // f32
    run_indexing_benches(&device, &kernels, &ids_data_1k, &f32_1k, f32_kernels);
    run_indexing_benches(&device, &kernels, &ids_data_10k, &f32_10k, f32_kernels);
    run_indexing_benches(&device, &kernels, &ids_data_100k, &f32_100k, f32_kernels);

    // f16
    run_indexing_benches(&device, &kernels, &ids_data_1k, &f16_1k, f16_kernels);
    run_indexing_benches(&device, &kernels, &ids_data_10k, &f16_10k, f16_kernels);
    run_indexing_benches(&device, &kernels, &ids_data_100k, &f16_100k, f16_kernels);

    // bf16
    run_indexing_benches(&device, &kernels, &ids_data_1k, &bf16_1k, bf16_kernels);
    run_indexing_benches(&device, &kernels, &ids_data_10k, &bf16_10k, bf16_kernels);
    run_indexing_benches(&device, &kernels, &ids_data_100k, &bf16_100k, bf16_kernels);

    // u8
    run_indexing_benches(&device, &kernels, &ids_data_1k, &u8_1k, u32_kernels);
    run_indexing_benches(&device, &kernels, &ids_data_10k, &u8_10k, u32_kernels);
    run_indexing_benches(&device, &kernels, &ids_data_100k, &u8_100k, u32_kernels);

    // u32
    run_indexing_benches(&device, &kernels, &ids_data_1k, &u32_1k, u32_kernels);
    run_indexing_benches(&device, &kernels, &ids_data_10k, &u32_10k, u32_kernels);
    run_indexing_benches(&device, &kernels, &ids_data_100k, &u32_100k, u32_kernels);

    // i64
    run_indexing_benches(&device, &kernels, &ids_data_1k, &i64_1k, i64_kernels);
    run_indexing_benches(&device, &kernels, &ids_data_10k, &i64_10k, i64_kernels);
    run_indexing_benches(&device, &kernels, &ids_data_100k, &i64_100k, i64_kernels);
}

fn run_indexing_benches<T: Clone>(
    device: &Device,
    kernels: &Kernels,
    ids: &IdsData,
    input: &[T],
    index_add_kernels: [&'static str; 3],
) {
    run_indexing_bench(device, kernels, &ids.u8, input, index_add_kernels[0]);
    run_indexing_bench(device, kernels, &ids.u32, input, index_add_kernels[1]);
    run_indexing_bench(device, kernels, &ids.i64, input, index_add_kernels[2]);
}

fn run_indexing_bench<T: Clone, U: Clone>(
    device: &Device,
    kernels: &Kernels,
    ids: &[T],
    input: &[U],
    kernel_name: &'static str,
) {
    let command_queue = device.new_command_queue();
    let options = MTLResourceOptions::StorageModeManaged;

    let iterations = 10000;
    let ids_buffer = device.new_buffer_with_data(
        ids.as_ptr() as *const core::ffi::c_void,
        core::mem::size_of_val(ids) as u64,
        options,
    );
    let input_buffer = device.new_buffer_with_data(
        input.as_ptr() as *const core::ffi::c_void,
        core::mem::size_of_val(input) as u64,
        options,
    );
    let mut output_buffer = device.new_buffer(core::mem::size_of_val(input) as u64, options);

    let ids_dim_size = ids.len() as NSUInteger;
    let left_size = input.len() as NSUInteger;
    let dst_dim_size = ids.len() as NSUInteger;
    let right_size = input.len() as NSUInteger;

    // Ghost pass to ensure kernel load time is not included in benchmarks
    autoreleasepool(|| {
        let command_buffer = command_queue.new_command_buffer();
        call_index_add(
            device,
            &command_buffer,
            kernels,
            kernel_name,
            &ids_buffer,
            &input_buffer,
            &mut output_buffer,
            ids_dim_size,
            left_size,
            dst_dim_size,
            right_size,
        )
        .unwrap();

        command_buffer.commit();
        command_buffer.wait_until_completed();
    });

    let total_time = autoreleasepool(|| {
        let command_buffer = command_queue.new_command_buffer();
        let start = Instant::now();
        for _ in 0..iterations {
            call_index_add(
                device,
                &command_buffer,
                kernels,
                kernel_name,
                &ids_buffer,
                &input_buffer,
                &mut output_buffer,
                ids_dim_size,
                left_size,
                dst_dim_size,
                right_size,
            )
            .unwrap();
        }
        command_buffer.commit();
        command_buffer.wait_until_completed();

        start.elapsed()
    });
    println!(
        "{0: <5} | {1: <11} | {2: <6} | {3: <5} | {4: <12?} | {5:?}",
        type_name::<U>().split("::").last().unwrap(),
        kernel_name.to_string(),
        input.len(),
        iterations,
        total_time,
        total_time / iterations
    );
}
