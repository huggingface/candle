use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle_core::{DType, Device, Tensor};
use criterion::{black_box, criterion_group, Criterion};
use std::time::Instant;


fn run(input: Tensor, weight: Tensor) {
    input.conv2d(&weight, 1, 1,1 ,1).unwrap();
}

fn run_conv2d_benchmark(c: &mut Criterion, device: &Device, dtype: DType, name: &str) {
   
    // const B: usize = 1;
    // const C: usize = 320;
    // const C_OUT : usize = 640;
    // const M: usize = 128;
    // const K: usize = 128;
    // const K_SIZE: usize = 2;
   
    const B: usize = 2;
    const C: usize = 1;
    const C_OUT : usize = 1;
    const M: usize = 24;
    const K: usize = 24;
    const K_SIZE: usize = 3;
   

    let weight = Tensor::ones((C_OUT, C, K_SIZE, K_SIZE), dtype, device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

    // let weight = Tensor::ones((C_OUT, K_SIZE, K_SIZE, C), dtype, device)
    //     .unwrap()
    //     .to_dtype(dtype)
    //     .unwrap()
    //     .transpose(3, 1).unwrap();
   
    //let bias = Tensor::zeros(K, dtype, device).unwrap();
    let input = Tensor::ones((B, C, M, K), dtype, device).unwrap();
    //let input = Tensor::ones((B, K,M, C), dtype, device).unwrap().transpose(1, 3).unwrap().transpose(2, 3).unwrap();


    println!("weight: {:?}", weight.layout());
    println!("input: {:?}", input.layout());

    let flops = B * C * M * K * K_SIZE * K_SIZE; 

    let mut group = c.benchmark_group(device.bench_name(name));
    //group.sample_size(10);
    group.throughput(criterion::Throughput::Bytes(flops as u64*4));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run(
                    black_box(input.clone()),
                    black_box(weight.clone())
                );
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    let device = BenchDeviceHandler::new().unwrap();
    for d in device.devices {
        run_conv2d_benchmark(c, &d, DType::F32, "conv2d_f32");
        if d.is_dtype_available(DType::F16){
            run_conv2d_benchmark(c, &d, DType::F16, "conv2d_f16");
        }
    }
}

criterion_group!(benches, criterion_benchmark);
