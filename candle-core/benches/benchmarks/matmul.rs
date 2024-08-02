use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle_core::{wgpu::MatmulAlgorithm, DType, Device, Tensor};
use criterion::{black_box, criterion_group, Criterion, Throughput};
use std::time::Instant;
//use tracing_subscriber::{self, fmt::format};

// fn run(a: &Tensor, b: &Tensor) {
//     a.matmul(&b.t().unwrap()).unwrap();
// }

fn run(a: &Tensor, b: &Tensor) {
    a.matmul(&b).unwrap();
}

const ID : &'static str = "matmul7_16x16";

fn run_bench(c: &mut Criterion, device: &Device) {

    let b = 1;
    let m = 1;
    let n = 2049;
    let k = 2049;

    let dtype = DType::F32;
    let lhs = Tensor::zeros((b, m, k), dtype, device).unwrap();
    let rhs = Tensor::zeros((b, k, n), dtype, device).unwrap();

    let flops = b * m * n * k;

    let mut group = c.benchmark_group(device.bench_name("matmul"));
    group.throughput(Throughput::Bytes(flops as u64));
    
   
    if device.is_webgpu(){
        if let Device::WebGpu(wgpu) = device{
            for alg in vec![MatmulAlgorithm::Matmul1, MatmulAlgorithm::Matmul5_16_64]{
                *(wgpu.matmul_alg.lock().unwrap()) = alg.clone(); 
                group.bench_function(format!("{:?}", alg), |b| {
                    b.iter_custom(|iters| {
                        let start = Instant::now();
                        for _i in 0..iters {
                            run(black_box(&lhs), black_box(&rhs));
                        }
                        device.sync().unwrap();
                        start.elapsed()
                    })
                });
            }           
        }
    }
    else{
        group.bench_function(ID, |b| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _i in 0..iters {
                    run(black_box(&lhs), black_box(&rhs));
                }
                device.sync().unwrap();
                start.elapsed()
            })
        });
    }

    
    group.finish();
    #[cfg(feature = "wgpu_debug")]
    match device {
        candle_core::Device::WebGpu(gpu) => {
            let info = pollster::block_on(gpu.get_debug_info()).unwrap();
            let map2 = candle_core::wgpu::debug_info::calulate_measurment(&info);
            candle_core::wgpu::debug_info::save_list(&map2, "wgpu_bench.json").unwrap();
        },
        _ => {},
    };

    let b = 1;
    let m = 2048;
    let n = 2048;
    let k = 2048;

    let dtype = DType::F32;
    let lhs = Tensor::zeros((b, m, k), dtype, device).unwrap();
    let rhs = Tensor::zeros((b, k, n), dtype, device).unwrap();

    let flops = b * m * n * k;

    let mut group = c.benchmark_group(device.bench_name("matmul2048"));
    group.throughput(Throughput::Bytes(flops as u64));
    if device.is_webgpu(){
        if let Device::WebGpu(wgpu) = device{
            for alg in vec![MatmulAlgorithm::Matmul1, MatmulAlgorithm::Matmul5, MatmulAlgorithm::Matmul5_32_32, MatmulAlgorithm::Matuml5_64_64, MatmulAlgorithm::Matmul5_64_64_8_8, MatmulAlgorithm::Matmul5_16_64]{
                *(wgpu.matmul_alg.lock().unwrap()) = alg.clone(); 
                group.bench_function(format!("{:?}", alg), |b| {
                    b.iter_custom(|iters| {
                        let start = Instant::now();
                        for _i in 0..iters {
                            run(black_box(&lhs), black_box(&rhs));
                        }
                        device.sync().unwrap();
                        start.elapsed()
                    })
                });
            }           
        }
    }
    else{
        group.bench_function(ID, |b| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _i in 0..iters {
                    run(black_box(&lhs), black_box(&rhs));
                }
                device.sync().unwrap();
                start.elapsed()
            })
        });
    }
    group.finish();

    #[cfg(feature = "wgpu_debug")]
    match device {
        candle_core::Device::WebGpu(gpu) => {
            let info = pollster::block_on(gpu.get_debug_info()).unwrap();
            let map2 = candle_core::wgpu::debug_info::calulate_measurment(&info);
            candle_core::wgpu::debug_info::save_list(&map2, "wgpu_bench2048.json").unwrap();
        },
        _ => {},
    };
}

fn criterion_benchmark(c: &mut Criterion) {
    let handler = BenchDeviceHandler::new().unwrap();
    for device in handler.devices {
        //if(device.is_webgpu()){
            run_bench(c, &device);
        //}
    }
}

criterion_group!(benches, criterion_benchmark);
