use std::time::{Duration, Instant};

use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle_core::{D, DType, Device, Module, Tensor, quantized};
use candle_core::quantized::QMatMul;

#[cfg(feature = "wgpu")]
use candle_core::wgpu::MatmulAlgorithm;

use criterion::{criterion_group, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

const GGLM_TYPE: candle_core::quantized::GgmlDType = candle_core::quantized::GgmlDType::Q8_1;

fn run(a: &Tensor, b: &QMatMul) {
    b.forward(a).unwrap();
}

fn bench_impl(b : &mut criterion::Bencher<'_, criterion::measurement::WallTime>, lhs : &Tensor, rhs : &candle_core::quantized::QMatMul, device : &Device){
    b.iter_custom(|iters| {
        let start = Instant::now();
        for _ in 0..iters {
            run(black_box(lhs), black_box(rhs));
        }
        device.sync().unwrap();
        start.elapsed()
    })
}

fn test_matmul(
    device: &Device,
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    bmnk: (usize, usize, usize, usize),
    size: usize,
    multiple_sizes: bool,
    tp: (bool, bool),
    typ : quantized::GgmlDType
) {
    let (b, m, n, k) = bmnk;
    let (tpa, tpb) = tp;
    let dtype = DType::F32;

    let b = 1;
    let lhs = if tpa {
        Tensor::zeros((k, m), dtype, device)
            .unwrap()
            .transpose(D::Minus1, D::Minus2)
            .unwrap()
    } else {
        Tensor::zeros((m, k), dtype, device).unwrap()
    };

    let rhs = if tpb {
        Tensor::zeros((n, k), dtype, device)
            .unwrap()
            .transpose(D::Minus1, D::Minus2)
            .unwrap()
    } else {
        Tensor::zeros((k, n), dtype, device).unwrap()
    };

    let rhs = quantized::QTensor::quantize(&rhs, typ).unwrap();
    let rhs = quantized::QMatMul::from_qtensor(rhs).unwrap();

    let flops = b * m * n * k;
    group.throughput(Throughput::Bytes(flops as u64));
    group.measurement_time(Duration::from_millis(250));
    group.sample_size(32);
    group.warm_up_time(Duration::from_secs_f32(0.25));
    if device.is_wgpu() {
        #[cfg(feature = "wgpu")]
        if let Device::Wgpu(wgpu) = device {
            {
                const TILE_SIZES: &[u32] = &[32, 64, 128];
                const WPT_VALUES: &[u32] = &[2, 4, 8, 16];

                use candle_core::wgpu::wgpu_functions::matmul::sgemm::GenericDynamicMatmulShaderSettings;

                for &tile_m in TILE_SIZES {
                    for &tile_n in TILE_SIZES {
                        for &tile_k in TILE_SIZES {
                            for &wptm in WPT_VALUES {
                                for &wptn in WPT_VALUES {
                                    use candle_core::wgpu::wgpu_functions::matmul::sgemm::{GenericMatmulSettings, StrideOptimization};
                                    let threads = (tile_m / wptm) * (tile_n / wptn);
                                    if threads % 8 != 0{
                                        continue;
                                    }
                                    let lpta = (tile_k*tile_m)/threads;
                                    if lpta % 4 != 0{
                                        continue;
                                    }
                                    let lptb = (tile_k*tile_n)/threads;
                                    if lptb % 4 != 0{
                                        continue;
                                    }
                                    if threads > 256{
                                        continue;
                                    }
                                    if(tile_k == 128 && (tile_m == 128 || tile_n == 128)){
                                        continue;
                                    }

                                    if tile_k + tile_m + tile_n >= 256 {
                                        continue;
                                    }

                                    //skip small threads as there are prob not the most performant solution
                                    if threads <= 16{ //at least 32 threads
                                        continue;
                                    }

                                    let mut matmul_alg = wgpu.quantized_matmul_alg.lock().unwrap();
                                    *matmul_alg = Some(GenericDynamicMatmulShaderSettings::new(
                                        GenericMatmulSettings::new(
                                            tile_m,
                                            tile_n,
                                            tile_k,
                                            StrideOptimization::None,
                                            StrideOptimization::None,
                                        ),
                                        wptm,
                                        wptn,
                                        false,
                                    ));
                                    drop(matmul_alg); // release lock early

                                    let func_name = device.bench_name(format!(
                                        "matmul_tile_{:?}({},{},{})_wptm{}_wptn{}{}{}",
                                        typ,
                                        tile_m, tile_n, tile_k,
                                        wptm,
                                        wptn,
                                        if tpa { "_tA" } else { "" },
                                        if tpb { "_tB" } else { "" },
                                    ));
                                    
                                    tracing::info!("TEST: {func_name}");
                                    if multiple_sizes {
                                        group.bench_with_input(
                                            BenchmarkId::new(func_name.clone(), size),
                                            &size,
                                            |b, _| bench_impl(b, &lhs, &rhs, device),
                                        );
                                    } else {
                                        group.bench_function(func_name, |b| bench_impl(b, &lhs, &rhs, device),);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        let func_name = device.bench_name(format!(
            "matmul{}{}",
            if tpa { "_tranposedA" } else { "" },
            if tpb { "_tranposedB" } else { "" }
        ));
        if multiple_sizes {
            group.bench_with_input(BenchmarkId::new(func_name, size), &size, |b, _| bench_impl(b, &lhs, &rhs, device));
        } else {
            group.bench_function(func_name, |b| bench_impl(b, &lhs, &rhs, device));
        }
    }
}

#[allow(dead_code)]
fn test_functions(
    device: &Device,
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    fm: impl Fn(usize) -> usize,
) {
    let sizes = vec![2050usize, 2048, 1032, 1024, 528, 512, 128, 120, 32, 16].into_iter();
    for size in sizes {
        test_matmul(
            device,
            group,
            (1, fm(size), size, size),
            size,
            true,
            (false, false),
            GGLM_TYPE
        );
    }
    #[cfg(feature = "wgpu")]
    if let candle_core::Device::Wgpu(gpu) = &device {
        gpu.print_bindgroup_reuseinfo2();
    };
}

fn criterion_benchmark(c: &mut Criterion) {
    let handler = BenchDeviceHandler::new().unwrap();

    let mut group = c.benchmark_group("matmul_(2048x2048 * 2048x2048)");
    for device in handler.devices.iter() {
        test_matmul(
            device,
            &mut group,
            (1, 2048, 2048, 2048),
            1,
            false,
            (false, false),
            GGLM_TYPE
        );
        test_matmul(
            device,
            &mut group,
            (1, 2048, 2048, 2048),
            1,
            false,
            (true, false),
            GGLM_TYPE
        );
    }
    group.finish();

}

criterion_group!(benches, criterion_benchmark);
