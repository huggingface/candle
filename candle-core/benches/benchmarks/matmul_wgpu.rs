use std::time::{Duration, Instant};

use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle_core::{DType, Device, Tensor, D};

#[cfg(feature = "wgpu")]
use candle_core::wgpu::MatmulAlgorithm;

use criterion::{criterion_group, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

fn run(a: &Tensor, b: &Tensor) {
    a.matmul(b).unwrap();
}

fn test_matmul(
    device: &Device,
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    bmnk: (usize, usize, usize, usize),
    _is_small_line: bool,
    size: usize,
    multiple_sizes: bool,
    tp: (bool, bool),
) {
    let (b, m, n, k) = bmnk;
    let (tpa, tpb) = tp;
    let dtype = DType::F32;

    let lhs = if tpa {
        Tensor::zeros((b, k, m), dtype, device)
            .unwrap()
            .transpose(D::Minus1, D::Minus2)
            .unwrap()
    } else {
        Tensor::zeros((b, m, k), dtype, device).unwrap()
    };

    let rhs = if tpb {
        Tensor::zeros((b, n, k), dtype, device)
            .unwrap()
            .transpose(D::Minus1, D::Minus2)
            .unwrap()
    } else {
        Tensor::zeros((b, k, n), dtype, device).unwrap()
    };

    let flops = b * m * n * k;
    group.throughput(Throughput::Bytes(flops as u64));
    group.measurement_time(Duration::from_secs(1));
    group.sample_size(32);
    group.warm_up_time(Duration::from_secs_f32(0.25));
    if device.is_wgpu() {
        #[cfg(feature = "wgpu")]
        if let Device::Wgpu(wgpu) = device {
            {
                let mut algs;
                algs = vec![
                    //MatmulAlgorithm::Matmul64_64_8_8,
                    //MatmulAlgorithm::Matmul64_64_4_8,
                    //MatmulAlgorithm::Matmul64_64,
                    MatmulAlgorithm::Matmul32_64,
                    //MatmulAlgorithm::Matmul32_64B,
                    //MatmulAlgorithm::Matmul1_64B,
                    //MatmulAlgorithm::Matmul32_32,
                    //MatmulAlgorithm::Matmul16_16,
                    //MatmulAlgorithm::Matmul24_24,
                    //MatmulAlgorithm::Matmul24_48,
                    //MatmulAlgorithm::MatmulX,
                    MatmulAlgorithm::Matmul7,
                    MatmulAlgorithm::Matmul1,
                ];

                if _is_small_line {
                    algs.push(MatmulAlgorithm::Matmul1_64);
                    algs.push(MatmulAlgorithm::Matmul1_64B);
                }

                for alg in algs {
                    *(wgpu.matmul_alg.lock().unwrap()) = alg.clone();

                    let func_name = device.bench_name(format!(
                        "matmul_{:?}{}{}",
                        alg,
                        if tpa { "_tranposedA" } else { "" },
                        if tpb { "_tranposedB" } else { "" }
                    ));
                    tracing::info!("TEST: {func_name}");
                    if multiple_sizes {
                        group.bench_with_input(
                            BenchmarkId::new(func_name.clone(), size),
                            &size,
                            |b, _| {
                                b.iter_custom(|iters| {
                                    tracing::info!("TEST_CUSTOM_ITER: {func_name}");
                                    let start = Instant::now();
                                    for _ in 0..iters {
                                        run(black_box(&lhs), black_box(&rhs));
                                    }
                                    device.sync().unwrap();
                                    start.elapsed()
                                })
                            },
                        );
                    } else {
                        group.bench_function(func_name, |b| {
                            b.iter_custom(|iters| {
                                let start = Instant::now();
                                for _ in 0..iters {
                                    run(black_box(&lhs), black_box(&rhs));
                                }
                                device.sync().unwrap();
                                start.elapsed()
                            })
                        });
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
            group.bench_with_input(BenchmarkId::new(func_name, size), &size, |b, _| {
                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _i in 0..iters {
                        run(black_box(&lhs), black_box(&rhs));
                    }
                    device.sync().unwrap();
                    start.elapsed()
                })
            });
        } else {
            group.bench_function(func_name, |b| {
                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        run(black_box(&lhs), black_box(&rhs));
                    }
                    device.sync().unwrap();
                    start.elapsed()
                })
            });
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
            fm(2) == 1,
            size,
            true,
            (false, false),
        );
    }
    #[cfg(feature = "wgpu")]
    if let candle_core::Device::Wgpu(gpu) = &device {
        gpu.print_bindgroup_reuseinfo2();
    };
}

fn criterion_benchmark(c: &mut Criterion) {
    let handler = BenchDeviceHandler::new().unwrap();

    // let mut group = c.benchmark_group("matmul_m_1");
    // for device in handler.devices.iter() {
    //     test_functions(device, &mut group, |_| 1);
    // }
    // group.finish();

    // let mut group = c.benchmark_group("matmul_full");
    // for device in handler.devices.iter() {
    //     test_functions(device, &mut group, |size| size);
    // }
    // group.finish();

    let mut group = c.benchmark_group("matmul_(2048x2048 * 2048x2048)");
    for device in handler.devices.iter() {
        test_matmul(
            device,
            &mut group,
            (1, 2048, 2048, 2048),
            false,
            1,
            false,
            (false, false),
        );
        test_matmul(
            device,
            &mut group,
            (1, 2048, 2048, 2048),
            false,
            1,
            false,
            (true, false),
        );
        // test_matmul(device, &mut group, (1, 2048, 2048, 2048), false, 1, false, (false, true));
        // test_matmul(device, &mut group, (1, 2048, 2048, 2048), false, 1, false, (true, true));
    }
    group.finish();

    // let mut group = c.benchmark_group("matmul_(2050x2050 * 2050x2050)");
    // for device in handler.devices.iter() {
    //     test_matmul(device, &mut group, (1, 2050, 2050, 2050), false, 1, false, (false, false));
    //     // test_matmul(device, &mut group, (1, 2048, 2048, 2048), false, 1, false, (true, false));
    //     // test_matmul(device, &mut group, (1, 2048, 2048, 2048), false, 1, false, (false, true));
    //     // test_matmul(device, &mut group, (1, 2048, 2048, 2048), false, 1, false, (true, true));
    //  }
    // group.finish();

    // let mut group = c.benchmark_group("matmul_2*(1x9 * 9x576)");
    // for device in handler.devices.iter() {
    //     test_matmul(device, &mut group, (2, 1, 576, 9), true, 1, false, (false, false));
    // }
    // group.finish();

    // let mut group = c.benchmark_group("matmul_(32x2304 * 2304x5120)");
    // for device in handler.devices.iter() {
    //     test_matmul(device, &mut group, (1, 32, 5120, 2304), false, 1, false, (false, false));
    //     // test_matmul(device, &mut group, (1, 32, 5120, 2304), false, 1, false, (true, false));
    //     // test_matmul(device, &mut group, (1, 32, 5120, 2304), false, 1, false, (false, true));
    //     // test_matmul(device, &mut group, 81, 32, 5120, 2304), false, 1, false, (true, true));
    // }
    // group.finish();

    // let mut group = c.benchmark_group("matmul_2*(1x2048 * 2048x5632)");
    // for device in handler.devices.iter() {
    //     test_matmul(device, &mut group, (2, 1, 5632, 2048), true, 1, false, (false, false));
    //     test_matmul(device, &mut group, (2, 1, 5632, 2048), true, 1, false, (true, false));
    //     test_matmul(device, &mut group, (2, 1, 5632, 2048), true, 1, false, (false, true));
    //     test_matmul(device, &mut group, (2, 1, 5632, 2048), true, 1, false, (true, true));
    // }
    //group.finish();

    // let mut group = c.benchmark_group("matmul_1*(1x2048 * 2048x1)");
    // for device in handler.devices.iter() {
    //     test_matmul(device, &mut group, (1, 1, 1, 2048), true, 1, false, (false, false));
    // }
    // group.finish();

    // let mut group = c.benchmark_group("matmul_(64x2304 * 2304x5120)");
    // for device in handler.devices.iter() {
    //     test_matmul(device, &mut group, (1, 64, 5120, 2304), false, 1, false, (false, false));
    //     // test_matmul(device, &mut group, (1, 64, 5120, 2304), false, 1, false, (true, false));
    //     // test_matmul(device, &mut group, (1, 64, 5120, 2304), false, 1, false, (false, true));
    //     // test_matmul(device, &mut group, (1, 64, 5120, 2304), false, 1, false, (true, true));
    // }
    // group.finish();

    // let mut group = c.benchmark_group("matmul_(24x1536 * 1536x6144)");
    // for device in handler.devices.iter() {
    //     test_matmul(device, &mut group, (1, 24, 6144, 1536), false, 1, false, (false, false));
    // }
    // group.finish();

    // let mut group = c.benchmark_group("matmul_2*(653x1536 * 1536x1536)");
    // for device in handler.devices.iter() {
    //     test_matmul(device, &mut group, (2, 653, 1536, 1536), false, 1, false, (false, false));
    // }
    // group.finish();

    // let mut group = c.benchmark_group("matmul_32*(32x2304 * 2304x5120)");
    // for device in handler.devices.iter() {
    //     test_matmul(device, &mut group, (32, 32, 5120, 2304), false, 1, false, (false, false));
    // }
    // group.finish();

    // let mut group = c.benchmark_group("matmul_(1101x1280 * 1280x1280)");
    // for device in handler.devices.iter() {
    //     test_matmul(device, &mut group, (1, 1101, 1280, 1280), false, 1, false, (false, false));
    // }
    // group.finish();

    // let mut group = c.benchmark_group("matmul_10*(4096x64 * 64x4173)");
    // for device in handler.devices.iter() {
    //     test_matmul(device, &mut group, (10, 4096, 4173, 64), false, 1, false, (false, false));
    // }
    // group.finish();

    // let mut group = c.benchmark_group("matmul_20*(1024x64 * 64x1101)");
    // for device in handler.devices.iter() {
    //     test_matmul(device, &mut group,  (20, 1024, 1101, 64), false, 1, false, (false, false));
    // }
    // group.finish();

    // let mut group = c.benchmark_group("matmul_64*(64x1664 * 1664x2560)");
    // for device in handler.devices.iter() {
    //     test_matmul(device, &mut group, (64, 64, 2560, 1664), false, 1, false, (false, false)alse);
    // }
    // group.finish();
}

criterion_group!(benches, criterion_benchmark);
