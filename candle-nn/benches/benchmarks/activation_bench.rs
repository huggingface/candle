use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle::{DType, Device, Module, Tensor};
use candle_nn::Activation;
use criterion::{black_box, criterion_group, Criterion};
use std::time::Instant;

fn run_activation_benchmark(
    c: &mut Criterion,
    device: &Device,
    dtype: DType,
    activation: Activation,
    name: &str,
) {
    let sizes = [512, 1024, 2048, 4096, 8192];

    for &size in &sizes {
        let input = Tensor::randn(0f32, 1f32, (1, size), device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();

        let bench_name = format!("{}_{}_{}", device.bench_name(name), dtype.as_str(), size);

        c.bench_function(&bench_name, |b| {
            b.iter_custom(|iters| {
                device.sync().unwrap();
                let start = Instant::now();
                for _i in 0..iters {
                    let _result = black_box(activation.forward(black_box(&input)).unwrap());
                }
                device.sync().unwrap();
                start.elapsed()
            })
        });
    }
}

fn run_core_tensor_benchmark<F>(
    c: &mut Criterion,
    device: &Device,
    dtype: DType,
    name: &str,
    activation_fn: F,
) where
    F: Fn(&Tensor) -> candle::Result<Tensor> + Copy,
{
    let sizes = [512, 1024, 2048, 4096, 8192];

    for &size in &sizes {
        // For GLU variants, we need even dimensions (they split the input)
        let input_size = if name.contains("glu") || name.contains("GLU") {
            size * 2 // Double the size so after GLU we get 'size' output
        } else {
            size
        };

        let input = Tensor::randn(0f32, 1f32, (1, input_size), device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();

        let bench_name = format!(
            "{}_core_{}_{}",
            device.bench_name(name),
            dtype.as_str(),
            size
        );

        c.bench_function(&bench_name, |b| {
            b.iter_custom(|iters| {
                device.sync().unwrap();
                let start = Instant::now();
                for _i in 0..iters {
                    let _result = black_box(activation_fn(black_box(&input)).unwrap());
                }
                device.sync().unwrap();
                start.elapsed()
            })
        });
    }
}

fn run_comparison_benchmark(c: &mut Criterion, device: &Device, dtype: DType, name: &str) {
    let sizes = [1024, 2048, 4096];

    for &size in &sizes {
        let input = Tensor::randn(0f32, 1f32, (1, size * 2), device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();

        let bench_name = format!(
            "{}_comparison_{}_{}",
            device.bench_name(name),
            dtype.as_str(),
            size
        );

        // Create a benchmark group for direct comparison
        let mut group = c.benchmark_group(&bench_name);

        // Benchmark via Activation enum
        group.bench_function("enum", |b| {
            let activation = match name {
                "glu" => Activation::Glu,
                "geglu" => Activation::GeGlu,
                "reglu" => Activation::ReGlu,
                _ => Activation::Glu,
            };
            b.iter_custom(|iters| {
                device.sync().unwrap();
                let start = Instant::now();
                for _i in 0..iters {
                    let _result = black_box(activation.forward(black_box(&input)).unwrap());
                }
                device.sync().unwrap();
                start.elapsed()
            })
        });

        // Benchmark via core tensor method
        group.bench_function("core", |b| {
            b.iter_custom(|iters| {
                device.sync().unwrap();
                let start = Instant::now();
                for _i in 0..iters {
                    let _result = match name {
                        "glu" => black_box(black_box(&input).glu().unwrap()),
                        "geglu" => black_box(black_box(&input).geglu().unwrap()),
                        "reglu" => black_box(black_box(&input).reglu().unwrap()),
                        _ => black_box(black_box(&input).glu().unwrap()),
                    };
                }
                device.sync().unwrap();
                start.elapsed()
            })
        });

        group.finish();
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let handler = BenchDeviceHandler::new().unwrap();

    for device in handler.devices {
        // Benchmark GLU variants via Activation enum
        run_activation_benchmark(c, &device, DType::F32, Activation::Glu, "glu_enum_f32");
        run_activation_benchmark(c, &device, DType::F32, Activation::GeGlu, "geglu_enum_f32");
        run_activation_benchmark(c, &device, DType::F32, Activation::ReGlu, "reglu_enum_f32");

        // Benchmark GLU variants via core tensor methods
        run_core_tensor_benchmark(c, &device, DType::F32, "glu_core", |t| t.glu());
        run_core_tensor_benchmark(c, &device, DType::F32, "geglu_core", |t| t.geglu());
        run_core_tensor_benchmark(c, &device, DType::F32, "reglu_core", |t| t.reglu());

        // Direct comparison benchmarks
        run_comparison_benchmark(c, &device, DType::F32, "glu");
        run_comparison_benchmark(c, &device, DType::F32, "geglu");
        run_comparison_benchmark(c, &device, DType::F32, "reglu");

        // Compare with existing activations (for context)
        run_activation_benchmark(c, &device, DType::F32, Activation::Silu, "silu_f32");
        run_activation_benchmark(c, &device, DType::F32, Activation::Swiglu, "swiglu_f32");
        run_activation_benchmark(c, &device, DType::F32, Activation::Gelu, "gelu_f32");

        // Core tensor equivalents for comparison
        run_core_tensor_benchmark(c, &device, DType::F32, "silu_core", |t| t.silu());
        run_core_tensor_benchmark(c, &device, DType::F32, "gelu_core", |t| t.gelu());
        run_core_tensor_benchmark(c, &device, DType::F32, "relu_core", |t| t.relu());

        // Test different data types for GLU variants
        if !device.is_metal() {
            run_core_tensor_benchmark(c, &device, DType::F64, "glu_core", |t| t.glu());
            run_core_tensor_benchmark(c, &device, DType::F64, "geglu_core", |t| t.geglu());
            run_core_tensor_benchmark(c, &device, DType::F64, "reglu_core", |t| t.reglu());
        }

        run_core_tensor_benchmark(c, &device, DType::F16, "glu_core", |t| t.glu());
        run_core_tensor_benchmark(c, &device, DType::F16, "geglu_core", |t| t.geglu());
        run_core_tensor_benchmark(c, &device, DType::F16, "reglu_core", |t| t.reglu());

        run_core_tensor_benchmark(c, &device, DType::BF16, "glu_core", |t| t.glu());
        run_core_tensor_benchmark(c, &device, DType::BF16, "geglu_core", |t| t.geglu());
        run_core_tensor_benchmark(c, &device, DType::BF16, "reglu_core", |t| t.reglu());
    }
}

criterion_group!(benches, criterion_benchmark);
