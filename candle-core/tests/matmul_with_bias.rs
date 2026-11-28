use candle_core::{DType, Device, Tensor};

fn sample(rows: usize, cols: usize, dev: &Device, dtype: DType) -> Tensor {
    Tensor::randn(0f32, 1f32, (rows, cols), dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    (a.to_dtype(DType::F32).unwrap() - b.to_dtype(DType::F32).unwrap())
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

#[test]
fn cpu_matmul_with_bias_matches_two_ops() {
    let dev = Device::Cpu;
    let (m, k, n) = (5usize, 7usize, 3usize);
    let a = sample(m, k, &dev, DType::F32);
    let b = sample(k, n, &dev, DType::F32);
    let bias = sample(1, n, &dev, DType::F32).reshape(n).unwrap();
    let got = a.matmul_with_bias(&b, &bias).unwrap();
    let want = a.matmul(&b).unwrap().broadcast_add(&bias).unwrap();
    assert!(max_abs_diff(&got, &want) < 1e-6);
}

#[cfg(feature = "metal")]
#[test]
fn metal_matmul_with_bias_matches_two_ops() {
    let dev = Device::new_metal(0).unwrap();
    for (m, k, n) in [(64usize, 64usize, 64usize), (67, 17, 33), (1, 1024, 1024)] {
        for dtype in [DType::F32, DType::F16] {
            let a = sample(m, k, &dev, dtype);
            let b = sample(k, n, &dev, dtype);
            let bias = sample(1, n, &dev, dtype).reshape(n).unwrap();
            let got = a.matmul_with_bias(&b, &bias).unwrap();
            let want = a.matmul(&b).unwrap().broadcast_add(&bias).unwrap();
            let diff = max_abs_diff(&got, &want);
            let tolerance = if dtype == DType::F16 { 0.1 } else { 1e-4 };
            assert!(diff < tolerance, "{m}x{k}x{n} {dtype:?}: diff {diff}");
        }
    }
}
