use candle::{DType, Device, Tensor};
use candle_nn::{Linear, Module};

fn rand(rows: usize, cols: usize, dev: &Device, dtype: DType) -> Tensor {
    Tensor::randn(0f32, 1f32, (rows, cols), dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
}

fn check(dev: &Device, dtype: DType, shape: (usize, usize)) {
    let (m, k) = shape;
    let x = rand(m, k, dev, dtype);
    let w = rand(4, k, dev, dtype);
    let b = rand(1, 4, dev, dtype).reshape(4).unwrap();
    let layer = Linear::new(w, Some(b));
    let got = layer.forward(&x).unwrap();
    let want = x
        .matmul(&layer.weight().t().unwrap())
        .unwrap()
        .broadcast_add(layer.bias().unwrap())
        .unwrap();
    let diff = (got.to_dtype(DType::F32).unwrap() - want.to_dtype(DType::F32).unwrap())
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let tol = if dtype == DType::F16 { 0.1 } else { 1e-5 };
    assert!(diff < tol, "{shape:?} {dtype:?}: diff {diff}");
}

#[test]
fn linear_bias_2d() {
    check(&Device::Cpu, DType::F32, (5, 7));
}

#[cfg(feature = "metal")]
#[test]
fn linear_bias_metal_3d() {
    let dev = Device::new_metal(0).unwrap();
    for dtype in [DType::F32, DType::F16] {
        check(&dev, dtype, (2, 3));
        check(&dev, dtype, (1, 5));
    }
    // 3D input takes the same fused 2D reshape path.
    let x = rand(2 * 3, 5, &dev, DType::F32).reshape((2, 3, 5)).unwrap();
    let w = rand(4, 5, &dev, DType::F32);
    let b = rand(1, 4, &dev, DType::F32).reshape(4).unwrap();
    let layer = Linear::new(w, Some(b));
    let got = layer.forward(&x).unwrap();
    let want = x
        .reshape((6, 5))
        .unwrap()
        .matmul(&layer.weight().t().unwrap())
        .unwrap()
        .broadcast_add(layer.bias().unwrap())
        .unwrap()
        .reshape((2, 3, 4))
        .unwrap();
    let diff = (got - want).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
    assert!(diff < 1e-5, "3d diff {diff}");
}
