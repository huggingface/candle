#![cfg(all(feature = "cuda", feature = "cuda-legacy-bf16"))]

use candle_core::cuda_backend::{
    cudarc::driver::{LaunchConfig, PushKernelArg},
    kernels, CudaDevice, WrapErr,
};
use candle_core::{DType, Device, Result, Tensor};
use half::bf16;

fn bf16(v: f32) -> bf16 {
    bf16::from_f32(v)
}

fn as_f32_vec(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

fn assert_close(actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
    assert_eq!(actual.len(), expected.len());

    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "index {i}: expected NaN, got {a}");
            continue;
        }

        let tol = atol + rtol * e.abs();
        let diff = (a - e).abs();

        assert!(
            diff <= tol,
            "index {i}: actual={a} expected={e} diff={diff} tol={tol}"
        );
    }
}

fn assert_tensor_close(actual: &Tensor, expected: &Tensor, atol: f32, rtol: f32) -> Result<()> {
    let actual = as_f32_vec(actual)?;
    let expected = as_f32_vec(expected)?;
    assert_close(&actual, &expected, atol, rtol);
    Ok(())
}

#[test]
fn cuda_legacy_bf16_elementwise_runtime() -> Result<()> {
    let cuda = Device::new_cuda(0)?;
    let cpu = Device::Cpu;

    let a_f32 = [-3.0f32, -1.5, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
    let b_f32 = [0.5f32, 2.0, -1.0, 0.25, 1.5, -2.0, 0.75, 4.0];

    let cast = Tensor::from_slice(&a_f32, a_f32.len(), &cuda)?
        .to_dtype(DType::BF16)?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;

    let cast_expected: Vec<f32> = a_f32.iter().map(|&v| bf16(v).to_f32()).collect();

    assert_close(&cast, &cast_expected, 0.0, 0.0);

    let a_bf16: Vec<bf16> = a_f32.iter().map(|&v| bf16(v)).collect();
    let b_bf16: Vec<bf16> = b_f32.iter().map(|&v| bf16(v)).collect();

    let a_cuda = Tensor::from_slice(&a_bf16, a_bf16.len(), &cuda)?;
    let b_cuda = Tensor::from_slice(&b_bf16, b_bf16.len(), &cuda)?;

    let a_cpu = Tensor::from_slice(&a_bf16, a_bf16.len(), &cpu)?;
    let b_cpu = Tensor::from_slice(&b_bf16, b_bf16.len(), &cpu)?;

    assert_tensor_close(&(&a_cuda + &b_cuda)?, &(&a_cpu + &b_cpu)?, 0.0, 0.0)?;

    assert_tensor_close(&(&a_cuda * &b_cuda)?, &(&a_cpu * &b_cpu)?, 0.0, 0.0)?;

    assert_tensor_close(&a_cuda.maximum(&b_cuda)?, &a_cpu.maximum(&b_cpu)?, 0.0, 0.0)?;

    assert_tensor_close(&a_cuda.minimum(&b_cuda)?, &a_cpu.minimum(&b_cpu)?, 0.0, 0.0)?;

    assert_tensor_close(
        &a_cuda.affine(1.5, -0.25)?,
        &a_cpu.affine(1.5, -0.25)?,
        0.0,
        0.0,
    )?;

    assert_tensor_close(&a_cuda.exp()?, &a_cpu.exp()?, 0.02, 0.02)?;

    assert_tensor_close(&a_cuda.tanh()?, &a_cpu.tanh()?, 0.02, 0.02)?;

    assert_tensor_close(&a_cuda.gelu_erf()?, &a_cpu.gelu_erf()?, 0.02, 0.02)?;

    let nan_a = [bf16(f32::NAN), bf16(1.0), bf16(-2.0), bf16(0.0)];
    let nan_b = [bf16(0.0), bf16(f32::NAN), bf16(-3.0), bf16(1.0)];

    let nan_a = Tensor::from_slice(&nan_a, nan_a.len(), &cuda)?;
    let nan_b = Tensor::from_slice(&nan_b, nan_b.len(), &cuda)?;

    let max = as_f32_vec(&nan_a.maximum(&nan_b)?)?;
    let min = as_f32_vec(&nan_a.minimum(&nan_b)?)?;

    assert!(max[0].is_nan());
    assert!(max[1].is_nan());
    assert!(min[0].is_nan());
    assert!(min[1].is_nan());

    Ok(())
}

#[test]
fn cuda_legacy_bf16_fast_reduce_runtime() -> Result<()> {
    let cuda = Device::new_cuda(0)?;
    let cpu = Device::Cpu;

    let rows = 4usize;
    let cols = 256usize;

    let values = vec![bf16(0.25); rows * cols];

    let cuda_x = Tensor::from_slice(&values, (rows, cols), &cuda)?;
    let cpu_x = Tensor::from_slice(&values, (rows, cols), &cpu)?;

    let cuda_sum = cuda_x.sum(1)?;
    let cpu_sum = cpu_x.sum(1)?;

    assert_tensor_close(&cuda_sum, &cpu_sum, 0.0, 0.0)?;

    let sums = as_f32_vec(&cuda_sum)?;
    assert_eq!(sums, vec![64.0; rows]);

    assert_tensor_close(&cuda_x.max(1)?, &cpu_x.max(1)?, 0.0, 0.0)?;

    assert_tensor_close(&cuda_x.min(1)?, &cpu_x.min(1)?, 0.0, 0.0)?;

    Ok(())
}

#[test]
fn cuda_legacy_bf16_atomic_sum_runtime() -> Result<()> {
    let dev = CudaDevice::new_with_stream(0)?;

    let rows = 256usize;
    let cols = 32usize;
    let numel = rows * cols;

    let input_host = vec![bf16(0.25); numel];
    let input = dev.clone_htod(&input_host)?;

    let info_host = vec![rows, cols, cols, 1, rows, cols];
    let info = dev.clone_htod(&info_host)?;

    let func = dev.get_or_load_func("sum_bf16", &kernels::REDUCE)?;

    let num_dims = 2usize;
    let num_sum_dims = 1usize;

    for iteration in 0..16 {
        let out = dev.alloc_zeros::<bf16>(cols)?;

        let mut builder = func.builder();
        builder.arg(&numel);
        builder.arg(&num_dims);
        builder.arg(&num_sum_dims);
        builder.arg(&info);
        builder.arg(&input);
        builder.arg(&out);

        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(numel as u32))
                .w()?;
        }

        let out = dev.clone_dtoh(&out)?;

        for (i, value) in out.iter().enumerate() {
            assert_eq!(value.to_f32(), 64.0, "iteration {iteration}, column {i}");
        }
    }

    Ok(())
}

#[test]
fn cuda_legacy_bf16_matmul_runtime() -> Result<()> {
    let cuda = Device::new_cuda(0)?;

    let a = [
        bf16(1.0),
        bf16(2.0),
        bf16(3.0),
        bf16(4.0),
        bf16(5.0),
        bf16(6.0),
    ];

    let b = [
        bf16(1.0),
        bf16(2.0),
        bf16(3.0),
        bf16(4.0),
        bf16(5.0),
        bf16(6.0),
    ];

    let a_cuda = Tensor::from_slice(&a, (2, 3), &cuda)?;
    let b_cuda = Tensor::from_slice(&b, (3, 2), &cuda)?;

    println!("CUDA BF16 GEMM: before matmul");

    let cuda_out = a_cuda.matmul(&b_cuda)?;

    println!("CUDA BF16 GEMM: matmul returned");

    let out = as_f32_vec(&cuda_out)?;

    println!("CUDA BF16 GEMM output: {out:?}");

    assert_close(&out, &[22.0, 28.0, 49.0, 64.0], 0.0, 0.0);

    Ok(())
}
