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

fn bf16_elementwise_case() -> Result<()> {
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

fn bf16_fast_reduce_case() -> Result<()> {
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

fn bf16_atomic_sum_case() -> Result<()> {
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

fn bf16_matmul_case() -> Result<()> {
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

fn bf16_memory_ops_case() -> Result<()> {
    let dev = CudaDevice::new_with_stream(0)?;

    {
        let numel = 8usize;
        let out = dev.alloc_zeros::<bf16>(numel)?;
        let value = bf16(1.25);

        let func = dev.get_or_load_func("fill_bf16", &kernels::FILL)?;
        let mut builder = func.builder();

        builder.arg(&out);
        builder.arg(&value);
        builder.arg(&numel);

        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(numel as u32))
                .w()?;
        }

        let out = dev.clone_dtoh(&out)?;
        for (i, value) in out.iter().enumerate() {
            assert_eq!(value.to_f32(), 1.25, "fill_bf16 mismatch at index {i}");
        }
    }

    {
        let src_host = [
            bf16(1.0),
            bf16(2.0),
            bf16(3.0),
            bf16(4.0),
            bf16(5.0),
            bf16(6.0),
        ];

        let src = dev.clone_htod(&src_host)?;
        let dst = dev.alloc_zeros::<bf16>(8)?;

        let d1 = 2u32;
        let d2 = 3u32;
        let src_s = 3u32;
        let dst_s = 4u32;

        let func = dev.get_or_load_func("copy2d_bf16", &kernels::FILL)?;
        let mut builder = func.builder();

        builder.arg(&src);
        builder.arg(&dst);
        builder.arg(&d1);
        builder.arg(&d2);
        builder.arg(&src_s);
        builder.arg(&dst_s);

        unsafe {
            builder.launch(LaunchConfig::for_num_elems(d1 * d2)).w()?;
        }

        let dst = dev.clone_dtoh(&dst)?;
        let dst: Vec<f32> = dst.iter().map(|v| v.to_f32()).collect();

        assert_eq!(dst, vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0]);
    }

    {
        let numel = 4usize;
        let num_dims = 1usize;
        let info = dev.clone_htod(&[4usize, 1usize])?;
        let out = dev.alloc_zeros::<bf16>(numel)?;
        let value = bf16(-0.5);

        let func = dev.get_or_load_func("const_set_bf16", &kernels::FILL)?;
        let mut builder = func.builder();

        builder.arg(&numel);
        builder.arg(&num_dims);
        builder.arg(&info);
        builder.arg(&value);
        builder.arg(&out);

        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(numel as u32))
                .w()?;
        }

        let out = dev.clone_dtoh(&out)?;
        for (i, value) in out.iter().enumerate() {
            assert_eq!(value.to_f32(), -0.5, "const_set_bf16 mismatch at index {i}");
        }
    }

    Ok(())
}

fn bf16_softmax_case() -> Result<()> {
    let dev = CudaDevice::new_with_stream(0)?;

    let rows = 2usize;
    let cols = 4usize;

    let src_f32 = [1.0f32, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0];

    let src_host: Vec<bf16> = src_f32.iter().copied().map(bf16).collect();

    let src = dev.clone_htod(&src_host)?;
    let dst = dev.alloc_zeros::<bf16>(rows * cols)?;

    let n_cols = cols as i32;

    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (1, 32, 1),
        shared_mem_bytes: 0,
    };

    let func = dev.get_or_load_func("softmax_bf16", &kernels::REDUCE)?;
    let mut builder = func.builder();

    builder.arg(&src);
    builder.arg(&dst);
    builder.arg(&n_cols);

    unsafe {
        builder.launch(cfg).w()?;
    }

    let got = dev.clone_dtoh(&dst)?;
    let got: Vec<f32> = got.iter().map(|v| v.to_f32()).collect();

    for row in 0..rows {
        let values = &src_f32[row * cols..(row + 1) * cols];

        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let exp: Vec<f32> = values.iter().map(|v| (*v - max).exp()).collect();

        let sum: f32 = exp.iter().sum();

        for col in 0..cols {
            let expected = exp[col] / sum;
            let actual = got[row * cols + col];

            assert!(
                (actual - expected).abs() <= 0.01,
                "softmax row={row} col={col}: actual={actual} expected={expected}"
            );
        }
    }

    Ok(())
}

fn bf16_norm_case() -> Result<()> {
    let dev = CudaDevice::new_with_stream(0)?;

    let rows = 2usize;
    let cols = 4usize;
    let eps = 1e-5f32;
    let block_size = 32i32;
    let n_cols = cols as i32;

    let src_f32 = [1.0f32, 2.0, 3.0, 4.0, -2.0, -1.0, 1.0, 2.0];

    let alpha_f32 = [1.0f32, 0.5, 1.5, 2.0];
    let beta_f32 = [0.1f32, -0.2, 0.3, -0.4];

    let src_host: Vec<bf16> = src_f32.iter().copied().map(bf16).collect();

    let alpha_host: Vec<bf16> = alpha_f32.iter().copied().map(bf16).collect();

    let beta_host: Vec<bf16> = beta_f32.iter().copied().map(bf16).collect();

    let src = dev.clone_htod(&src_host)?;
    let alpha = dev.clone_htod(&alpha_host)?;
    let beta = dev.clone_htod(&beta_host)?;

    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    {
        let dst = dev.alloc_zeros::<bf16>(rows * cols)?;

        let func = dev.get_or_load_func("rmsnorm_bf16", &kernels::REDUCE)?;

        let mut builder = func.builder();

        builder.arg(&src);
        builder.arg(&dst);
        builder.arg(&alpha);
        builder.arg(&n_cols);
        builder.arg(&block_size);
        builder.arg(&eps);

        unsafe {
            builder.launch(cfg).w()?;
        }

        let got = dev.clone_dtoh(&dst)?;
        let got: Vec<f32> = got.iter().map(|v| v.to_f32()).collect();

        for row in 0..rows {
            let values = &src_f32[row * cols..(row + 1) * cols];

            let mean_sq = values.iter().map(|v| v * v).sum::<f32>() / cols as f32;

            let denom = (mean_sq + eps).sqrt();

            for col in 0..cols {
                let expected = values[col] / denom * alpha_f32[col];

                let actual = got[row * cols + col];

                assert!(
                    (actual - expected).abs() <= 0.03,
                    "rmsnorm row={row} col={col}: actual={actual} expected={expected}"
                );
            }
        }
    }

    {
        let dst = dev.alloc_zeros::<bf16>(rows * cols)?;

        let func = dev.get_or_load_func("layernorm_bf16", &kernels::REDUCE)?;

        let mut builder = func.builder();

        builder.arg(&src);
        builder.arg(&dst);
        builder.arg(&alpha);
        builder.arg(&beta);
        builder.arg(&n_cols);
        builder.arg(&block_size);
        builder.arg(&eps);

        unsafe {
            builder.launch(cfg).w()?;
        }

        let got = dev.clone_dtoh(&dst)?;
        let got: Vec<f32> = got.iter().map(|v| v.to_f32()).collect();

        for row in 0..rows {
            let values = &src_f32[row * cols..(row + 1) * cols];

            let mean = values.iter().sum::<f32>() / cols as f32;

            let mean_sq = values.iter().map(|v| v * v).sum::<f32>() / cols as f32;

            let var = mean_sq - mean * mean;
            let inv_std = (var + eps).sqrt().recip();

            for col in 0..cols {
                let expected = (values[col] - mean) * inv_std * alpha_f32[col] + beta_f32[col];

                let actual = got[row * cols + col];

                assert!(
                    (actual - expected).abs() <= 0.03,
                    "layernorm row={row} col={col}: actual={actual} expected={expected}"
                );
            }
        }
    }

    Ok(())
}

fn run_unary_bf16_kernel(
    dev: &CudaDevice,
    kernel_name: &str,
    input_f32: &[f32],
) -> Result<Vec<f32>> {
    let input_host: Vec<bf16> = input_f32.iter().copied().map(bf16).collect();
    let input = dev.clone_htod(&input_host)?;
    let output = dev.alloc_zeros::<bf16>(input_f32.len())?;

    let numel = input_f32.len();
    let num_dims = 1usize;
    let info = dev.clone_htod(&[numel, 1usize])?;

    let func = dev.get_or_load_func(kernel_name, &kernels::UNARY)?;
    let mut builder = func.builder();

    builder.arg(&numel);
    builder.arg(&num_dims);
    builder.arg(&info);
    builder.arg(&input);
    builder.arg(&output);

    unsafe {
        builder
            .launch(LaunchConfig::for_num_elems(numel as u32))
            .w()?;
    }

    let output = dev.clone_dtoh(&output)?;
    Ok(output.iter().map(|v| v.to_f32()).collect())
}

fn run_unary1_bf16_kernel(
    dev: &CudaDevice,
    kernel_name: &str,
    input_f32: &[f32],
    param_f32: f32,
) -> Result<Vec<f32>> {
    let input_host: Vec<bf16> = input_f32.iter().copied().map(bf16).collect();
    let input = dev.clone_htod(&input_host)?;
    let output = dev.alloc_zeros::<bf16>(input_f32.len())?;

    let numel = input_f32.len();
    let num_dims = 1usize;
    let info = dev.clone_htod(&[numel, 1usize])?;
    let param = bf16(param_f32);

    let func = dev.get_or_load_func(kernel_name, &kernels::UNARY)?;
    let mut builder = func.builder();

    builder.arg(&numel);
    builder.arg(&num_dims);
    builder.arg(&info);
    builder.arg(&param);
    builder.arg(&input);
    builder.arg(&output);

    unsafe {
        builder
            .launch(LaunchConfig::for_num_elems(numel as u32))
            .w()?;
    }

    let output = dev.clone_dtoh(&output)?;
    Ok(output.iter().map(|v| v.to_f32()).collect())
}

fn bf16_unary_full_case() -> Result<()> {
    let dev = CudaDevice::new_with_stream(0)?;

    struct Case {
        kernel: &'static str,
        input: &'static [f32],
        reference: fn(f32) -> f32,
        atol: f32,
        rtol: f32,
    }

    fn recip(x: f32) -> f32 {
        1.0 / x
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn silu(x: f32) -> f32 {
        x * sigmoid(x)
    }

    fn normcdf(x: f32) -> f32 {
        0.5 * libm::erfcf(-x * std::f32::consts::FRAC_1_SQRT_2)
    }

    let positive = &[0.25f32, 0.5, 1.0, 2.0, 4.0];
    let signed = &[-2.0f32, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0];
    let rounding = &[-2.75f32, -1.5, -0.25, 0.25, 1.5, 2.75];

    let cases = [
        Case {
            kernel: "urecip_bf16",
            input: positive,
            reference: recip,
            atol: 0.02,
            rtol: 0.02,
        },
        Case {
            kernel: "ulog_bf16",
            input: positive,
            reference: f32::ln,
            atol: 0.02,
            rtol: 0.02,
        },
        Case {
            kernel: "usin_bf16",
            input: signed,
            reference: f32::sin,
            atol: 0.02,
            rtol: 0.02,
        },
        Case {
            kernel: "ucos_bf16",
            input: signed,
            reference: f32::cos,
            atol: 0.02,
            rtol: 0.02,
        },
        Case {
            kernel: "uerf_bf16",
            input: signed,
            reference: libm::erff,
            atol: 0.02,
            rtol: 0.02,
        },
        Case {
            kernel: "uceil_bf16",
            input: rounding,
            reference: f32::ceil,
            atol: 0.0,
            rtol: 0.0,
        },
        Case {
            kernel: "ufloor_bf16",
            input: rounding,
            reference: f32::floor,
            atol: 0.0,
            rtol: 0.0,
        },
        Case {
            kernel: "uround_bf16",
            input: rounding,
            reference: f32::round,
            atol: 0.0,
            rtol: 0.0,
        },
        Case {
            kernel: "unormcdf_bf16",
            input: signed,
            reference: normcdf,
            atol: 0.02,
            rtol: 0.02,
        },
        Case {
            kernel: "uabs_bf16",
            input: signed,
            reference: f32::abs,
            atol: 0.0,
            rtol: 0.0,
        },
        Case {
            kernel: "usqrt_bf16",
            input: positive,
            reference: f32::sqrt,
            atol: 0.02,
            rtol: 0.02,
        },
        Case {
            kernel: "usilu_bf16",
            input: signed,
            reference: silu,
            atol: 0.03,
            rtol: 0.03,
        },
        Case {
            kernel: "usigmoid_bf16",
            input: signed,
            reference: sigmoid,
            atol: 0.02,
            rtol: 0.02,
        },
    ];

    for case in cases {
        println!("BF16 unary gate: {}", case.kernel);

        let got = run_unary_bf16_kernel(&dev, case.kernel, case.input)?;

        for (i, (&actual, &input)) in got.iter().zip(case.input.iter()).enumerate() {
            let expected_bf16 = bf16((case.reference)(input)).to_f32();
            let tol = case.atol + case.rtol * expected_bf16.abs();

            assert!(
                (actual - expected_bf16).abs() <= tol,
                "{} index={i} input={input}: actual={actual} expected={expected_bf16} tol={tol}",
                case.kernel
            );
        }
    }

    let input = [0.25f32, 0.5, 1.0, 2.0, 4.0];

    let got = run_unary1_bf16_kernel(&dev, "upowf_bf16", &input, 1.5)?;

    for (i, (&actual, &x)) in got.iter().zip(input.iter()).enumerate() {
        let expected = bf16(x.powf(1.5)).to_f32();

        assert!(
            (actual - expected).abs() <= 0.03 + 0.03 * expected.abs(),
            "upowf_bf16 index={i}: actual={actual} expected={expected}"
        );
    }

    Ok(())
}

fn bf16_rope_case() -> Result<()> {
    let dev = CudaDevice::new_with_stream(0)?;

    let src_host: Vec<bf16> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        .into_iter()
        .map(bf16)
        .collect();

    let cos_host = vec![bf16(0.0); 4];
    let sin_host = vec![bf16(1.0); 4];

    let src = dev.clone_htod(&src_host)?;
    let cos = dev.clone_htod(&cos_host)?;
    let sin = dev.clone_htod(&sin_host)?;

    let cfg = LaunchConfig::for_num_elems(4);

    {
        let dst = dev.alloc_zeros::<bf16>(8)?;

        let bh = 1u32;
        let td = 8u32;
        let stride_b = 0u32;

        let func = dev.get_or_load_func("rope_i_bf16", &kernels::REDUCE)?;

        let mut builder = func.builder();
        builder.arg(&src);
        builder.arg(&cos);
        builder.arg(&sin);
        builder.arg(&dst);
        builder.arg(&bh);
        builder.arg(&td);
        builder.arg(&stride_b);

        unsafe {
            builder.launch(cfg).w()?;
        }

        let got: Vec<f32> = dev.clone_dtoh(&dst)?.iter().map(|v| v.to_f32()).collect();

        assert_eq!(got, vec![-2.0, 1.0, -4.0, 3.0, -6.0, 5.0, -8.0, 7.0]);
    }

    {
        let dst = dev.alloc_zeros::<bf16>(8)?;

        let bh = 1u32;
        let td = 8u32;
        let d = 4u32;
        let stride_b = 0u32;

        let func = dev.get_or_load_func("rope_bf16", &kernels::REDUCE)?;

        let mut builder = func.builder();
        builder.arg(&src);
        builder.arg(&cos);
        builder.arg(&sin);
        builder.arg(&dst);
        builder.arg(&bh);
        builder.arg(&td);
        builder.arg(&d);
        builder.arg(&stride_b);

        unsafe {
            builder.launch(cfg).w()?;
        }

        let got: Vec<f32> = dev.clone_dtoh(&dst)?.iter().map(|v| v.to_f32()).collect();

        assert_eq!(got, vec![-3.0, -4.0, 1.0, 2.0, -7.0, -8.0, 5.0, 6.0]);
    }

    {
        let dst = dev.alloc_zeros::<bf16>(8)?;

        let b = 1u32;
        let t = 2u32;
        let h = 1u32;
        let d = 4u32;
        let stride_b = 0u32;

        let func = dev.get_or_load_func("rope_thd_bf16", &kernels::REDUCE)?;

        let mut builder = func.builder();
        builder.arg(&src);
        builder.arg(&cos);
        builder.arg(&sin);
        builder.arg(&dst);
        builder.arg(&b);
        builder.arg(&t);
        builder.arg(&h);
        builder.arg(&d);
        builder.arg(&stride_b);

        unsafe {
            builder.launch(cfg).w()?;
        }

        let got: Vec<f32> = dev.clone_dtoh(&dst)?.iter().map(|v| v.to_f32()).collect();

        assert_eq!(got, vec![-3.0, -4.0, 1.0, 2.0, -7.0, -8.0, 5.0, 6.0]);
    }

    Ok(())
}

fn bf16_argmin_argmax_case() -> Result<()> {
    let cuda = Device::new_cuda(0)?;

    let values = [
        bf16(3.0),
        bf16(1.0),
        bf16(4.0),
        bf16(2.0),
        bf16(-2.0),
        bf16(8.0),
        bf16(0.5),
        bf16(7.0),
    ];

    let x = Tensor::from_slice(&values, (2, 4), &cuda)?;

    let argmin = x.argmin(1)?.to_vec1::<u32>()?;
    let argmax = x.argmax(1)?.to_vec1::<u32>()?;

    assert_eq!(argmin, vec![1, 0]);
    assert_eq!(argmax, vec![2, 1]);

    Ok(())
}

fn bf16_sort_case() -> Result<()> {
    let cuda = Device::new_cuda(0)?;

    let values = [
        bf16(3.0),
        bf16(1.0),
        bf16(4.0),
        bf16(2.0),
        bf16(8.0),
        bf16(5.0),
        bf16(7.0),
        bf16(6.0),
    ];

    let x = Tensor::from_slice(&values, (2, 4), &cuda)?;

    let asc_idx = x.arg_sort_last_dim(true)?.to_vec2::<u32>()?;
    assert_eq!(asc_idx, vec![vec![1, 3, 0, 2], vec![1, 3, 2, 0],]);

    let (asc, asc_idx2) = x.sort_last_dim(true)?;

    assert_eq!(asc_idx2.to_vec2::<u32>()?, asc_idx);

    assert_eq!(
        as_f32_vec(&asc)?,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    );

    let (desc, desc_idx) = x.sort_last_dim(false)?;

    assert_eq!(
        desc_idx.to_vec2::<u32>()?,
        vec![vec![2, 0, 3, 1], vec![0, 2, 3, 1],]
    );

    assert_eq!(
        as_f32_vec(&desc)?,
        vec![4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0]
    );

    Ok(())
}

fn bf16_where_case() -> Result<()> {
    let cuda = Device::new_cuda(0)?;

    let cond = Tensor::from_slice(&[1u8, 0, 1, 0, 0, 1, 0, 1], (2, 4), &cuda)?;

    let t = Tensor::from_slice(
        &[
            bf16(1.0),
            bf16(2.0),
            bf16(3.0),
            bf16(4.0),
            bf16(5.0),
            bf16(6.0),
            bf16(7.0),
            bf16(8.0),
        ],
        (2, 4),
        &cuda,
    )?;

    let f = Tensor::from_slice(
        &[
            bf16(-1.0),
            bf16(-2.0),
            bf16(-3.0),
            bf16(-4.0),
            bf16(-5.0),
            bf16(-6.0),
            bf16(-7.0),
            bf16(-8.0),
        ],
        (2, 4),
        &cuda,
    )?;

    let out = cond.where_cond(&t, &f)?;

    assert_eq!(
        as_f32_vec(&out)?,
        vec![1.0, -2.0, 3.0, -4.0, -5.0, 6.0, -7.0, 8.0,]
    );

    Ok(())
}

fn bf16_index_select_gather_case() -> Result<()> {
    let cuda = Device::new_cuda(0)?;

    {
        let x = Tensor::from_slice(
            &[
                bf16(1.0),
                bf16(2.0),
                bf16(3.0),
                bf16(4.0),
                bf16(5.0),
                bf16(6.0),
                bf16(7.0),
                bf16(8.0),
            ],
            (4, 2),
            &cuda,
        )?;

        let ids = Tensor::from_slice(&[2u32, 0, 3], 3, &cuda)?;

        let out = x.index_select(&ids, 0)?;

        assert_eq!(as_f32_vec(&out)?, vec![5.0, 6.0, 1.0, 2.0, 7.0, 8.0]);
    }

    {
        let x = Tensor::from_slice(
            &[
                bf16(1.0),
                bf16(2.0),
                bf16(3.0),
                bf16(4.0),
                bf16(5.0),
                bf16(6.0),
                bf16(7.0),
                bf16(8.0),
            ],
            (2, 4),
            &cuda,
        )?;

        let ids = Tensor::from_slice(&[3u32, 1, 0, 2], (2, 2), &cuda)?;

        let out = x.gather(&ids, 1)?;

        assert_eq!(as_f32_vec(&out)?, vec![4.0, 2.0, 5.0, 7.0]);
    }

    Ok(())
}

fn bf16_index_add_case() -> Result<()> {
    let cuda = Device::new_cuda(0)?;

    let dst = Tensor::zeros((4, 2), DType::BF16, &cuda)?;

    let ids = Tensor::from_slice(&[2u32, 0, 2], 3, &cuda)?;

    let src = Tensor::from_slice(
        &[
            bf16(1.0),
            bf16(2.0),
            bf16(3.0),
            bf16(4.0),
            bf16(5.0),
            bf16(6.0),
        ],
        (3, 2),
        &cuda,
    )?;

    let out = dst.index_add(&ids, &src, 0)?;

    assert_eq!(
        as_f32_vec(&out)?,
        vec![3.0, 4.0, 0.0, 0.0, 6.0, 8.0, 0.0, 0.0,]
    );

    Ok(())
}

fn bf16_scatter_case() -> Result<()> {
    let cuda = Device::new_cuda(0)?;

    {
        let dst = Tensor::zeros((2, 4), DType::BF16, &cuda)?;

        let ids = Tensor::from_slice(&[1u32, 3, 0, 2], (2, 2), &cuda)?;

        let src = Tensor::from_slice(
            &[bf16(10.0), bf16(20.0), bf16(30.0), bf16(40.0)],
            (2, 2),
            &cuda,
        )?;

        let out = dst.scatter(&ids, &src, 1)?;

        assert_eq!(
            as_f32_vec(&out)?,
            vec![0.0, 10.0, 0.0, 20.0, 30.0, 0.0, 40.0, 0.0,]
        );
    }

    {
        let dst = Tensor::zeros((2, 4), DType::BF16, &cuda)?;

        let ids = Tensor::from_slice(&[1u32, 1, 2, 2], (2, 2), &cuda)?;

        let src = Tensor::from_slice(&[bf16(1.0), bf16(2.0), bf16(3.0), bf16(4.0)], (2, 2), &cuda)?;

        let out = dst.scatter_add(&ids, &src, 1)?;

        assert_eq!(
            as_f32_vec(&out)?,
            vec![0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0,]
        );
    }

    Ok(())
}

fn bf16_conv_case() -> Result<()> {
    let cuda = Device::new_cuda(0)?;
    let cpu = Device::Cpu;

    {
        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0];

        let kernel = [0.5f32, -1.0, 2.0];

        let x_cuda = Tensor::from_slice(&input, (1, 1, 5), &cuda)?.to_dtype(DType::BF16)?;

        let w_cuda = Tensor::from_slice(&kernel, (1, 1, 3), &cuda)?.to_dtype(DType::BF16)?;

        let x_cpu = Tensor::from_slice(&input, (1, 1, 5), &cpu)?;

        let w_cpu = Tensor::from_slice(&kernel, (1, 1, 3), &cpu)?;

        let got = x_cuda.conv1d(&w_cuda, 1, 1, 1, 1)?;

        let expected = x_cpu.conv1d(&w_cpu, 1, 1, 1, 1)?;

        assert_tensor_close(&got, &expected, 0.03, 0.03)?;

        let got = x_cuda.conv_transpose1d(&w_cuda, 1, 0, 1, 1, 1)?;

        let expected = x_cpu.conv_transpose1d(&w_cpu, 1, 0, 1, 1, 1)?;

        assert_tensor_close(&got, &expected, 0.03, 0.03)?;
    }

    {
        let input = [
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ];

        let kernel = [0.5f32, -1.0, 2.0, 0.25];

        let x_cuda = Tensor::from_slice(&input, (1, 1, 4, 4), &cuda)?.to_dtype(DType::BF16)?;

        let w_cuda = Tensor::from_slice(&kernel, (1, 1, 2, 2), &cuda)?.to_dtype(DType::BF16)?;

        let x_cpu = Tensor::from_slice(&input, (1, 1, 4, 4), &cpu)?;

        let w_cpu = Tensor::from_slice(&kernel, (1, 1, 2, 2), &cpu)?;

        let got = x_cuda.conv2d(&w_cuda, 0, 1, 1, 1)?;

        let expected = x_cpu.conv2d(&w_cpu, 0, 1, 1, 1)?;

        assert_tensor_close(&got, &expected, 0.05, 0.03)?;

        let got = x_cuda.conv_transpose2d(&w_cuda, 0, 0, 1, 1)?;

        let expected = x_cpu.conv_transpose2d(&w_cpu, 0, 0, 1, 1)?;

        assert_tensor_close(&got, &expected, 0.05, 0.03)?;

        let got = x_cuda.avg_pool2d(2)?;
        let expected = x_cpu.avg_pool2d(2)?;

        assert_tensor_close(&got, &expected, 0.03, 0.03)?;

        let got = x_cuda.max_pool2d(2)?;
        let expected = x_cpu.max_pool2d(2)?;

        assert_tensor_close(&got, &expected, 0.0, 0.0)?;

        let got = x_cuda.upsample_nearest2d(7, 7)?;

        let expected = x_cpu.upsample_nearest2d(7, 7)?;

        assert_tensor_close(&got, &expected, 0.0, 0.0)?;

        let got = x_cuda.upsample_bilinear2d(7, 7, false)?;

        let expected = x_cpu.upsample_bilinear2d(7, 7, false)?;

        assert_tensor_close(&got, &expected, 0.05, 0.03)?;
    }

    Ok(())
}

#[test]
fn cuda_legacy_bf16_elementwise_runtime() -> Result<()> {
    bf16_elementwise_case()?;
    bf16_unary_full_case()?;
    Ok(())
}

#[test]
fn cuda_legacy_bf16_reduction_runtime() -> Result<()> {
    bf16_fast_reduce_case()?;
    bf16_argmin_argmax_case()?;
    bf16_atomic_sum_case()?;
    Ok(())
}

#[test]
fn cuda_legacy_bf16_transformer_ops_runtime() -> Result<()> {
    bf16_softmax_case()?;
    bf16_norm_case()?;
    bf16_rope_case()?;
    Ok(())
}

#[test]
fn cuda_legacy_bf16_memory_ops_runtime() -> Result<()> {
    bf16_memory_ops_case()?;
    Ok(())
}

#[test]
fn cuda_legacy_bf16_indexing_runtime() -> Result<()> {
    bf16_where_case()?;
    bf16_sort_case()?;
    bf16_index_select_gather_case()?;
    bf16_index_add_case()?;
    bf16_scatter_case()?;
    Ok(())
}

#[test]
fn cuda_legacy_bf16_matmul_runtime() -> Result<()> {
    bf16_matmul_case()?;
    Ok(())
}

#[test]
fn cuda_legacy_bf16_conv_runtime() -> Result<()> {
    bf16_conv_case()?;
    Ok(())
}
